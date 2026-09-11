from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts.node import Node
from splendor_v1.training.self_play import (
    root_visit_policy,
    select_self_play_action,
)


@dataclass
class LeafRequest:
    observation: np.ndarray
    legal_action_ids: list[int]


@dataclass
class _Metrics:
    search_cpu_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    search_calls: int = 0
    backups: int = 0
    batch_sizes: list[int] = field(default_factory=list)


class TorchBatchEvaluator:
    """Maintains a separate inference copy of the training model."""

    def __init__(self, source_model, device="cuda"):
        import torch

        self.torch = torch
        self.device = torch.device(device)

        if (
            self.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA is unavailable. Use device='cpu' "
                "to test on CPU."
            )

        self.model = deepcopy(source_model).to(
            self.device
        ).eval()

        self.model.requires_grad_(False)

    def sync_from(self, source_model):
        self.model.load_state_dict(
            source_model.state_dict(),
            strict=True,
        )

        self.model.eval()

        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def __call__(self, requests):
        if not requests:
            return []

        torch = self.torch

        lengths = [
            len(request.legal_action_ids)
            for request in requests
        ]

        if min(lengths) == 0:
            raise ValueError(
                "Neural requests must have legal actions."
            )

        batch_size = len(requests)
        width = max(lengths)

        ids = np.zeros(
            (batch_size, width),
            dtype=np.int64,
        )

        valid = np.zeros(
            (batch_size, width),
            dtype=np.bool_,
        )

        for row, request in enumerate(requests):
            length = lengths[row]
            ids[row, :length] = request.legal_action_ids
            valid[row, :length] = True

        parameter = next(self.model.parameters())

        with torch.inference_mode():
            observations = torch.as_tensor(
                np.stack([
                    request.observation
                    for request in requests
                ]),
                dtype=parameter.dtype,
                device=self.device,
            )

            indices = torch.as_tensor(
                ids,
                dtype=torch.long,
                device=self.device,
            )

            mask = torch.as_tensor(
                valid,
                dtype=torch.bool,
                device=self.device,
            )

            # Evaluate the full policy head for the GPU batch.
            logits, values = self.model(observations)

            if (
                logits.ndim != 2
                or logits.shape[0] != batch_size
            ):
                raise ValueError(
                    "Policy output must have shape "
                    "[batch, action_space]."
                )

            if values.numel() != batch_size:
                raise ValueError(
                    "Expected one value per observation."
                )

            legal_logits = logits.gather(
                1,
                indices,
            ).masked_fill(
                ~mask,
                -float("inf"),
            )

            probabilities = torch.softmax(
                legal_logits,
                dim=1,
            )

            policies = probabilities.cpu().tolist()
            value_numbers = values.reshape(-1).cpu().tolist()

        return [
            (policy[:length], value)
            for policy, length, value in zip(
                policies,
                lengths,
                value_numbers,
            )
        ]


def _search_requests(
    env,
    mcts,
    state,
    root,
    metrics,
    add_root_noise,
    teacher_mode,
):
    """Pause at evaluation; finish backup before selecting again."""

    clock = perf_counter()
    metrics.search_calls += 1

    try:
        root_player = state.current_player

        if root is None:
            root = Node(state=state.clone())
        else:
            root.parent = None

            if root.state is None:
                root.state = state.clone()

        if (
            root.state.game_over
            or not mcts.get_legal_actions(env, root)
        ):
            return root

        noise_added = False

        if (
            add_root_noise
            and root.expanded
            and root.children
        ):
            mcts.add_dirichlet_noise(root)
            noise_added = True

        # Reused roots still receive this many NEW simulations.
        for _ in range(mcts.simulations):
            node = mcts.select(
                env,
                root,
                root_player,
            )

            mcts.materialize_state(env, node)

            if (
                node.state.game_over
                or env._check_terminated(node.state)
            ):
                node.expanded = True

                value = mcts.terminal_value(
                    node.state,
                    root_player,
                )

            else:
                legal_actions = mcts.get_legal_actions(
                    env,
                    node,
                )

                if not legal_actions:
                    node.state.game_over = True
                    node.state.winners = []
                    node.expanded = True
                    value = 0.0

                else:
                    if node.expanded:
                        raise RuntimeError(
                            "PUCT returned an expanded "
                            "nonterminal node with legal actions."
                        )

                    if teacher_mode:
                        # Matches your existing teacher-mode behavior.
                        priors = [
                            1.0 / len(legal_actions)
                        ] * len(legal_actions)

                        value = 0.0

                    else:
                        request = LeafRequest(
                            observation=(
                                env.observation_encoder.encoder(
                                    node.state
                                )
                            ),
                            legal_action_ids=[
                                env.action_to_id(action)
                                for action in legal_actions
                            ],
                        )

                        metrics.search_cpu_seconds += (
                            perf_counter() - clock
                        )

                        # Do not count time spent waiting for other games.
                        clock = None

                        priors, value = yield request

                        clock = perf_counter()

                        if len(priors) != len(legal_actions):
                            raise ValueError(
                                "Wrong number of action priors."
                            )

                        if (
                            node.state.current_player
                            != root_player
                        ):
                            value = -value

                    for action, prior in zip(
                        legal_actions,
                        priors,
                    ):
                        node.children.append(
                            Node(
                                state=None,
                                parent=node,
                                action=action,
                                prior=float(prior),
                            )
                        )

                    node.expanded = True

                    if (
                        add_root_noise
                        and node is root
                        and not noise_added
                        and root.children
                    ):
                        mcts.add_dirichlet_noise(root)
                        noise_added = True

            mcts.backup(node, value)
            metrics.backups += 1

        return root

    finally:
        if clock is not None:
            metrics.search_cpu_seconds += (
                perf_counter() - clock
            )


def _play_game(
    env,
    mcts,
    metrics,
    game_index,
    game_seed,
    max_turns,
    temperature,
    add_root_noise,
    teacher_mode,
    collect_debug,
):
    env.reset(seed=game_seed)

    if len(env.state.players) != 2:
        raise ValueError(
            "This runner supports the two-player "
            "root-relative value convention."
        )

    root = None
    history = []
    debug_samples = []

    completed = False
    reason = "max_turns"
    winners = []

    for _ in range(max_turns):
        if env.state.game_over:
            completed = True
            winners = list(env.state.winners)
            reason = None
            break

        state = env.state
        player = state.current_player

        observation = env.observation_encoder.encoder(
            state
        )

        root = yield from _search_requests(
            env,
            mcts,
            state,
            root,
            metrics,
            add_root_noise,
            teacher_mode,
        )

        if not root.children:
            reason = "no_legal_actions"
            break

        action = select_self_play_action(
            root,
            temperature=temperature,
        )

        next_root = next(
            (
                child
                for child in root.children
                if child.action == action
            ),
            None,
        )

        if next_root is None:
            raise RuntimeError(
                "Selected action has no matching root child."
            )

        policy = root_visit_policy(env, root)

        if collect_debug and state.turn_number % 20 == 0:
            debug_samples.append({
                "game_index": game_index,
                "turn": state.turn_number,
                "observation": observation.copy(),
                "target_policy": policy.copy(),
                "legal_action_ids": [
                    env.action_to_id(child.action)
                    for child in root.children
                ],
            })

        history.append(
            (observation, policy, player)
        )

        _, _, terminated, truncated, info = env.step(
            action
        )

        root = next_root
        root.parent = None

        if env.state.current_player != player:
            mcts.flip_tree_values(root)

        if terminated:
            completed = True
            winners = list(info["winners"])
            reason = None
            break

        if truncated:
            reason = "truncated"
            break

    examples = []

    if completed:
        for observation, policy, player in history:
            # Use the same terminal convention as search.
            value = float(
                mcts.terminal_value(env.state, player)
            )

            examples.append(
                (observation, policy, value)
            )

    result = {
        "game_index": game_index,
        "completed": completed,
        "winners": winners,
        "game_length": len(history),
        "positions_added": len(examples),
        "reason": reason,
    }

    return result, examples, debug_samples


class BatchedSelfPlayRunner:
    """One CPU scheduler coordinating independent games and GPU batches."""

    def __init__(
        self,
        mcts,
        *,
        device="cuda",
        env_factory=SplendorEnv,
        max_turns=300,
        temperature=1.0,
        add_root_noise=True,
        evaluator=None,
    ):
        self.mcts = mcts
        self.env_factory = env_factory
        self.max_turns = max_turns
        self.temperature = temperature
        self.add_root_noise = add_root_noise
        self._running = False

        self._validate()

        self.evaluator = (
            evaluator
            if evaluator is not None
            else TorchBatchEvaluator(
                mcts.model,
                device,
            )
        )

    def _validate(self):
        if (
            self.mcts.selection_type != "puct"
            or self.mcts.rollout_type != "neural"
        ):
            raise ValueError(
                "Requires selection_type='puct' "
                "and rollout_type='neural'."
            )

        if self.mcts.model is None:
            raise ValueError("An MCTS model is required.")

        if (
            not isinstance(self.mcts.simulations, int)
            or self.mcts.simulations < 2
        ):
            raise ValueError(
                "Self-play requires at least two simulations."
            )

        if (
            not isinstance(self.max_turns, int)
            or self.max_turns < 1
        ):
            raise ValueError(
                "max_turns must be a positive integer."
            )

        if (
            not np.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError(
                "temperature must be finite and nonnegative."
            )

    def play_batch(
        self,
        replay_buffer,
        *,
        num_games=10,
        start_game_index=0,
        seed=None,
        dynamic_seeding=False,
        teacher_mode=False,
        policy_debug_samples=None,
    ):
        if self._running:
            raise RuntimeError(
                "This runner is already processing a batch."
            )

        self._validate()

        if (
            not isinstance(num_games, int)
            or num_games < 1
        ):
            raise ValueError(
                "num_games must be a positive integer."
            )

        self._running = True

        generators = []
        started = perf_counter()
        metrics = _Metrics()
        results = [None] * num_games

        try:
            sync_start = perf_counter()

            self.evaluator.sync_from(
                self.mcts.model
            )

            sync_time = perf_counter() - sync_start

            envs = [
                self.env_factory()
                for _ in range(num_games)
            ]

            if len({id(env) for env in envs}) != num_games:
                raise ValueError(
                    "env_factory must create a fresh "
                    "environment for every game."
                )

            for i, env in enumerate(envs):
                game_index = start_game_index + i

                game_seed = (
                    seed + game_index
                    if dynamic_seeding and seed is not None
                    else seed
                )

                generators.append(
                    _play_game(
                        env,
                        copy(self.mcts),
                        metrics,
                        game_index,
                        game_seed,
                        self.max_turns,
                        self.temperature,
                        self.add_root_noise,
                        teacher_mode,
                        policy_debug_samples is not None,
                    )
                )

            pending = []

            for i, generator in enumerate(generators):
                try:
                    pending.append(
                        (i, next(generator))
                    )
                except StopIteration as done:
                    results[i] = done.value

            while pending:
                evaluation_start = perf_counter()

                answers = self.evaluator([
                    request
                    for _, request in pending
                ])

                metrics.evaluation_seconds += (
                    perf_counter() - evaluation_start
                )

                if len(answers) != len(pending):
                    raise ValueError(
                        "Expected one evaluation result "
                        "per request."
                    )

                metrics.batch_sizes.append(
                    len(pending)
                )

                next_pending = []

                for (i, _), answer in zip(
                    pending,
                    answers,
                ):
                    try:
                        request = generators[i].send(
                            answer
                        )

                        next_pending.append(
                            (i, request)
                        )

                    except StopIteration as done:
                        results[i] = done.value

                pending = next_pending

            # Write completed histories in game-index order.
            # Evaluation errors occur before replay insertion.
            for _, examples, debug_samples in results:
                for example in examples:
                    replay_buffer.add(example)

                if policy_debug_samples is not None:
                    policy_debug_samples.extend(
                        debug_samples
                    )

            game_results = [
                result[0]
                for result in results
            ]

            sizes = metrics.batch_sizes

            return {
                "games_attempted": num_games,
                "games_completed": sum(
                    result["completed"]
                    for result in game_results
                ),
                "positions_added": sum(
                    result["positions_added"]
                    for result in game_results
                ),
                "game_results": game_results,
                "batch_time": perf_counter() - started,
                "mcts_time": (
                    metrics.search_cpu_seconds
                    + metrics.evaluation_seconds
                ),
                "search_cpu_time": (
                    metrics.search_cpu_seconds
                ),
                "evaluation_time": (
                    metrics.evaluation_seconds
                ),
                "weight_sync_time": sync_time,
                "search_calls": metrics.search_calls,
                "simulations_completed": metrics.backups,
                "neural_evaluations": sum(sizes),
                "neural_batches": len(sizes),
                "mean_batch_size": (
                    sum(sizes) / len(sizes)
                    if sizes else 0.0
                ),
                "min_batch_size": (
                    min(sizes) if sizes else 0
                ),
                "max_batch_size": (
                    max(sizes) if sizes else 0
                ),
            }

        finally:
            for generator in generators:
                generator.close()

            self._running = False