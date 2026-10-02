"""Native game/search execution with the existing Model 4/V6 rich replay schema."""
import copy
import json
import os
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np

from splendor_v1.env.core.enums import ActionType
from splendor_v1.env.core.cost_lookup_table_v3 import T3_PAYMENT_LOOKUP
from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts_batched.multiprocess_self_play_worker import (
    SelfPlayGameResult, summarize_search_samples,
)
from splendor_v1.rust_engine import from_python, to_python
from splendor_v1.rust_engine.mcts import RustMCTS
from splendor_v1.rust_engine.failures import EmptyActionError, empty_action_error
from splendor_v1.training_v2.state_serializer import serialize_state
from splendor_v1.training_v5.model_replay_generator_v5_pruning import ModelReplayGenerator

EXECUTION = "rust_threaded_cross_game_gpu_batching"


def native_search_settings(search_config, adaptive_simulations=True):
    settings = asdict(search_config)
    settings.pop("max_game_steps")
    settings["adaptive_simulations"] = bool(adaptive_simulations)
    return settings


class NativeReplayRecorder(ModelReplayGenerator):
    """Reuse replay formatting/validation; never execute Python rules or MCTS.

    Python state objects exist only at replay boundaries. Legal IDs, observations,
    rewards, termination, moves, and search all come from the native backend.
    """

    def __init__(self, env, native_search, search_config, adaptive_simulations=True):
        settings = native_search_settings(search_config, adaptive_simulations)
        metadata = SimpleNamespace(**settings, rollout_type="neural", selection_type="puct")
        super().__init__(env=env, mcts=metadata, replay_buffer=None,
            state_serializer=serialize_state, temperature=1.0,
            temperature_fn=lambda turn: 1.0 if turn < 20 else 0.5 if turn < 40 else 0.0,
            add_root_noise=True, root_noise_fn=lambda turn: turn < 40,
            teacher_mode=False, action_space_version=1, max_game_steps=search_config.max_game_steps)
        self.native_search = native_search
        self._native_state = None
        self._snapshot = None
        self._actions = None

    def _semantic_action(self, action_id, state):
        action = self.env.id_to_action(int(action_id))
        if action.action_type == ActionType.BUY_VISIBLE:
            card = state.visible_cards[action.tier][action.slot]
            table = self.env._get_tier_payment_lookup(card.tier)
        elif action.action_type == ActionType.BUY_RESERVED:
            card = state.players[state.current_player].reserved_cards[action.reserved_index]
            table = T3_PAYMENT_LOOKUP
        else:
            return action
        gold = self.env.map_payment_to_card(table[action.payment_id], self.env.get_color_mapping(card))
        return replace(action, gold_payment=gold)

    def capture(self, step_index):
        self._native_state = self.native_search.root_state()
        self.env.state = to_python(self._native_state)
        self._snapshot = json.loads(self._native_state.snapshot_json())
        for field in ("visible_card_ids", "deck_card_ids"):
            self._snapshot[field] = {int(t): row for t, row in self._snapshot[field].items()}
        ids = self.native_search.native.root_action_ids()
        if not ids:
            raise empty_action_error(self.env, self._snapshot, step_index)
        self._actions = [self._semantic_action(i, self.env.state) for i in ids]
        self.actions_by_id = dict(zip(ids, self._actions))
        return self.build_base_sample(self.env.state, step_index)

    def encode_observation(self, state):
        return np.asarray(self._native_state.observation(), dtype=np.float32)

    def serialize_state(self, state):
        return self._snapshot

    def get_legal_actions(self, state):
        return self._actions

    def add_search_data(self, sample, summary):
        children = [SimpleNamespace(action=self.actions_by_id[c["action_id"]],
            visits=c["visits"], value=c["value"], prior=c["prior"], network_prior=c["network_prior"])
            for c in summary["children"]]
        root = SimpleNamespace(children=children, visits=summary["visits"], value=summary["value"],
                               search_metadata=summary["metadata"])
        sample.update(self.extract_root_statistics(root))
        sample.update(self.extract_pruning_search_metadata(root))
        sample["policy_source"] = "mcts"
        self.validate_search_data(sample)


def run_native_game(*, worker_id, evaluator, job, config, adaptive_simulations=True):
    """Return a complete trajectory. Exceptions never commit an unfinished game."""
    config.validate()
    started = time.perf_counter()
    env = SplendorEnv(num_players=2)
    env.reset(seed=int(job.seed))  # Preserve the original NumPy shuffle exactly.
    search = RustMCTS(from_python(env.state), evaluator=evaluator,
                     **native_search_settings(config.search, adaptive_simulations))
    mcts_seed, action_seed = np.random.SeedSequence(int(job.seed)).spawn(2)
    search.rng = np.random.default_rng(mcts_seed)
    search.action_rng = np.random.default_rng(action_seed)
    recorder = NativeReplayRecorder(env, search, config.search, adaptive_simulations)
    trajectory = []
    for step_index in range(config.search.max_game_steps):
        try:
            sample = recorder.capture(step_index)
        except EmptyActionError as exc:
            exc.add_context(job, config, trajectory, "rust")
            raise
        state = recorder.env.state
        noise = recorder.get_add_root_noise(state)
        sample["root_noise_enabled"] = noise
        search.search(add_root_noise=noise)
        recorder.add_search_data(sample, search.summary)
        temperature = recorder.get_temperature(state)
        action_id = search.select_action(temperature)
        if action_id not in recorder.actions_by_id:
            raise RuntimeError(f"Native search selected an illegal move: seed={job.seed}, decision={step_index}")
        recorder.record_chosen_action(sample, recorder.actions_by_id[action_id])
        sample["temperature"] = temperature
        reward, terminated = search.advance(action_id)
        recorder.record_transition(sample, reward, terminated)
        trajectory.append(sample)
        if terminated:
            break
    else:
        raise RuntimeError(f"Native self-play exceeded maximum game length: seed={job.seed}, "
                           f"max_game_steps={config.search.max_game_steps}")
    final_state = to_python(search.root_state())
    if not trajectory or not trajectory[-1]["terminated_after_action"] or not final_state.winners:
        raise RuntimeError("Native game did not finish with a valid winner and final transition")
    extra = dict(model_generation=config.model_generation,
                 model_checkpoint=config.model_checkpoint_label, search=recorder.get_search_metadata())
    if job.extra_game_metadata:
        extra.update(copy.deepcopy(job.extra_game_metadata))
    extra.update(self_play_execution=EXECUTION, execution_backend="rust",
                 multiprocess_self_play=False, worker_id=int(worker_id), worker_pid=os.getpid())
    metadata = recorder.build_game_metadata(state=final_state, seed=int(job.seed),
        source="model_self_play", num_positions=len(trajectory),
        split=job.split if job.split is not None else config.split, extra_metadata=extra)
    return SelfPlayGameResult(success=True, worker_id=worker_id, pid=os.getpid(),
        game_id=job.game_id, seed=job.seed, seconds=time.perf_counter() - started,
        num_positions=len(trajectory), game_metadata=metadata,
        search_summary=summarize_search_samples(SimpleNamespace(buffer=trajectory)),
        samples=trajectory)
