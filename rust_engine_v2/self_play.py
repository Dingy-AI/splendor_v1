"""Rich replay at real-decision boundaries; simulations/scheduling stay native."""
import copy
import os
import time
from types import SimpleNamespace

from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts_batched.multiprocess_self_play_worker import (
    SelfPlayGameResult, summarize_search_samples,
)
from splendor_v1.rust_engine.self_play import NativeReplayRecorder
from splendor_v1.rust_engine.failures import EmptyActionError
from splendor_v1.rust_engine_v2 import from_python, to_python
from splendor_v1.rust_engine_v2.arena import ArenaSearch
from splendor_v1.rust_engine_v2.self_play_settings import EXECUTION


class GameSession:
    def __init__(self, arena, slot, job, config, inference_metadata):
        self.started = time.perf_counter()
        self.slot, self.job, self.config = slot, job, config
        self.inference_metadata = inference_metadata
        env = SplendorEnv(num_players=2)
        env.reset(seed=int(job.seed))
        self.search = ArenaSearch(arena, slot, from_python(env.state), config.search, job.seed)
        self.recorder = NativeReplayRecorder(env, self.search, config.search)
        self.trajectory = []
        self.sample = None
        self.start_decision()

    def start_decision(self):
        if len(self.trajectory) >= self.config.search.max_game_steps:
            raise RuntimeError(f"Native self-play exceeded maximum game length: seed={self.job.seed}, "
                               f"max_game_steps={self.config.search.max_game_steps}")
        try:
            self.sample = self.recorder.capture(len(self.trajectory))
        except EmptyActionError as exc:
            exc.add_context(self.job, self.config, self.trajectory, "rust_v2")
            exc.diagnostic["inference"] = self.inference_metadata
            raise
        noise = self.recorder.get_add_root_noise(self.recorder.env.state)
        self.sample["root_noise_enabled"] = noise
        self.search.begin(add_root_noise=noise)
        if not self.search.active:
            raise RuntimeError(f"No legal actions in nonterminal game: seed={self.job.seed}")

    def finish_decision(self):
        self.recorder.add_search_data(self.sample, self.search.summary)
        temperature = self.recorder.get_temperature(self.recorder.env.state)
        action_id = self.search.select_action(temperature)
        if action_id not in self.recorder.actions_by_id:
            raise RuntimeError(f"Search selected an illegal action: seed={self.job.seed}")
        self.recorder.record_chosen_action(self.sample, self.recorder.actions_by_id[action_id])
        self.sample["temperature"] = temperature
        reward, terminated = self.search.advance(action_id)
        self.recorder.record_transition(self.sample, reward, terminated)
        self.trajectory.append(self.sample)
        self.sample = None
        if terminated:
            return self.finish_game()
        self.start_decision()
        return None

    def finish_game(self):
        final_state = to_python(self.search.root_state())
        if not self.trajectory[-1]["terminated_after_action"] or not final_state.winners:
            raise RuntimeError("Game did not finish with a valid winner/final transition")
        extra = dict(model_generation=self.config.model_generation,
            model_checkpoint=self.config.model_checkpoint_label, search=self.recorder.get_search_metadata())
        if self.job.extra_game_metadata:
            extra.update(copy.deepcopy(self.job.extra_game_metadata))
        extra.update(self_play_execution=EXECUTION, execution_backend="rust_v2",
            multiprocess_self_play=False, multiprocess_coordinator=False,
            worker_id=self.slot, worker_pid=os.getpid(), inference=self.inference_metadata)
        metadata = self.recorder.build_game_metadata(state=final_state, seed=int(self.job.seed),
            source="model_self_play", num_positions=len(self.trajectory),
            split=self.job.split if self.job.split is not None else self.config.split,
            extra_metadata=extra)
        return SelfPlayGameResult(success=True, worker_id=self.slot, pid=os.getpid(),
            game_id=self.job.game_id, seed=self.job.seed, seconds=time.perf_counter() - self.started,
            num_positions=len(self.trajectory), game_metadata=metadata,
            search_summary=summarize_search_samples(SimpleNamespace(buffer=self.trajectory)),
            samples=self.trajectory)
