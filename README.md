# splendor_v1

# creating the environment
python3 -m venv .venv


Initial Splendor Agent and Environment

Linux: source .venv/bin/activate


Activate virtual environment: source .venv/Scripts/activate

Install requirements: pip install -r requirements.txt

python -m splendor_v1.evaluation.run_evaluation

pytest -s

python -m pytest splendor_v1/tests/test_19_observation_size.py

python -m splendor_v1.scripts.run_training

# creates a function call profile of function time allocation
python -m cProfile -o training_profile.prof splendor_v1/scripts/run_training.py

# reads the function call profile 
python splendor_v1/scripts/read_profile.py

# compares the old function and new function 
python -m splendor_v1.scripts.script_benchmark_legal_buy_reserved


# logger tool
tensorboard --logdir=checkpoints_logger
http://localhost:6006

# splendor pre-training generator
python -m splendor_v1.training.generate_training_set


# splendor pre-training 
python -m splendor_v1.training.train_heuristic_pretrain --replay splendor_v1/training/data/heuristic_replay_buffer_2.pkl --epochs 5 --batch-size 32


# splendor pre-training continue
 python -m splendor_v1.training.train_heuristic_pretrain --replay splendor_v1/training/data/heuristic_replay_buffer.pkl --epochs 10 --batch-size 32 --resume checkpoints/heuristic_pretrain/heuristic_pretrain_epoch_05.pt

  python -m splendor_v1.training.train_heuristic_pretrain --replay splendor_v1/training/data/heuristic_replay_buffer_2.pkl --epochs 10 --batch-size 32 --resume checkpoints/heuristic_pretrain/heuristic_pretrain_epoch_05.pt

# combine pkl files
python -m splendor_v1.training.combine_pkl_files


we have a new cycle ->

# does the model training
python -m splendor_v1.training.model_generate_training_set

# does the model replay
 python -m splendor_v1.training.train_model_replay --replay checkpoints/heuristic_pretrain/model_replay_buffer.pkl --epochs 30 --resume checkpoints/heuristic_pretrain/m1_model.pt --reset-optimizer

 and repeat until we cant :o


# heuristic pretrain split script
python -m splendor_v1.training.train_heuristic_pretrain_split \
    --replay \
    splendor_v1/training/data/h16/no_split/h16_old_tagged.pkl \
    splendor_v1/training/data/h16/h16_with_split.pkl \
    --output-dir checkpoints/blake_h16


 # Model Name List
Avery → Gen 1
Blake → Gen 2
Casey → Gen 3
Drew → Gen 4
Emery → Gen 5
Finley → Gen 6
Gray → Gen 7
Harper → Gen 8
Jordan → Gen 9
Kai → Gen 10
Logan → Gen 11
Morgan → Gen 12
Noel → Gen 13
Parker → Gen 14
Quinn → Gen 15
Riley → Gen 16
Sage → Gen 17
Taylor → Gen 18

# Optuna run for G2H12
 python -m splendor_v1.training_v2.optuna_hpo_blake --replay "splendor_v1/training_v2/data/h12/all_h12_replay_buffers_combined.pkl" --split-mode legacy-contiguous --val-fraction 0.10 --trials 30 --epochs 12 --study-name "blake_g2h12_hpo_v1" --storage "sqlite:///blake_g2h12_hpo_v1.db" --results-dir "hpo/blake_g2h12_hpo_v1" --device cuda


 # new training run
 python -m  splendor_v1.training_v2.run_training_v2_mini

 # script to train model 3 using model 2 data with new WDL head. 
python -m splendor_v1.training_v3.pretrain_model_3_wdl_legacy_h12 --replay "splendor_v1/training/data/h12/all_h12_replay_buffers_combined.pkl" --head-epochs 10 --joint-epochs 30


# model 4 using model 2 data with WDL Head and action scorer
python -m splendor_v1.training_v4.pretrain_model_4_legacy_h12

# Extended command

python -m splendor_v1.training_v4.pretrain_model_4_legacy_h12 `
    --policy-warmup-epochs 20 `
    --joint-epochs 10 `
    --batch-size 32 `
    --policy-lr 1e-3 `
    --joint-lr 1e-4 `
    --grad-clip 1.0

# rich fine-tuning with updated database
python -m splendor_v1.training_v4.finetune_model_4_rich_replay `
    --replay splendor_v1/training_v4/data/replay_200_games.pkl `
    --epochs 5 `
    --batch-size 256 `
    --learning-rate 1e-4 `
    --grad-clip 1.0

# training v5 command
python -m splendor_v1.training_v5.run_training_v5

to resume training set 
RESUME_TRAINING = True
and make sure the variables below are correct

RESUME_CHECKPOINT_PATH = (
    "checkpoints/gen_4/self_play_v5_pruning/"
    "model_100_games_last.pt"
)

RESUME_REPLAY_PATH = (
    "splendor_v1/training_v5/data/"
    "replay_buffer_model4_mcts_v5_pruning.pkl"
)


# batch replay profiler - purpose is to look at how batching in the neural evaluation could speed up simulation time
python -m splendor_v1.training_v5.profile_v5_performance \
    --skip-game-profile \
    --batch-sizes 1 2 4 8 16 32 64 128

python -m splendor_v1.training_v5.profile_v5_performance \
    --profile-games 3
    --batch-sizes 1 2 4 8 16 32 64 128

# confirming that our current neural evaluator and the direct neural evaluator have matching game states
# based on model_1900_games.py

python -m splendor_v1.mcts_batched.smoke_test_direct_evaluator

python -m splendor_v1.mcts_batched.smoke_test_direct_evaluator --simulations 400 --num-mcts-states 5

# testing the 3 way batching vs direct vs original
python -m splendor_v1.mcts_batched.smoke_test_three_way_evaluator

python -m splendor_v1.mcts_batched.smoke_test_three_way_evaluator --simulations 400 --mcts-states 5

#batch speed testing with direct comparison
 python -m splendor_v1.mcts_batched.concurrent_self_play_smoke           

# batch testing speed without direct
python -m splendor_v1.mcts_batched.concurrent_self_play_smoke --games 16 --concurrent-games 16 --max-batch-size 16 --skip-direct

python -m splendor_v1.mcts_batched.concurrent_self_play_smoke     --games 64     --concurrent-games 64     --max-batch-size 32     --skip-direct     --output splendor_v1/training_v5/data/concurrent_batching_64_games_batch32.json

# smoke test for multiprocess inference single
python -m splendor_v1.mcts_batched.smoke_test_multiprocess_inference

# smoke test for multiprocess parallel play
python -m splendor_v1.mcts_batched.smoke_test_multiprocess_self_play

# training_v6 infrastructure
Current Model 4
      ↓
save inference snapshot
      ↓
move parent model + Adam state to CPU
      ↓
┌────────────────────────────────────┐
│ parallel self-play                 │
│                                    │
│ 6 CPU game/MCTS processes          │
│          ↓                         │
│ centralized GPU Model 4 process    │
│          ↓                         │
│ whole games returned to MAIN       │
│          ↓                         │
│ persistent ReplayBuffer.add_game() │
└────────────────────────────────────┘
      ↓
GPU server shuts down
      ↓
move training model + optimizer to GPU
      ↓
train_network()
      ↓
validate_network()
      ↓
checkpoint
      ↓
save new inference weights
      ↓
next iteration

# run_training 

load Model 4 + replay
        ↓
save inference snapshot
        ↓
6 CPU self-play processes
        ↓
central GPU inference server
        ↓
10 real games complete
        ↓
MAIN commits them to replay
        ↓
GPU server shuts down
        ↓
parent Model 4 returns to GPU
        ↓
train_network()
        ↓
validate_network()
        ↓
checkpoint / replay save


# main computer
32 workers

# secondary computer ?
| Evaluation games | If A wins 52.5% | If A wins 55% | If A wins 57.5% | If A wins 60% |
|---:|---:|---:|---:|---:|
| 50 | 55.6% | 76.0% | 83.9% | 89.9% |
| 100 | 61.8% | 81.6% | 90.3% | 97.2% |
| 200 | 73.8% | **91.1%** | 98.0% | 99.7% |
| 400 | 82.9% | **97.4%** | 99.84% | 99.996% |
| 500 | 84.8% | **98.6%** | 99.96% | 99.9996% |
| 1,000 | 93.9% | **99.91%** | ~100% | ~100% |
| 2,000 | 98.7% | ~100% | ~100% | ~100% |

# single game debugger
python -m splendor_v1.training_v6.debug_single_self_play_game

# parallel evaluation against heuristic/random (NOTE we need to becareful to not overheat the computer)
 python -m splendor_v1.evaluation.run_parallel_baseline_evaluation_v6   --model splendor_v1/training_v6/data/model_4000_games.pt   --opponent h16   --games 100   --workers 3   --gpu-max-batch-size 3  --simulations 400   --min-simulations 80   --check-interval 20   --target-visits-per-action 20   --single-action-simulations 4   --stability-checks 3   --h16-rollouts 8  --output-json splendor_v1/evaluation/results/model4000_vs_h16_200.json    

# Comparison
Baseline
      Simulation Pruning
      Parallelization on CPU and GPU 
      200/250 Games / HR approximation at 20 - 32 Workers


Rust_v1 
      400 games / HR approximation at 32 workers 

# Rust Benchmark
python -m splendor_v1.rust_engine.benchmark_self_play --games 16 --workers 16 --device cuda --report splendor_v1/rust_engine/native_benchmark.json

# Stress test
python -m splendor_v1.rust_engine.benchmark_self_play --games 100 --workers 32 --batch-size 32 --device cuda --report splendor_v1/rust_engine/native_benchmark.json

# Rebuilding the Rust
python -m maturin develop --release --manifest-path splendor_v1/rust_engine/Cargo.toml

# RUST RUN
python -m splendor_v1.rust_engine.run_training

# RUST_V2 Rebuild
python -m maturin develop --release --manifest-path splendor_v1/rust_engine_v2/Cargo.toml

# RUST_V2 pytest 
python -m pytest splendor_v1/rust_engine_v2 -q

# RUST_V2 game result comparison # goes up to 750 games /hr
python -m splendor_v1.rust_engine_v2.benchmark_self_play --games 100 --workers 32 --batch-size 32 --device cuda --precision fp32 --report splendor_v1/rust_engine_v2/benchmark_fp32_32.json

# Rust_V2 game reesult 2,436 / hr

python -m splendor_v1.rust_engine_v2.benchmark_self_play --games 512 --workers 128 --batch-size 128 --device cuda --precision fp32 --report splendor_v1/rust_engine_v2/benchmark_fp32_128.json

# Rust_V2 game result 3,800 / hr
python -m splendor_v1.rust_engine_v2.benchmark_self_play --games 1024 --workers 256 --batch-size 256 --device cuda --precision fp32 --report splendor_v1/rust_engine_v2/benchmark_fp32_256.json

# RUST_V2 Reduced precision Test
python -m splendor_v1.rust_engine_v2.profile_inference --device cuda --precisions fp32 fp16 bf16 --report splendor_v1/rust_engine_v2/inference_gpu_profile.json

# RUST_V2 training 
python -m splendor_v1.rust_engine_v2.run_training


# 1209
python -m splendor_v1.rust_engine_v2.benchmark_self_play --games 100 --workers 64 --batch-size 64 --device cuda --precision fp32 --report splendor_v1/rust_engine_v2/benchmark_fp32_64.json

# 
python -m splendor_v1.rust_engine_v2.benchmark_self_play --games 512 --workers 128 --batch-size 128 --device cuda --precision fp32 --report splendor_v1/rust_engine_v2/benchmark_fp32_128.json

# OFFICIAL RUN COMMAND NOW make sure we are changing the v6 run_training_v6.py to match. Do not run the run_training_v6.py directly now or IT WILL FRY YOUR COMPUTER
python -m splendor_v1.rust_engine_v2.run_training --precision fp32