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
