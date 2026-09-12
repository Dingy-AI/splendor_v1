# splendor_v1
Initial Splendor Agent and Environment

python -m venv .venv

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