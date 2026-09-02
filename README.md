# splendor_v1
Initial Splendor Agent and Environment

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