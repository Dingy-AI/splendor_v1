# splendor_v1
Initial Splendor Agent and Environment

Activate virtual environment: source .venv/Scripts/activate

Install requirements: pip install -r requirements.txt

python -m splendor_v1.evaluation.run_evaluation

pytest -s

python -m pytest splendor_v1/tests/test_19_observation_size.py

python -m splendor_v1.scripts.run_training

python -m cProfile -o training_profile.prof splendor_v1/scripts/run_training.py

python splendor_v1/scripts/read_profile.py