import pickle
from pprint import pprint


PATH = (
    "splendor_v1/training_v2/data/"
    "test_model2_replay.pkl"
)


with open(PATH, "rb") as f:
    data = pickle.load(f)


print("\n================ TOP LEVEL ================\n")

print(type(data))

if isinstance(data, dict):
    print("Keys:")
    pprint(list(data.keys()))


print("\n================ BASIC INFO ================\n")

for key in [
    "format_version",
    "sample_version",
    "capacity",
    "position",
    "next_game_id",
    "metadata",
]:
    if key in data:
        print(f"{key}:")
        pprint(data[key])
        print()


buffer = data.get(
    "buffer",
    [],
)

games = data.get(
    "games",
    {},
)

game_sample_counts = data.get(
    "game_sample_counts",
    {},
)


print("Replay samples:", len(buffer))
print("Games:", len(games))

print("\nGame sample counts:")
pprint(game_sample_counts)


print("\n================ GAMES ================\n")

for game_id, game in games.items():

    print(f"\n--- GAME {game_id} ---")

    pprint(game)


print("\n================ FIRST SAMPLE ================\n")

if buffer:

    sample = buffer[0]

    print("Sample keys:")
    pprint(list(sample.keys()))

    print("\nFull first sample:")
    pprint(sample)


print("\n================ LAST SAMPLE ================\n")

if buffer:

    pprint(buffer[-1])