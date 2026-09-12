import pickle
import numpy as np
from splendor_v1.env.env import SplendorEnv


# with open("checkpoints_archived/fixed_seed_420_yolo/replay_3600_games.pkl", "rb") as f:
#     data = pickle.load(f)


with open("splendor_v1/training/data/heuristic_replay_buffer.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print("Type:", type(data))

if hasattr(data, "__len__"):
    print("Length:", len(data))

if isinstance(data, (list, tuple)):
    print("First item type:", type(data[0]))
    print("First item:", data[0])

elif isinstance(data, dict):
    print("Keys:", data.keys())
    for key, value in data.items():
        print(key, type(value))


buffer = data["buffer"]
sample = buffer[0]

print("Sample type:", type(sample))

if isinstance(sample, (tuple, list)):
    print("Sample length:", len(sample))

    for i, item in enumerate(sample):
        print(f"\nItem {i}")
        print("Type:", type(item))

        if hasattr(item, "shape"):
            print("Shape:", item.shape)
        else:
            print("Value:", item)


obs, policy, value = buffer[0]

print("Observation shape:", obs.shape)

print("Policy shape:", policy.shape)
print("Policy sum:", policy.sum())
print("Nonzero actions:", np.count_nonzero(policy))
print("Max probability:", policy.max())
print("Best action id:", policy.argmax())

print("Value:", value)


top_ids = np.argsort(policy)[-10:][::-1]

for action_id in top_ids:
    print(
        action_id,
        policy[action_id]
    )


for i in range(5):
    obs, policy, value = buffer[i]

    print(
        i,
        "sum =", policy.sum(),
        "nonzero =", np.count_nonzero(policy),
        "max =", policy.max(),
        "value =", value
    )


top_ids = np.argsort(policy)[-10:][::-1]


env = SplendorEnv()

for action_id in top_ids:
    if policy[action_id] > 0:
        action = env.id_to_action(action_id)

        print(
            action_id,
            policy[action_id],
            action
        )