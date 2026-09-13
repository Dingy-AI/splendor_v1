import pickle
import numpy as np
from splendor_v1.env.env import SplendorEnv
from collections import Counter, defaultdict


# with open("checkpoints/heuristic_pretrain/heuristic_replay_buffer_combined.pkl", "rb") as f:
#     data = pickle.load(f)


# with open("splendor_v1/training/data/model_replay_buffer.pkl", "rb") as f:
#     data = pickle.load(f)


with open("checkpoints_archived/fixed_seed_420_yolo/replay_1409_games.pkl", "rb") as f:
    data = pickle.load(f)



print(data.keys())
print("Buffer length:", len(data["buffer"]))
print("Capacity:", data["capacity"])
print("Position:", data["position"])

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




# ============================================================
# ACTION TYPE STATISTICS
# ============================================================

env = SplendorEnv()

picked_action_types = Counter()
policy_mass_by_type = defaultdict(float)

total_positions = len(buffer)


for obs, policy, value in buffer:

    # --------------------------------------------------------
    # 1. Count the action MCTS would actually pick
    # --------------------------------------------------------

    best_action_id = int(np.argmax(policy))

    best_action = env.id_to_action(
        best_action_id
    )

    picked_action_types[
        best_action.action_type
    ] += 1


    # --------------------------------------------------------
    # 2. Count total policy probability mass by action type
    # --------------------------------------------------------

    nonzero_ids = np.flatnonzero(policy)

    for action_id in nonzero_ids:

        action = env.id_to_action(
            int(action_id)
        )

        policy_mass_by_type[
            action.action_type
        ] += float(
            policy[action_id]
        )


# ============================================================
# PRINT ARGMAX ACTION COUNTS
# ============================================================

print()
print("=" * 60)
print("ARGMAX ACTION TYPES")
print("=" * 60)

for action_type, count in (
    picked_action_types.most_common()
):

    percentage = (
        count
        / total_positions
        * 100
    )

    print(
        f"{action_type.name:20s} "
        f"{count:8d} "
        f"{percentage:6.2f}%"
    )


# ============================================================
# PRINT AVERAGE POLICY MASS
# ============================================================

print()
print("=" * 60)
print("AVERAGE MCTS POLICY MASS BY ACTION TYPE")
print("=" * 60)

for action_type, total_mass in sorted(
    policy_mass_by_type.items(),
    key=lambda x: x[1],
    reverse=True,
):

    average_mass = (
        total_mass
        / total_positions
    )

    print(
        f"{action_type.name:20s} "
        f"{average_mass:8.4f} "
        f"{average_mass * 100:6.2f}%"
    )