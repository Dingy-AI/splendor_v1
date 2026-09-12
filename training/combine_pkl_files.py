import pickle


buffer_path_1 = (
    "splendor_v1/training/data/"
    "heuristic_replay_buffer.pkl"
)

buffer_path_2 = (
    "splendor_v1/training/data/"
    "heuristic_replay_buffer_2.pkl"
)

output_path = (
    "splendor_v1/training/data/"
    "heuristic_replay_buffer_combined.pkl"
)


# --------------------------------------------------
# Load both buffers
# --------------------------------------------------

with open(buffer_path_1, "rb") as f:
    data_1 = pickle.load(f)

with open(buffer_path_2, "rb") as f:
    data_2 = pickle.load(f)


buffer_1 = data_1["buffer"]
buffer_2 = data_2["buffer"]


print("Buffer 1:", len(buffer_1))
print("Buffer 2:", len(buffer_2))


# --------------------------------------------------
# Combine
# --------------------------------------------------

combined_buffer = (
    buffer_1
    + buffer_2
)


print(
    "Combined:",
    len(combined_buffer)
)


# --------------------------------------------------
# Create new replay-buffer state
# --------------------------------------------------

combined_data = {
    "capacity": len(combined_buffer),
    "buffer": combined_buffer,

    # Buffer is currently full.
    # Next inserted sample would replace index 0.
    "position": 0,
}


# --------------------------------------------------
# Save
# --------------------------------------------------

with open(output_path, "wb") as f:
    pickle.dump(
        combined_data,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


print(
    "Saved combined replay buffer to:",
    output_path,
)


with open(output_path, "rb") as f:
    test_data = pickle.load(f)

print(
    "Combined length:",
    len(test_data["buffer"])
)

obs, policy, value = test_data["buffer"][0]

print("Observation:", obs.shape)
print("Policy:", policy.shape)
print("Value:", value)