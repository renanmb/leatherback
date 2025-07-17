import os
import pandas as pd
import numpy as np
import onnxruntime as ort

script_dir = os.path.dirname(__file__)
# relative_path = os.path.join("..", "leatherback")
# full_path = os.path.abspath(os.path.join(script_dir, relative_path))
# usd_path = os.path.abspath(os.path.join(full_path, "leatherback_simple_better.usd"))
policy_file_path = os.path.join(script_dir, "../logs/skrl/leatherback_direct/2025-07-16_16-38-13_ppo_torch/checkpoints/exported/policy_agent.onnx")
log_file_path = os.path.join(script_dir, "../leatherback_logs/env_18_obs.csv")

# Load CSV
df = pd.read_csv(log_file_path)

# Extract inputs and logged actions
input_cols = [
    "position_error",
    "target_heading_cos",
    "target_heading_sin",
    "root_lin_vel_x",
    "root_lin_vel_y",
    "root_ang_vel_z",
    "throttle_state",
    "steering_state",
]

# print(f"Loaded {len(df)} rows from CSV")

# Extract inputs for ONNX model
inputs = df[input_cols].to_numpy().astype(np.float32)

# Extract logged actions
logged_actions = df[["action_throttle", "action_steering"]].to_numpy().astype(np.float32)

session = ort.InferenceSession(policy_file_path)
# ort_inputs = {session.get_inputs()[0].name: obs.numpy()}
output_names = [output.name for output in session.get_outputs()]
# input_names = [input.name for input in session.get_inputs()]
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# print("ONNX output names:", output_names) # output_names = actions
# outputs = self.session.run(output_names, ort_inputs)
# predicted_actions = session.run([output_name], {input_name: inputs})[0]
# predicted_actions = session.run(output_names, {input_name: inputs})[0]

# print("Input shape:", inputs.shape)
# print("First input sample:", inputs[0] if len(inputs) > 0 else "No inputs")

predicted_actions = []

for i in range(inputs.shape[0]):
    input_sample = inputs[i:i+1]  # shape: (1, 8)
    ort_inputs = {session.get_inputs()[0].name: input_sample}
    # output = session.run(output_names, input_names)
    # output = session.run(output_names, {input_name: input_sample})[0]
    outputs = session.run(output_names, ort_inputs)
    action = outputs[0].reshape(-1)
    # action = outputs[0]
    # print(outputs)
    # output = session.run([output_name], {input_name: input_sample})[0]
    # predicted_actions.append(output[0])  # shape: (2,) or whatever your output is
    predicted_actions.append(action)

predicted_actions = np.array(predicted_actions)

# Get output and flatten to 1D array like .view(-1).numpy()
# action = outputs[0].reshape(-1)

# Element-wise difference
differences = np.abs(predicted_actions - logged_actions)

# Mean squared error or other metric
mse = np.mean((predicted_actions - logged_actions) ** 2, axis=0)

# Print some example comparisons
for i in range(min(5, len(inputs))):
    print(f"\nSample {i+1}")
    print(f"Logged     : {logged_actions[i]}")
    print(f"Predicted  : {predicted_actions[i]}")
    print(f"Difference : {differences[i]}")

# # Get the last 10 indices (or fewer if not enough samples)
# for i in range(max(0, len(inputs) - 10), len(inputs)):
#     print(f"\nSample {i+1}")
#     print(f"Logged     : {logged_actions[i]}")
#     print(f"Predicted  : {predicted_actions[i]}")
#     print(f"Difference : {differences[i]}")

import matplotlib.pyplot as plt

plt.plot(logged_actions[:, 0], label="Logged Throttle", linestyle="--")
plt.plot(predicted_actions[:, 0], label="Predicted Throttle")
plt.legend()
plt.title("Throttle: Logged vs Predicted")
plt.show()

plt.plot(logged_actions[:, 1], label="Logged Steering", linestyle="--")
plt.plot(predicted_actions[:, 1], label="Predicted Steering")
plt.legend()
plt.title("Steering: Logged vs Predicted")
plt.show()