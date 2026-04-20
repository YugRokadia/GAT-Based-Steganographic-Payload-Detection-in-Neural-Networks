import torch
import struct
import numpy as np
import random
import os

# -----------------------------
# FLOAT <-> BITS
# -----------------------------
def float_to_bits(f):
    return struct.unpack('I', struct.pack('f', f))[0]

def bits_to_float(b):
    return struct.unpack('f', struct.pack('I', b))[0]

# -----------------------------
# PAYLOAD
# -----------------------------
def generate_payload(n_bits):
    return np.random.choice([0, 1], size=n_bits, p=[0.65, 0.35])

# -----------------------------
# EMBEDDING (2–3 BIT RANDOM)
# -----------------------------
def embed_bits(val, payload, idx, total_bits):

    bits = float_to_bits(val)

    n = random.choice([2, 3])

    if idx + n > total_bits:
        n = total_bits - idx

    new_bits = 0
    for i in range(n):
        new_bits |= (int(payload[idx + i]) << i)

    mask = (1 << n) - 1
    bits = (bits & ~mask) | new_bits

    return bits_to_float(bits), idx + n

# -----------------------------
# TAMPER FUNCTION (WITH LABEL TRACKING)
# -----------------------------
def tamper_layers(state_dict, target_layers, payload):

    idx = 0
    total_bits = len(payload)

    layer_labels = {}

    for key in state_dict:

        tensor = state_dict[key]

        if tensor.dtype != torch.float32:
            continue

        flat = tensor.view(-1)

        if key not in target_layers:
            layer_labels[key] = 0
            continue

        num_weights = flat.shape[0]
        num_modify = int(0.3 * num_weights)

        indices = np.random.choice(num_weights, size=num_modify, replace=False)

        modified = False

        for i in indices:

            if idx >= total_bits:
                break

            val = flat[i].item()

            new_val, idx = embed_bits(val, payload, idx, total_bits)

            flat[i] = new_val
            modified = True

        layer_labels[key] = 1 if modified else 0

    return state_dict, layer_labels

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def tamper_model(input_path):

    base_name = input_path.replace(".pth", "")

    original = torch.load(input_path)

    layer_keys = [
        k for k in original
        if original[k].dtype == torch.float32
    ]

    total_weights = sum(original[k].numel() for k in layer_keys)
    payload_size = int(total_weights * 0.10)

    print(f"\nProcessing: {input_path}")
    print(f"Payload bits: {payload_size}")

    # =============================
    # FULL TAMPERING
    # =============================
    payload_full = generate_payload(payload_size)

    full_state, full_labels = tamper_layers(
        torch.load(input_path),
        layer_keys,
        payload_full
    )

    full_path = base_name + "_full_tampered.pth"

    torch.save(full_state, full_path)

    np.save(full_path.replace(".pth", "_labels.npy"), full_labels)

    print("✔ Full tampered + labels saved")

    # =============================
    # PARTIAL TAMPERING
    # =============================
    payload_partial = generate_payload(payload_size)

    num_partial = max(1, len(layer_keys) // 2)
    partial_layers = random.sample(layer_keys, num_partial)

    partial_state, partial_labels = tamper_layers(
        torch.load(input_path),
        partial_layers,
        payload_partial
    )

    partial_path = base_name + "_partial_tampered.pth"

    torch.save(partial_state, partial_path)

    np.save(partial_path.replace(".pth", "_labels.npy"), partial_labels)

    print("✔ Partial tampered + labels saved")

# -----------------------------
# RUN
# -----------------------------
models = [
    "cnn_seed_137.pth",
    "cnn_seed_421.pth",
    "cnn_seed_982.pth",
    "cnn_seed_1543.pth",
    "cnn_seed_2761.pth",
    "cnn_seed_312.pth",
    "cnn_seed_777.pth",
    "cnn_seed_999.pth",
    "cnn_seed_2024.pth",
    "cnn_seed_5555.pth",
    "cnn_seed_8888.pth",
    "cnn_seed_4321.pth",
    "cnn_seed_6789.pth",
    "cnn_seed_2468.pth",
    "cnn_seed_1357.pth",
    "cnn_seed_9090.pth",
    "cnn_seed_8080.pth",
    "cnn_seed_7070.pth",
    "cnn_seed_6060.pth",
    "cnn_seed_5050.pth"
]

for m in models:
    tamper_model(m)

print("\nAll tampered models generated 🚀")