import torch
import numpy as np
import glob
import os
from scipy.stats import entropy

# -----------------------------
# FLOAT → UINT32 (FAST)
# -----------------------------
def floats_to_bits(arr):
    return arr.view(np.uint32)

# -----------------------------
# SAFE ENTROPY
# -----------------------------
def safe_entropy(probs):
    probs = np.array(probs) + 1e-12
    return entropy(probs, base=2)

# -----------------------------
# RUN LENGTH
# -----------------------------
def run_length(bits):
    if len(bits) == 0:
        return 0

    changes = np.where(bits[:-1] != bits[1:])[0]

    if len(changes) == 0:
        return len(bits)

    run_lengths = np.diff(np.concatenate(([-1], changes, [len(bits)-1])))
    return np.mean(run_lengths)

# -----------------------------
# FEATURE EXTRACTION (PER LAYER)
# -----------------------------
def extract_features(tensor):

    flat = tensor.detach().view(-1).cpu().numpy().astype(np.float32)

    if len(flat) < 2:
        return [0] * 22   # ✅ FIXED SIZE

    bits = floats_to_bits(flat)

    # -----------------------------
    # BIT LEVEL
    # -----------------------------
    lsb  = (bits >> 0) & 1
    lsb2 = (bits >> 1) & 1
    lsb3 = (bits >> 2) & 1

    p0 = np.mean(lsb == 0)
    p1 = np.mean(lsb == 1)

    ent1 = safe_entropy([p0, p1])
    imbalance = abs(p0 - p1)

    transition_rate = np.mean(lsb[:-1] != lsb[1:])
    run_len = run_length(lsb)

    ent2 = safe_entropy([
        np.mean(lsb2 == 0),
        np.mean(lsb2 == 1)
    ])

    ent3 = safe_entropy([
        np.mean(lsb3 == 0),
        np.mean(lsb3 == 1)
    ])

    # -----------------------------
    # BIT PAIRS
    # -----------------------------
    bit_pairs = (lsb2 << 1) | lsb

    pair_probs = [
        np.mean(bit_pairs == i) for i in range(4)
    ]

    # -----------------------------
    # BYTE LEVEL
    # -----------------------------
    bytes_arr = bits.view(np.uint8).reshape(-1, 4)

    byte_entropy = []
    for i in range(4):
        hist = np.bincount(bytes_arr[:, i], minlength=256)
        prob = hist / len(bytes_arr)
        byte_entropy.append(safe_entropy(prob))

    # -----------------------------
    # FLOAT STRUCTURE
    # -----------------------------
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF

    exp_mean = np.mean(exponent)
    exp_std  = np.std(exponent)
    mantissa_std = np.std(mantissa)

    # -----------------------------
    # WEIGHT STATS
    # -----------------------------
    mean = np.mean(flat)
    std = np.std(flat)
    diff_std = np.std(np.diff(flat))
    abs_mean = np.mean(np.abs(flat))
    zero_frac = np.mean(flat == 0)

    # -----------------------------
    # FINAL FEATURE VECTOR (22)
    # -----------------------------
    return [
        ent1,
        imbalance,
        transition_rate,
        run_len,
        ent2,
        ent3,

        mean,
        std,
        diff_std,
        abs_mean,
        zero_frac,

        exp_mean,
        exp_std,
        mantissa_std,

        *pair_probs,
        *byte_entropy
    ]

# -----------------------------
# MODEL FEATURES (ORDER SAFE)
# -----------------------------
def extract_model_features(model_path):

    state_dict = torch.load(model_path)

    features = []
    keys = []

    for key in state_dict:
        tensor = state_dict[key]

        if tensor.dtype == torch.float32:
            features.append(extract_features(tensor))
            keys.append(key)

    return np.array(features), keys, state_dict

# -----------------------------
# LOAD LABELS (ORDER SAFE)
# -----------------------------
def load_labels(model_path, keys):

    label_path = model_path.replace(".pth", "_labels.npy")

    if not os.path.exists(label_path):
        return None

    label_dict = np.load(label_path, allow_pickle=True).item()

    labels = []

    for key in keys:
        labels.append(label_dict.get(key, 0))   # safe fallback

    return np.array(labels)

# -----------------------------
# BUILD DATASET
# -----------------------------
models = sorted(glob.glob("cnn_seed_*.pth"))

dataset = []

print("\nExtracting features...\n")

for path in models:

    feats, keys, state_dict = extract_model_features(path)

    labels = load_labels(path, keys)

    if labels is not None:
        dataset.append({
            "features": feats,
            "node_labels": labels
        })
    else:
        label = 1 if "tampered" in path else 0

        dataset.append({
            "features": feats,
            "label": label
        })

    print(f"Processed: {path} | Shape: {feats.shape}")

# -----------------------------
# GLOBAL NORMALIZATION
# -----------------------------
all_features = np.vstack([d["features"] for d in dataset])

mean = all_features.mean(axis=0)
std = all_features.std(axis=0) + 1e-8

for d in dataset:
    d["features"] = (d["features"] - mean) / std

# -----------------------------
# SAVE
# -----------------------------
np.save("graph_dataset.npy", dataset, allow_pickle=True)

print("\nDataset saved as graph_dataset.npy 🚀")