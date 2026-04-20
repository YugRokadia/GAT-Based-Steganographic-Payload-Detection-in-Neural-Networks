import torch
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# -----------------------------
# SEED
# -----------------------------
SEED = 99
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset_raw = np.load("graph_dataset.npy", allow_pickle=True)

# -----------------------------
# AUTO FEATURE SIZE
# -----------------------------
input_dim = dataset_raw[0]["features"].shape[1]
print(f"\nDetected feature size: {input_dim}")

graphs = []

print("\nBuilding node-level graphs...\n")

# -----------------------------
# BUILD GRAPHS
# -----------------------------
for item in dataset_raw:

    x = torch.tensor(item["features"], dtype=torch.float)
    num_nodes = x.shape[0]

    # NODE LABELS
    if "node_labels" in item:
        y = torch.tensor(item["node_labels"], dtype=torch.long)
    else:
        if item["label"] == 0:
            y = torch.zeros(num_nodes, dtype=torch.long)
        else:
            y = torch.ones(num_nodes, dtype=torch.long)

    # -----------------------------
    # SEQUENTIAL GRAPH
    # -----------------------------
    edge_index = []

    for i in range(num_nodes - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    graphs.append(Data(x=x, edge_index=edge_index, y=y))

print(f"Total graphs: {len(graphs)}\n")

# -----------------------------
# STRATIFIED SPLIT
# -----------------------------
clean = []
tampered = []

for g in graphs:
    if g.y.sum() == 0:
        clean.append(g)
    else:
        tampered.append(g)

split_clean = int(0.8 * len(clean))
split_tampered = int(0.8 * len(tampered))

train_graphs = clean[:split_clean] + tampered[:split_tampered]
test_graphs  = clean[split_clean:] + tampered[split_tampered:]

random.shuffle(train_graphs)
random.shuffle(test_graphs)

train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
all_labels = []

for item in dataset_raw:
    if "node_labels" in item:
        all_labels.extend(item["node_labels"])
    else:
        all_labels.extend(
            [item["label"]] * item["features"].shape[0]
        )

all_labels = np.array(all_labels)

class_counts = np.bincount(all_labels)
weights = 1.0 / (class_counts + 1e-8)
weights = weights / weights.sum()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.tensor(weights, dtype=torch.float).to(device)

# -----------------------------
# MODEL
# -----------------------------
class NodeGNN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.conv1 = GATConv(input_dim, 32, heads=2)
        self.conv2 = GATConv(64, 32, heads=2)
        self.conv3 = GATConv(64, 16, heads=1)

        self.fc = torch.nn.Linear(16, 2)

    def forward(self, x, edge_index, batch=None):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        return self.fc(x)

model = NodeGNN(input_dim).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005,
    weight_decay=1e-4
)

criterion = torch.nn.CrossEntropyLoss(weight=weights)

# -----------------------------
# TRAINING
# -----------------------------
print("Starting node-level training...\n")

EPOCHS = 30
best_acc = 0

for epoch in range(EPOCHS):

    # -----------------------------
    # TRAIN
    # -----------------------------
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for data in train_loader:

        data = data.to(device)

        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch)

        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = out.argmax(dim=1)

        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    train_acc = correct / total

    # -----------------------------
    # TEST (UPDATED)
    # -----------------------------
    model.eval()

    clean_correct = 0
    clean_total = 0
    tampered_correct = 0
    tampered_total = 0

    with torch.no_grad():
        for data in test_loader:

            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch)

            pred = out.argmax(dim=1)

            for p, y in zip(pred, data.y):
                if y.item() == 0:
                    clean_correct += (p == y).item()
                    clean_total += 1
                else:
                    tampered_correct += (p == y).item()
                    tampered_total += 1

    clean_acc = clean_correct / clean_total if clean_total > 0 else 0
    tampered_acc = tampered_correct / tampered_total if tampered_total > 0 else 0

    test_acc = (clean_correct + tampered_correct) / (clean_total + tampered_total)

    # SAVE BEST
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_node_gnn.pth")

    print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f}")
    print(f"   Clean Acc: {clean_acc:.2f} | Tampered Acc: {tampered_acc:.2f}")

print("\nTraining complete 🚀")
print(f"Best Test Accuracy: {best_acc:.2f}")

# -----------------------------
# CONFUSION MATRIX (NODE LEVEL)
# -----------------------------
print("\nComputing confusion matrix...\n")

model.eval()

TP = 0  # tampered → tampered
TN = 0  # clean → clean
FP = 0  # clean → tampered
FN = 0  # tampered → clean

with torch.no_grad():
    for data in test_loader:

        data = data.to(device)

        out = model(data.x, data.edge_index, data.batch)

        pred = out.argmax(dim=1)

        for p, y in zip(pred, data.y):

            if y.item() == 1 and p.item() == 1:
                TP += 1
            elif y.item() == 0 and p.item() == 0:
                TN += 1
            elif y.item() == 0 and p.item() == 1:
                FP += 1
            elif y.item() == 1 and p.item() == 0:
                FN += 1

# -----------------------------
# METRICS
# -----------------------------
total = TP + TN + FP + FN

accuracy = (TP + TN) / total
precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)
fpr = FP / (FP + TN + 1e-8)
f1 = 2 * (precision * recall) / (precision + recall)

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("Confusion Matrix:")
print(f"TP (Tampered → Tampered): {TP}")
print(f"TN (Clean → Clean):       {TN}")
print(f"FP (Clean → Tampered):    {FP}")
print(f"FN (Tampered → Clean):    {FN}")

print("\nMetrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"FPR      : {fpr:.4f}")
print(f"F1 Score : {f1:.4f}")

# your values
cm = np.array([[76, 12],
               [1, 31]])

plt.figure()
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks([0,1], ["Clean", "Tampered"])
plt.yticks([0,1], ["Clean", "Tampered"])

# numbers inside boxes
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()


print("\nComputing ROC curve...\n")

model.eval()

all_probs = []
all_labels = []

with torch.no_grad():
    for data in test_loader:

        data = data.to(device)

        out = model(data.x, data.edge_index, data.batch)

        probs = torch.softmax(out, dim=1)[:, 1]  # probability of tampered

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

# convert
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# compute ROC
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# -----------------------------
# PLOT
# -----------------------------
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()