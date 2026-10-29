"""
EWC paper-style experiment (Kirkpatrick 2017 Architecture)
==========================================================
Reproduces the layout of Fig. 2A from Kirkpatrick et al. (2017)
Adapted for Rotated MNIST (0, 45, 90 degrees) on full dataset.

This script TRAINS the three methods and writes the raw accuracy curves to
    ewc_rotated_mnist_data.csv
It then renders the figure via plot_ewc.py. Because the data is saved to CSV,
you can re-make/tweak the plot WITHOUT re-running training:  python plot_ewc.py

Performance notes:
- The whole dataset is rotated ONCE and kept resident on the GPU, so there is
  no per-epoch CPU rotation and no per-batch host->device copy. This removes the
  data-loading bottleneck entirely (DataLoader/num_workers no longer needed).
"""

import os
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import random_split

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
SEED            = 42
BATCH           = 64
HIDDEN          = 2000     # MATCHES PAPER: 2000 neurons per layer
EPOCHS_PER_TASK = 30
LR              = 1e-3
EVALS_PER_EPOCH = 1
EVAL_BATCH      = 2000     # large eval batch is fine on GPU (no dropout in eval)
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cross-Validation Search Grids
LAMBDA_EWC_GRID = [10e5, 10e6, 10e7, 10e8, 10e9, 10e10]
LAMBDA_L2_GRID  = [0.001, 0.005, 0.01, 0.05, 0.1]

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

# ──────────────────────────────────────────────────────────────
# DATA  —  3 Rotated-MNIST tasks (0°, 45°, 90°)
# ──────────────────────────────────────────────────────────────
print(f"Loading Full MNIST on {DEVICE}...")

# HARDCODED CLUSTER PATH
HERE      = '/iopsstor/scratch/cscs/tperez/cluster/'
DATA_ROOT = os.path.join(HERE, 'data')

# Base transform: ToTensor only. We rotate BEFORE normalization so empty corners
# are filled with actual 0.0 (black), preventing gray artifacts.
base_tf = transforms.ToTensor()

# download=True fetches the dataset if not present (do this from a login node;
# compute nodes have no internet).
train_full_raw = datasets.MNIST(DATA_ROOT, train=True,  download=True, transform=base_tf)
test_full_raw  = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=base_tf)

# Split 60k train into 50k train / 10k val
g = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(train_full_raw, [50000, 10000], generator=g)
test_ds = test_full_raw

# Task Angles: A=0, B=45, C=90
angles = [0, 45, 90]

# ── Pre-rotate ONCE and hold everything on the GPU ────────────────────────────
_normalize = transforms.Normalize((0.1307,), (0.3081,))

def materialize(subset, angle):
    """Rotate + normalize + flatten the whole split once -> (X, y) on DEVICE."""
    xs, ys = [], []
    for img, lbl in subset:
        if angle != 0:
            img = TF.rotate(img, angle, fill=0.0)
        xs.append(_normalize(img).view(-1))
        ys.append(lbl)
    X = torch.stack(xs).to(DEVICE)
    y = torch.tensor(ys, device=DEVICE)
    return X, y

print("Pre-rotating datasets (one-time, ~minutes)...")
train_sets = [materialize(train_ds, a) for a in angles]   # [(X,y), ...] per angle
val_sets   = [materialize(val_ds,   a) for a in angles]
test_sets  = [materialize(test_ds,  a) for a in angles]
print("Datasets resident on", DEVICE)

def iter_minibatches(X, y, bs, shuffle, gen=None):
    n = X.shape[0]
    if shuffle:
        idx = torch.randperm(n, device=X.device, generator=gen)
        for i in range(0, n, bs):
            sel = idx[i:i + bs]
            yield X[sel], y[sel]
    else:
        for i in range(0, n, bs):
            yield X[i:i + bs], y[i:i + bs]

# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────
def make_model():
    """MATCHES PAPER: 2 Hidden layers, 2000 units, 50% Dropout"""
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Linear(784, HIDDEN),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(HIDDEN, 10),
    ).to(DEVICE)

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    correct = 0
    for i in range(0, X.shape[0], EVAL_BATCH):
        pred = model(X[i:i + EVAL_BATCH]).argmax(1)
        correct += (pred == y[i:i + EVAL_BATCH]).sum().item()
    return correct / X.shape[0]

def snapshot(model):
    return {n: p.clone().detach() for n, p in model.named_parameters()}

def compute_fisher(model, X, n_samples=1000):
    fish = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    n = min(n_samples, X.shape[0])
    for i in range(n):
        xi = X[i].unsqueeze(0)
        model.zero_grad()
        log_probs = F.log_softmax(model(xi), dim=1)
        y_sample = torch.multinomial(log_probs.exp().detach(), num_samples=1).squeeze(1)
        F.nll_loss(log_probs, y_sample).backward()
        for nm, p in model.named_parameters():
            if p.grad is not None:
                fish[nm] += p.grad.detach() ** 2
    return {nm: v / n for nm, v in fish.items()}

# ──────────────────────────────────────────────────────────────
# TRAINING & CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────
def train_candidate(start_model, train_X, train_y, method, lam, anchor, ewc_memories):
    model = copy.deepcopy(start_model)
    opt = optim.Adam(model.parameters(), lr=LR)

    n_batches  = (train_X.shape[0] + BATCH - 1) // BATCH
    eval_every = max(1, n_batches // EVALS_PER_EPOCH)
    phase_accs = [[] for _ in range(3)]
    gen = torch.Generator(device=train_X.device).manual_seed(SEED)

    for epoch in range(EPOCHS_PER_TASK):
        model.train()
        for batch_i, (xb, yb) in enumerate(iter_minibatches(train_X, train_y, BATCH, True, gen)):
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)

            if method == 'l2' and anchor is not None:
                pen = sum((p - anchor[n]).pow(2).sum() for n, p in model.named_parameters())
                loss += (lam / 2) * pen

            if method == 'ewc' and len(ewc_memories) > 0:
                pen = 0.0
                for old_anchor, old_fisher in ewc_memories:
                    for n, p in model.named_parameters():
                        pen += (old_fisher[n] * (p - old_anchor[n]).pow(2)).sum()
                loss += (lam / 2) * pen

            loss.backward()
            opt.step()

            if (batch_i + 1) % eval_every == 0 or batch_i == n_batches - 1:
                for tid in range(3):
                    phase_accs[tid].append(accuracy(model, *test_sets[tid]))
                model.train()

    return model, phase_accs

def run(method):
    model = make_model()
    anchor       = None
    ewc_memories = []
    accuracies   = [[] for _ in range(3)]

    for phase in range(3):
        train_X, train_y = train_sets[phase]
        grid = [0.0]

        if phase > 0:
            if method == 'ewc':
                grid = LAMBDA_EWC_GRID
            elif method == 'l2':
                grid = LAMBDA_L2_GRID

        best_val_acc = -1.0
        best_candidate_state = None
        best_phase_accs = None
        best_lam = None

        for lam in grid:
            candidate_model, phase_accs = train_candidate(
                model, train_X, train_y, method, lam, anchor, ewc_memories
            )

            val_accs = [accuracy(candidate_model, *val_sets[t]) for t in range(phase + 1)]
            avg_val_acc = sum(val_accs) / len(val_accs)

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_candidate_state = candidate_model.state_dict()
                best_phase_accs = phase_accs
                best_lam = lam

        model.load_state_dict(best_candidate_state)
        for tid in range(3):
            accuracies[tid].extend(best_phase_accs[tid])

        current_snapshot = snapshot(model)

        if method == 'l2':
            anchor = current_snapshot

        if method == 'ewc':
            new_f = compute_fisher(model, train_X)
            ewc_memories.append((current_snapshot, new_f))

        lam_str = f" (λ={best_lam})" if phase > 0 and method != 'sgd' else ""
        print(f"  [{method.upper():>3}] finished phase {phase + 1}{lam_str:^16} | "
              f"Task Accs: " + ", ".join(f"{accuracies[t][-1]:.3f}" for t in range(3)))

    return accuracies

# ──────────────────────────────────────────────────────────────
# RUN ALL THREE METHODS
# ──────────────────────────────────────────────────────────────
results = {}
for method in ['sgd', 'l2', 'ewc']:
    print(f"\nTraining {method.upper()} with Cross-Validation ...")
    results[method] = run(method)
print("\nAll runs complete.\n")

# ──────────────────────────────────────────────────────────────
# SAVE RAW DATA -> CSV  (so plotting can be rerun independently)
# ──────────────────────────────────────────────────────────────
csv_path = os.path.join(HERE, 'ewc_rotated_mnist_data.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['step', 'method', 'task', 'accuracy'])
    for m in ['sgd', 'l2', 'ewc']:
        for tid in range(3):
            for step, acc in enumerate(results[m][tid], start=1):
                w.writerow([step, m, tid, f"{acc:.6f}"])
print(f"Saved data -> {csv_path}")

# ──────────────────────────────────────────────────────────────
# RENDER PLOT  (single source of truth lives in plot_ewc.py)
# ──────────────────────────────────────────────────────────────
out_path = os.path.join(HERE, 'ewc_rotated_mnist_plot.png')
try:
    from plot_ewc import plot_from_csv
    plot_from_csv(csv_path, out_path)
    print(f"Saved plot -> {out_path}")
except Exception as e:
    print(f"[warn] plot step skipped: {e}\n       Re-make it any time with:  python plot_ewc.py")
