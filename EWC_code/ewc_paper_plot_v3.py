"""
EWC paper-style plot (Kirkpatrick 2017 Architecture)
====================================================
Reproduces the layout of Fig. 2A from Kirkpatrick et al. (2017) on
permuted-MNIST. 

Key Features:
- 2 Hidden Layers with 2000 neurons each and 50% Dropout.
- Sequential cross-validation for hyperparameter tuning.
- Exact EWC implementation (Fisher and Anchor for EVERY past task).
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
SEED            = 42
BATCH           = 64
HIDDEN          = 2000     # MATCHES PAPER: 2000 neurons per layer
EPOCHS_PER_TASK = 30      
LR              = 1e-3
N_TRAIN_SUBSET  = 12_000   
N_VAL_SUBSET    = 3_000    
EVALS_PER_EPOCH = 1     
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cross-Validation Search Grids
LAMBDA_EWC_GRID = [10.0, 100.0, 1000.0, 10000.0, 50000.0, 100000.0]
LAMBDA_L2_GRID  = [0.001, 0.005, 0.01, 0.05, 0.1]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────
# DATA  —  3 permuted-MNIST tasks
# ──────────────────────────────────────────────────────────────
print(f"Loading MNIST on {DEVICE}...")
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, 'data')
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_full = datasets.MNIST(DATA_ROOT, train=True,  download=True, transform=tf)
test_full  = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)

# Subsets for train, validation, and speed
g = torch.Generator().manual_seed(SEED)
all_train_idx    = torch.randperm(len(train_full), generator=g).tolist()
train_subset_idx = all_train_idx[:N_TRAIN_SUBSET]
val_subset_idx   = all_train_idx[N_TRAIN_SUBSET:N_TRAIN_SUBSET + N_VAL_SUBSET]

train_ds = Subset(train_full, train_subset_idx)
val_ds   = Subset(train_full, val_subset_idx)

N_TEST_SUBSET   = 2000  
test_subset_idx = torch.randperm(len(test_full), generator=g)[:N_TEST_SUBSET].tolist()
test_ds         = Subset(test_full, test_subset_idx)

perms = [
    torch.randperm(784, generator=g),
    torch.randperm(784, generator=g),
    torch.randperm(784, generator=g),
]

def make_loader(ds, perm, shuffle):
    def collate(batch):
        x = torch.stack([img.view(-1)[perm] for img, _ in batch])
        y = torch.tensor([lbl for _, lbl in batch])
        return x, y
    return DataLoader(ds, BATCH, shuffle=shuffle, collate_fn=collate)

train_loaders = [make_loader(train_ds, p, True)  for p in perms]
val_loaders   = [make_loader(val_ds, p, False)   for p in perms]
test_loaders  = [make_loader(test_ds,  p, False) for p in perms]

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
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total   += len(yb)
    return correct / total

def snapshot(model):
    return {n: p.clone().detach() for n, p in model.named_parameters()}

def compute_fisher(model, loader, n_samples=1000):
    fish = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    seen = 0
    model.eval()
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        for xi in xb:
            if seen >= n_samples:
                break
            model.zero_grad()
            log_probs = F.log_softmax(model(xi.unsqueeze(0)), dim=1)
            y_sample = torch.multinomial(log_probs.exp().detach(), num_samples=1).squeeze(1)
            F.nll_loss(log_probs, y_sample).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fish[n] += p.grad.detach() ** 2
            seen += 1
        if seen >= n_samples:
            break
    return {n: v / seen for n, v in fish.items()}

# ──────────────────────────────────────────────────────────────
# TRAINING & CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────
def train_candidate(start_model, train_loader, method, lam, anchor, ewc_memories):
    model = copy.deepcopy(start_model)
    opt = optim.Adam(model.parameters(), lr=LR)
    
    n_batches   = len(train_loader)
    eval_every  = max(1, n_batches // EVALS_PER_EPOCH)
    phase_accs  = [[] for _ in range(3)]

    for epoch in range(EPOCHS_PER_TASK):
        model.train()
        for batch_i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
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
                    phase_accs[tid].append(accuracy(model, test_loaders[tid]))
                model.train()

    return model, phase_accs

def run(method):
    model = make_model()
    anchor       = None   
    ewc_memories = []     
    accuracies   = [[] for _ in range(3)]

    for phase, train_loader in enumerate(train_loaders):
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
                model, train_loader, method, lam, anchor, ewc_memories
            )
            
            val_accs = [accuracy(candidate_model, val_loaders[t]) for t in range(phase + 1)]
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
            new_f = compute_fisher(model, train_loader)
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
# PLOT 
# ──────────────────────────────────────────────────────────────
n_steps_total       = len(results['sgd'][0])
steps_per_phase     = n_steps_total // 3
phase_boundaries_x  = [steps_per_phase, 2 * steps_per_phase]

x = np.arange(1, n_steps_total + 1)

method_color = {'sgd': '#3a6fa0', 'l2': '#5a9b5a', 'ewc': '#c14242'}
method_label = {'sgd': 'SGD',     'l2': r'L$_2$', 'ewc': 'EWC'}

fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.2), sharex=True)

for tid, ax in enumerate(axes):
    start = tid * steps_per_phase

    for method in ['sgd', 'l2', 'ewc']:
        accs = np.array(results[method][tid])
        ax.plot(
            x[start:], accs[start:],
            color=method_color[method],
            linewidth=2,
            label=method_label[method] if tid == 0 else None,
        )

    for b in phase_boundaries_x:
        ax.axvline(b + 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_ylim(0.75, 1.02)
    ax.set_yticks([0.75, 1.0])
    ax.set_yticklabels(['0.75', '1.0'])
    ax.axhline(0.1, color='gray', linewidth=0.7, linestyle=':', alpha=0.5)
    ax.set_ylabel(f"Task {chr(ord('A') + tid)}", fontsize=12, rotation=90, labelpad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_xticks([])
    ax.set_xlim(0.5, n_steps_total + 0.5)

top_ax = axes[0]
phase_centres = [steps_per_phase * (i + 0.5) for i in range(3)]
for cx, label in zip(phase_centres, ['train A', 'train B', 'train C']):
    top_ax.text(cx, 1.05, label, ha='center', va='bottom',
                fontsize=12, transform=top_ax.get_xaxis_transform())

last_x = n_steps_total + 0.5
last_y = {m: results[m][0][-1] for m in ['sgd', 'l2', 'ewc']}
ys_sorted = sorted(last_y.items(), key=lambda kv: kv[1])
min_gap = 0.025
adjusted = []
prev_y = -1
for m, y in ys_sorted:
    y = max(y, prev_y + min_gap)
    adjusted.append((m, y))
    prev_y = y
for m, y in adjusted:
    top_ax.text(last_x + 0.5, y, method_label[m],
                color=method_color[m], fontsize=12, fontweight='bold',
                va='center', ha='left')

axes[-1].set_xlabel("Training time", fontsize=12)

plt.subplots_adjust(left=0.13, right=0.94, top=0.92, bottom=0.10, hspace=0.35)

out_path = os.path.join(HERE, 'ewc_cv_paper_plot_v3.png')
plt.savefig(out_path, dpi=170, bbox_inches='tight')
plt.show()
print(f"Saved -> {out_path}")