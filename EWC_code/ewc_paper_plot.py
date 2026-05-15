"""
EWC paper-style plot
====================
Reproduces the layout of Fig. 2A from Kirkpatrick et al. (2017) on
permuted-MNIST: three stacked subplots (one per task) showing the
fraction-correct of EWC, L2 and plain SGD across a single training
timeline that goes through `train A`, `train B`, `train C`.

The hyper-parameters below have been tuned so that:
    • SGD shows clear catastrophic forgetting on old tasks,
    • L2 protects old tasks moderately but slowly drifts,
    • EWC keeps task A and task B near their peak accuracy
      while still learning new tasks.

Run:
    python ewc_paper_plot.py
Outputs:
    ewc_paper_plot.png  (saved next to this script)
"""

import os
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
BATCH           = 256
HIDDEN          = 600      # widened from 400. Larger hidden layers expand the
                           # soft-Fisher subspace where EWC is free to learn
                           # new tasks, so B and C plateaus rise.
EPOCHS_PER_TASK = 15       # raised from 12. EWC converges more slowly than
                           # SGD because it optimises inside a constrained
                           # subspace; extra iterations let it reach deeper
                           # minima inside that subspace on the new task.
LR              = 1e-3
LAMBDA_EWC      = 1000.0   # EWC penalty (Fisher-weighted). With the sharper
                           # Fisher (n_samples=3000) and the *true* Fisher
                           # (sampled labels) used in compute_fisher, even the
                           # small-importance weights have meaningful F_i, so
                           # huge lambdas (2000+) over-constrain globally and
                           # hurt new-task learning. ~1000 is the sweet spot.
LAMBDA_L2       = 3.0      # L2 penalty per past snapshot. Lowered from 30 so
                           # the cumulative penalty (×1 in phase 2, ×2 in
                           # phase 3) no longer freezes the network: L2 can
                           # now drift away from the task-A anchor (more
                           # forgetting on A) and the unconstrained directions
                           # are free enough to fit B and C reasonably well.
                           # Trade-off: EWC's visual lead over L2 shrinks
                           # because L2 moves closer to a plain-SGD baseline.
N_TRAIN_SUBSET  = 15_000   # subset of MNIST train (keeps run < ~3 min on CPU)
EVALS_PER_EPOCH = 4        # how often to evaluate during each epoch
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────
# DATA  —  3 permuted-MNIST tasks
# ──────────────────────────────────────────────────────────────
print("Loading MNIST ...")
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, 'data')
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_full = datasets.MNIST(DATA_ROOT, train=True,  download=True, transform=tf)
test_full  = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)

# subset for speed
g = torch.Generator().manual_seed(SEED)
train_subset_idx = torch.randperm(len(train_full), generator=g)[:N_TRAIN_SUBSET].tolist()
train_ds         = Subset(train_full, train_subset_idx)
test_ds          = test_full

# Three pixel permutations — one per task, all drawn at random as in
# Kirkpatrick et al. (2017). Task A is no longer the un-permuted MNIST,
# which removes a slight asymmetry (un-permuted pixels give the network's
# random init an unfair head start on task A relative to B and C).
perms = [
    torch.randperm(784, generator=g),
    torch.randperm(784, generator=g),
    torch.randperm(784, generator=g),
]

def make_loader(ds, perm, shuffle):
    """Return a DataLoader that flattens images and applies `perm` to pixels."""
    def collate(batch):
        x = torch.stack([img.view(-1)[perm] for img, _ in batch])
        y = torch.tensor([lbl for _, lbl in batch])
        return x, y
    return DataLoader(ds, BATCH, shuffle=shuffle, collate_fn=collate)

train_loaders = [make_loader(train_ds, p, True)  for p in perms]
test_loaders  = [make_loader(test_ds,  p, False) for p in perms]

# ──────────────────────────────────────────────────────────────
# MODEL — 2-hidden-layer MLP, shared across all 3 tasks
# ──────────────────────────────────────────────────────────────
def make_model():
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Linear(784,    HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
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

def compute_fisher(model, loader, n_samples=5000):
    """
    Diagonal of the *true* Fisher information matrix (Kirkpatrick et al.):
        F_i = E_x  E_{y ~ p_theta(y|x)} [ (d log p_theta(y|x) / d theta_i )^2 ]
    estimated as the per-sample average of squared gradients, where the
    label is *sampled from the model's predicted distribution* rather than
    being taken from the ground truth. This avoids the empirical-Fisher
    bias that under-estimates importance at confidently-predicted samples
    (where the gradient of -log p(y_true|x) collapses near zero) and gives
    a sharper, more honest map of which weights matter for the task.
    """
    fish = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    seen = 0
    model.eval()
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        for xi in xb:                                   # one sample at a time
            if seen >= n_samples:
                break
            model.zero_grad()
            log_probs = F.log_softmax(model(xi.unsqueeze(0)), dim=1)
            # sample y from the model's predicted distribution
            y_sample = torch.multinomial(log_probs.exp().detach(),
                                         num_samples=1).squeeze(1)
            F.nll_loss(log_probs, y_sample).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fish[n] += p.grad.detach() ** 2     # per-sample squared grad
            seen += 1
        if seen >= n_samples:
            break
    return {n: v / seen for n, v in fish.items()}

# ──────────────────────────────────────────────────────────────
# TRAINING — one full sequential run for a given method
# ──────────────────────────────────────────────────────────────
def run(method):
    model = make_model()

    snaps        = []     # L2  : list of past snapshots (one per finished task)
    fisher_accum = None   # EWC : running sum of Fisher diagonals
    anchor       = None   # EWC : params at end of last finished task

    # accuracies[task_id] = list of test-accs, one entry per evaluation step
    accuracies = [[] for _ in range(3)]

    for phase, train_loader in enumerate(train_loaders):
        # Re-create the optimiser at every phase boundary so Adam's running
        # 1st/2nd-moment estimates start fresh. Carrying them over from the
        # previous task makes the very first steps on the new loss + new
        # penalty wildly mis-scaled (e.g. EWC briefly *over*-pulls toward the
        # anchor, so task-A accuracy spuriously goes UP at the boundary).
        opt = optim.Adam(model.parameters(), lr=LR)

        # how often to evaluate inside an epoch
        n_batches   = len(train_loader)
        eval_every  = max(1, n_batches // EVALS_PER_EPOCH)

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            for batch_i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)

                # ── L2 : uniform pull-back to every past snapshot ──
                if method == 'l2' and snaps:
                    pen = 0.0
                    for snap in snaps:
                        for n, p in model.named_parameters():
                            pen = pen + (p - snap[n]).pow(2).sum()
                    loss = loss + (LAMBDA_L2 / 2) * pen

                # ── EWC : Fisher-weighted pull-back to last snapshot ──
                if method == 'ewc' and anchor is not None:
                    pen = 0.0
                    for n, p in model.named_parameters():
                        pen = pen + (fisher_accum[n] *
                                     (p - anchor[n]).pow(2)).sum()
                    loss = loss + (LAMBDA_EWC / 2) * pen

                loss.backward()
                opt.step()

                # evaluate at regular intra-epoch intervals
                if (batch_i + 1) % eval_every == 0 or batch_i == n_batches - 1:
                    for tid in range(3):
                        accuracies[tid].append(accuracy(model, test_loaders[tid]))
                    model.train()

        # ── End of phase : update regularisation state ──
        snap = snapshot(model)
        snaps.append(snap)            # L2 keeps every past snapshot
        anchor = snap                 # EWC always anchors to latest

        if method == 'ewc':
            new_f = compute_fisher(model, train_loader)
            fisher_accum = new_f if fisher_accum is None else \
                {n: fisher_accum[n] + new_f[n] for n in new_f}

        print(f"  [{method.upper():>3}]  finished phase {phase + 1}  "
              f"task accs = "
              + ", ".join(f"{accuracies[t][-1]:.3f}" for t in range(3)))

    return accuracies

# ──────────────────────────────────────────────────────────────
# RUN ALL THREE METHODS
# ──────────────────────────────────────────────────────────────
results = {}
for method in ['sgd', 'l2', 'ewc']:
    print(f"\nTraining {method.upper()} ...")
    results[method] = run(method)
print("\nAll runs complete.\n")

# ──────────────────────────────────────────────────────────────
# PLOT  —  paper-style 3 stacked rows, single training timeline
# ──────────────────────────────────────────────────────────────
n_steps_total       = len(results['sgd'][0])             # total #evaluations
steps_per_phase     = n_steps_total // 3                 # evals per task
phase_boundaries_x  = [steps_per_phase, 2 * steps_per_phase]  # vertical dashed lines

x = np.arange(1, n_steps_total + 1)

method_color = {'sgd': '#3a6fa0', 'l2': '#5a9b5a', 'ewc': '#c14242'}
method_label = {'sgd': 'SGD',     'l2': r'L$_2$', 'ewc': 'EWC'}

fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.2), sharex=True)

for tid, ax in enumerate(axes):
    # task `tid` has not been seen until phase `tid` begins
    start = tid * steps_per_phase

    for method in ['sgd', 'l2', 'ewc']:
        accs = np.array(results[method][tid])
        ax.plot(
            x[start:], accs[start:],
            color=method_color[method],
            linewidth=2,
            label=method_label[method] if tid == 0 else None,
        )

    # phase boundaries
    for b in phase_boundaries_x:
        ax.axvline(b + 0.5, color='gray', linestyle='--',
                   linewidth=1, alpha=0.7)

    # cosmetics
    # NB: y-axis extends down to 0 so that L2's collapse on the newly
    # introduced task is visible — with the cumulative L2 penalty it
    # often falls well below the 0.8 cutoff used in the original paper.
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1.0'])
    ax.axhline(0.1, color='gray', linewidth=0.7, linestyle=':',
               alpha=0.5)   # chance level (10 classes)
    ax.set_ylabel(f"Task {chr(ord('A') + tid)}", fontsize=12, rotation=90,
                  labelpad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_xticks([])
    ax.set_xlim(0.5, n_steps_total + 0.5)

# ── phase labels above the top panel ──
top_ax = axes[0]
phase_centres = [steps_per_phase * (i + 0.5) for i in range(3)]
for cx, label in zip(phase_centres, ['train A', 'train B', 'train C']):
    top_ax.text(cx, 1.05, label, ha='center', va='bottom',
                fontsize=12, transform=top_ax.get_xaxis_transform())

# ── method labels on the right of the top panel (paper style) ──
last_x = n_steps_total + 0.5
last_y = {m: results[m][0][-1] for m in ['sgd', 'l2', 'ewc']}
# nudge labels apart so they don't overlap
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

# ── shared axis labels ──
axes[-1].set_xlabel("Training time", fontsize=12)
#fig.text(0.04, 0.07, "Frac. correct", ha='left', va='bottom', fontsize=11)

plt.subplots_adjust(left=0.13, right=0.94, top=0.92, bottom=0.10, hspace=0.35)

out_path = os.path.join(HERE, 'ewc_paper_plot.png')
plt.savefig(out_path, dpi=170, bbox_inches='tight')
plt.show()
print(f"Saved -> {out_path}")
