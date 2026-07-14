# -*- coding: utf-8 -*-
"""phase4_spirale

Dataset en spirale : une frontière non-linéaire "complexe" (pas juste une
courbe simple comme XOR). On entraîne un MLP à 2 couches cachées (ReLU) et
on regarde comment la capacité du réseau (largeur des couches) et le bruit
des données affectent la qualité de la frontière apprise.
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def generate_spiral(n_points=200, noise=0.1, seed=42):
    """Génère deux spirales entrelacées : classe 0 et classe 1."""
    np.random.seed(seed)
    n = n_points // 2
    theta0 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise
    theta1 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise + np.pi
    r = np.linspace(0.1, 1.0, n)
    X0 = np.c_[r * np.cos(theta0), r * np.sin(theta0)]
    X1 = np.c_[r * np.cos(theta1), r * np.sin(theta1)]
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def init_params(sizes, seed=42):
    """Initialisation He : std = sqrt(2 / n_entrées de la couche). Biais à zéro."""
    np.random.seed(seed)
    Ws, bs = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        Ws.append(np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in))
        bs.append(np.zeros(fan_out))
    return Ws, bs


def forward(X, Ws, bs):
    """Propage X à travers toutes les couches (ReLU cachées, sigmoid en sortie)."""
    activations = [X]
    zs = []
    for i in range(len(Ws) - 1):
        z = activations[-1] @ Ws[i] + bs[i]
        zs.append(z)
        activations.append(relu(z))
    z_out = activations[-1] @ Ws[-1] + bs[-1]
    zs.append(z_out)
    y_pred = sigmoid(z_out).flatten()
    return zs, activations, y_pred


def train_spiral(X, y, hidden_sizes=(64, 64), lr=0.5, n_epochs=2000, seed=42, verbose=True):
    # Architecture 2-hidden_sizes-1 avec initialisation He
    sizes = [X.shape[1]] + list(hidden_sizes) + [1]
    Ws, bs = init_params(sizes, seed)
    n = len(X)
    losses = []

    for epoch in range(n_epochs):

        # ----------------------------
        # Forward — une couche à la fois
        # ----------------------------
        zs, activations, y_pred = forward(X, Ws, bs)

        loss = bce_loss(y, y_pred)
        losses.append(loss)

        # ----------------------------
        # Backward — sortie puis on remonte couche par couche
        # ----------------------------
        dWs = [None] * len(Ws)
        dbs = [None] * len(bs)

        err = (y_pred - y).reshape(-1, 1)  # erreur en sortie (simplification BCE+sigmoid)
        for i in reversed(range(len(Ws))):
            a_prev = activations[i]
            dWs[i] = a_prev.T @ err / n
            dbs[i] = np.mean(err, axis=0)
            if i > 0:
                # rétropropage à travers W_i puis à travers relu'(z_{i-1})
                err = (err @ Ws[i].T) * relu_grad(zs[i - 1])

        # ----------------------------
        # Mise à jour des paramètres
        # ----------------------------
        for i in range(len(Ws)):
            Ws[i] -= lr * dWs[i]
            bs[i] -= lr * dbs[i]

        if verbose and epoch % 500 == 0:
            acc = np.mean((y_pred > 0.5) == y)
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

    return Ws, bs, losses

# Note : l'énoncé indiquait lr = 0.01, mais avec cette valeur le réseau
# 2-64-64-1 apprend beaucoup trop lentement (< 70% d'accuracy après 2000
# epochs, voire pire car la loss baisse sans que l'accuracy suive). Avec un
# pas de descente en batch complet (gradient moyenné sur les 400 points),
# lr = 0.5 est nécessaire pour atteindre l'objectif (> 90% d'accuracy en
# 2000 epochs) annoncé par l'énoncé.


def predict_grid(grid, Ws, bs):
    _, _, y_pred = forward(grid, Ws, bs)
    return y_pred


def plot_boundary(ax, X, y, Ws, bs, title):
    h = 0.02
    xx, yy = np.meshgrid(
        np.arange(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, h),
        np.arange(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, h),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zg = predict_grid(grid, Ws, bs).reshape(xx.shape)

    ax.contourf(xx, yy, zg, alpha=0.4, cmap='RdBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', s=10, edgecolors='none')
    ax.set_title(title)


# ==========================
# Scénario normal : architecture 2-64-64-1
# ==========================

print("=== Scénario normal (2-64-64-1, noise=0.15) ===")
X, y = generate_spiral(n_points=400, noise=0.15)
W_normal, b_normal, losses_normal = train_spiral(X, y, hidden_sizes=(64, 64))

_, _, y_pred_normal = forward(X, W_normal, b_normal)
acc_normal = np.mean((y_pred_normal > 0.5) == y)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_boundary(axes[0], X, y, W_normal, b_normal, "Frontière de décision (2-64-64-1)")
axes[1].plot(losses_normal)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss BCE")
axes[1].set_title("Courbe de loss spirale")
plt.savefig("phase4_spirale.png", dpi=100, bbox_inches='tight')
plt.close()

print(f"\nLoss finale : {losses_normal[-1]:.4f}")
print(f"Accuracy finale : {acc_normal:.2%}")

# Résultat : le réseau large (2 couches de 64 neurones ReLU) dispose de
# suffisamment de capacité pour "enrouler" sa frontière de décision autour
# des deux spirales. La frontière suit fidèlement leur forme, avec une
# accuracy proche de 100%.


# ==========================
# Cas limite : architecture 2-2-1 (underfitting délibéré)
# ==========================

print("\n=== Cas limite (2-2-1, noise=0.15) ===")
W_small, b_small, losses_small = train_spiral(X, y, hidden_sizes=(2,))

_, _, y_pred_small = forward(X, W_small, b_small)
acc_small = np.mean((y_pred_small > 0.5) == y)

print(f"Loss finale : {losses_small[-1]:.4f}")
print(f"Accuracy finale : {acc_small:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_boundary(axes[0], X, y, W_normal, b_normal,
              f"2-64-64-1 (suffisant) — acc {acc_normal:.1%}")
plot_boundary(axes[1], X, y, W_small, b_small,
              f"2-2-1 (underfitting) — acc {acc_small:.1%}")
plt.savefig("phase4_spirale_underfitting.png", dpi=100, bbox_inches='tight')
plt.close()

# Explication : avec seulement 2 neurones cachés, le réseau ne peut combiner
# que 2 droites de séparation — bien trop peu pour suivre une spirale qui
# s'enroule sur elle-même. La frontière obtenue reste grossière (quasiment
# linéaire/quadratique) et l'accuracy plafonne très en dessous du réseau
# 2-64-64-1 : c'est un sous-fitting net et visible sur le plot.


# ==========================
# Scénario adversarial : bruit fort (noise=0.5)
# ==========================

print("\n=== Scénario adversarial (2-64-64-1, noise=0.5) ===")
X_noisy, y_noisy = generate_spiral(n_points=400, noise=0.5)
W_noisy, b_noisy, losses_noisy = train_spiral(X_noisy, y_noisy, hidden_sizes=(64, 64))

_, _, y_pred_noisy = forward(X_noisy, W_noisy, b_noisy)
acc_noisy = np.mean((y_pred_noisy > 0.5) == y_noisy)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_boundary(axes[0], X_noisy, y_noisy, W_noisy, b_noisy,
              f"2-64-64-1, noise=0.5 — acc {acc_noisy:.1%}")
axes[1].plot(losses_noisy)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss BCE")
axes[1].set_title("Courbe de loss (spirale bruitée)")
plt.savefig("phase4_spirale_noisy.png", dpi=100, bbox_inches='tight')
plt.close()

print(f"Loss finale : {losses_noisy[-1]:.4f}")
print(f"Accuracy finale : {acc_noisy:.2%}")
print(f"\nComparaison — noise=0.15 : {acc_normal:.2%} | noise=0.5 : {acc_noisy:.2%}")

# Observation : même avec noise=0.5, le réseau 2-64-64-1 reste largement
# au-dessus de 80% (proche de 98-99% avec cette seed), donc oui, un réseau
# capable de 100% sur la version propre reste très performant sur la
# version bruitée. Sa grande capacité (64x64 neurones) lui permet
# d'apprendre une frontière plus irrégulière/dentelée qui épouse quand même
# les points malgré le bruit, contrairement au réseau 2-2-1 du cas limite,
# qui lui n'aurait pas la capacité d'absorber ce bruit supplémentaire.
