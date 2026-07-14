# -*- coding: utf-8 -*-
"""phase3_xor

XOR : un neurone unique ne peut pas séparer les 4 points (ils ne sont pas
linéairement séparables). On ajoute une couche cachée pour résoudre le
problème, et on observe ce qui se passe si cette couche est trop petite
ou si les données sont bruitées.
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_loss_bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_xor(X, y, hidden_size=2, learning_rate=0.5, n_epochs=10000, seed=42, verbose=True):
    # Architecture 2-hidden_size-1
    np.random.seed(seed)

    W1 = np.random.randn(2, hidden_size) * 0.5
    b1 = np.random.randn(hidden_size) * 0.5
    W2 = np.random.randn(hidden_size, 1) * 0.5
    b2 = np.random.randn(1) * 0.5

    losses = []

    for epoch in range(n_epochs):

        # ----------------------------
        # Forward pass couche cachée
        # ----------------------------
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)  # shape [n, hidden_size]

        # ----------------------------
        # Forward pass couche de sortie
        # ----------------------------
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)  # shape [n, 1]

        y_pred = a2.flatten()
        loss = compute_loss_bce(y, y_pred)
        losses.append(loss)

        # ----------------------------
        # Backprop couche de sortie (simplification BCE + sigmoid)
        # ----------------------------
        error2 = (y_pred - y).reshape(-1, 1)           # [n, 1]
        dW2 = a1.T @ error2 / len(X)                     # [hidden_size, 1]
        db2 = np.mean(error2, axis=0)                     # [1,]

        # ----------------------------
        # Backprop couche cachée : chain rule à travers W2 puis a1
        # sigmoid'(z1) = a1 * (1 - a1)
        # ----------------------------
        error1 = (error2 @ W2.T) * (a1 * (1 - a1))       # [n, hidden_size]
        dW1 = X.T @ error1 / len(X)                        # [2, hidden_size]
        db1 = np.mean(error1, axis=0)                       # [hidden_size,]

        # ----------------------------
        # Mise à jour des paramètres
        # ----------------------------
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if verbose and epoch % 2000 == 0:
            acc = np.mean((y_pred > 0.5) == y)
            print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

    return W1, b1, W2, b2, losses


def plot_decision_boundary(W1, b1, W2, b2, X, y, filename, title):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    z1g = sigmoid(np.dot(grid, W1) + b1)
    z2g = sigmoid(np.dot(z1g, W2) + b2).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z2g, alpha=0.4, cmap='RdBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='RdBu', edgecolors='k')
    plt.title(title)
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


# ==========================
# Scénario normal : architecture 2-2-1
# ==========================

print("=== Scénario normal (2-2-1) ===")
W1, b1, W2, b2, losses = train_xor(X_xor, y_xor, hidden_size=2)

y_pred_final = sigmoid(sigmoid(X_xor @ W1 + b1) @ W2 + b2).flatten()
plot_decision_boundary(
    W1, b1, W2, b2, X_xor, y_xor,
    "phase3_xor_boundary.png",
    "XOR : frontière de décision du réseau 2-2-1"
)

print(f"\nLoss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {np.mean((y_pred_final > 0.5) == y_xor):.2%}")

# Résultat : avec 2 neurones cachés, chacun apprend sa propre droite de
# séparation (par ex. "OR" et "NAND"), et la couche de sortie combine ces
# deux droites pour reconstituer XOR. La frontière obtenue est non-linéaire :
# une région rouge autour de (0,0) et (1,1), une région bleue autour de
# (0,1) et (1,0), ce qu'aucune droite unique ne pourrait produire.


# ==========================
# Cas limite : couche cachée réduite à 1 seul neurone (2-1-1)
# ==========================

print("\n=== Cas limite : 1 seul neurone caché (2-1-1) ===")
W1_s, b1_s, W2_s, b2_s, losses_s = train_xor(X_xor, y_xor, hidden_size=1)

y_pred_s = sigmoid(sigmoid(X_xor @ W1_s + b1_s) @ W2_s + b2_s).flatten()
plot_decision_boundary(
    W1_s, b1_s, W2_s, b2_s, X_xor, y_xor,
    "phase3_xor_boundary_1neuron.png",
    "XOR : frontière avec 1 seul neurone caché (2-1-1)"
)

print(f"Loss finale : {losses_s[-1]:.4f}")
print(f"Accuracy finale : {np.mean((y_pred_s > 0.5) == y_xor):.2%}")

# Explication : avec un seul neurone caché, ce neurone ne peut tracer qu'une
# seule droite de séparation dans l'espace d'entrée. Or XOR a besoin d'au
# moins deux droites combinées (une pour isoler chaque point de la classe 1)
# pour être séparé correctement. Le réseau 2-1-1 se comporte donc à peu près
# comme un neurone unique : il reste bloqué autour de 50-75% d'accuracy et
# sa loss ne descend pas vers 0, quel que soit le nombre d'epochs.


# ==========================
# Scénario adversarial : 5% de bruit sur les coordonnées
# ==========================

print("\n=== Scénario adversarial : XOR bruité (2-2-1) ===")
np.random.seed(42)
X_xor_noisy = X_xor + np.random.randn(*X_xor.shape) * 0.05

W1_n, b1_n, W2_n, b2_n, losses_n = train_xor(X_xor_noisy, y_xor, hidden_size=2)

y_pred_n = sigmoid(sigmoid(X_xor_noisy @ W1_n + b1_n) @ W2_n + b2_n).flatten()
plot_decision_boundary(
    W1_n, b1_n, W2_n, b2_n, X_xor_noisy, y_xor,
    "phase3_xor_boundary_noisy.png",
    "XOR bruité (5%) : frontière de décision du réseau 2-2-1"
)

print(f"Loss finale : {losses_n[-1]:.4f}")
print(f"Accuracy finale : {np.mean((y_pred_n > 0.5) == y_xor):.2%}")

# Observation : un bruit de 5% sur seulement 4 points reste faible face à la
# capacité du réseau 2-2-1 (deux droites disponibles pour séparer les
# points). Le réseau converge généralement encore à 100% d'accuracy, avec
# une loss finale légèrement plus élevée que sans bruit. La frontière se
# déforme un peu autour des points déplacés, mais la séparation non-linéaire
# reste globalement la même. Avec un bruit plus important, ou avec un seul
# neurone caché, ce petit décalage pourrait suffire à empêcher la
# convergence complète.
