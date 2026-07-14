# -*- coding: utf-8 -*-
"""phase8_pipeline_personnel

Dataset choisi : Breast Cancer Wisconsin (sklearn.datasets.load_breast_cancer)
569 exemples, 30 features numériques, classification binaire (malin=0 /
bénin=1). Bon point d'entrée : petit, propre, mais un vrai problème médical
réel — contrairement à MNIST, on ne sait pas d'avance jusqu'où on peut
pousser l'accuracy.

Pipeline complet : chargement → preprocessing → numpy from-scratch → Keras
→ comparaison, puis les 3 scénarios qualité (cas limite, adversarial) et
deux explorations bonus.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import time

# ---- 1. Chargement ----

data = load_breast_cancer()
X, y = data.data, data.target
print(f"Shape X : {X.shape} | Classes : {np.unique(y)}")
print(f"Répartition classes : {np.bincount(y)} (0=malin, 1=bénin)")

# ---- 2. Préprocessing ----

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit sur X_train uniquement : X_test ne doit jamais influencer les
# statistiques de normalisation (sinon data leakage — le modèle "voit"
# indirectement des informations du test avant l'évaluation).
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)


# ---- 3. Pipeline numpy from-scratch ----
# Architecture [n_features -> 16 -> 8 -> 1], He init, ReLU caché, sigmoid sortie
# (même schéma que la phase 4, généralisé à un nombre de couches quelconque)

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
    np.random.seed(seed)
    Ws, bs = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        Ws.append(np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in))
        bs.append(np.zeros(fan_out))
    return Ws, bs


def forward(X, Ws, bs):
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


def train_numpy(X, y, hidden_sizes=(16, 8), lr=0.5, n_epochs=200, seed=42, verbose=True):
    sizes = [X.shape[1]] + list(hidden_sizes) + [1]
    Ws, bs = init_params(sizes, seed)
    n = len(X)
    losses = []

    for epoch in range(n_epochs):
        zs, activations, y_pred = forward(X, Ws, bs)
        loss = bce_loss(y, y_pred)
        losses.append(loss)

        dWs = [None] * len(Ws)
        dbs = [None] * len(bs)
        err = (y_pred - y).reshape(-1, 1)
        for i in reversed(range(len(Ws))):
            a_prev = activations[i]
            dWs[i] = a_prev.T @ err / n
            dbs[i] = np.mean(err, axis=0)
            if i > 0:
                err = (err @ Ws[i].T) * relu_grad(zs[i - 1])

        for i in range(len(Ws)):
            Ws[i] -= lr * dWs[i]
            bs[i] -= lr * dbs[i]

        if verbose and epoch % 50 == 0:
            acc = np.mean((y_pred > 0.5) == y)
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

    return Ws, bs, losses


print("\n=== Pipeline numpy from-scratch (200 epochs) ===")
start = time.time()
Ws_np, bs_np, losses_np = train_numpy(X_train_s, y_train)
time_np = time.time() - start

_, _, y_pred_test_np = forward(X_test_s, Ws_np, bs_np)
test_acc_np = np.mean((y_pred_test_np > 0.5) == y_test)
test_loss_np = bce_loss(y_test, y_pred_test_np)

print(f"\nLoss finale (train) : {losses_np[-1]:.4f}")
print(f"Test accuracy (numpy) : {test_acc_np:.4f}")
print(f"Temps d'entraînement (numpy, 200 epochs) : {time_np:.2f}s")


# ---- 4. Pipeline Keras ----

def build_keras_model(n_features):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    return model


print("\n=== Pipeline Keras (50 epochs) ===")
tf.random.set_seed(42)
model_keras = build_keras_model(X.shape[1])

start = time.time()
history_keras = model_keras.fit(X_train_s, y_train,
                                 epochs=50, batch_size=32,
                                 validation_split=0.1, verbose=0)
time_keras = time.time() - start

test_loss_keras, test_acc_keras = model_keras.evaluate(X_test_s, y_test, verbose=0)

print(f"Test accuracy (Keras) : {test_acc_keras:.4f}")
print(f"Temps d'entraînement (Keras, 50 epochs) : {time_keras:.2f}s")


# ---- 5. Comparaison agrégée ----

print("\n=== TABLEAU COMPARATIF NUMPY vs KERAS ===")
print(f"{'Pipeline':10s} | {'Test accuracy':14s} | {'Loss finale':12s} | {'Epochs':8s} | {'Temps (s)':10s}")
print("-" * 70)
print(f"{'Numpy':10s} | {test_acc_np:<14.4f} | {losses_np[-1]:<12.4f} | {200:<8d} | {time_np:<10.2f}")
print(f"{'Keras':10s} | {test_acc_keras:<14.4f} | {history_keras.history['loss'][-1]:<12.4f} | {50:<8d} | {time_keras:<10.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(losses_np, label=f'numpy (200 epochs)')
axes[0].plot(history_keras.history['loss'], label=f'Keras (50 epochs)')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (train)")
axes[0].set_title("Courbes de loss — numpy vs Keras")
axes[0].legend()

axes[1].bar(['Numpy', 'Keras'], [test_acc_np, test_acc_keras], color=['C0', 'C1'])
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Test accuracy")
axes[1].set_title("Test accuracy — numpy vs Keras")
for i, v in enumerate([test_acc_np, test_acc_keras]):
    axes[1].text(i, v + 0.02, f"{v:.2%}", ha='center')

plt.savefig("phase8_comparaison.png", dpi=100, bbox_inches='tight')
plt.close()
print("\nGraphe sauvegardé : phase8_comparaison.png")

# numpy a besoin de 4x plus d'epochs (200 vs 50)
# pour atteindre une accuracy comparable à Keras, et malgré ça reste plus
# lent à converger par epoch (pas de momentum/adaptation du pas comme
# Adam). Sur un dataset aussi petit et propre que Breast Cancer, les deux
# pipelines finissent souvent très proches en test accuracy — l'écart se
# creuserait sur un problème plus gros ou plus bruité (cf. phases 5-7).


# ==========================
# Cas limite : zéros suspects (données manquantes encodées comme 0)
# ==========================

print("\n=== Cas limite : zéros suspects dans les features ===")

# Breast Cancer n'a pas de zéros suspects nativement (contrairement à Pima
# Diabetes, où glycémie=0 ou IMC=0 sont physiologiquement impossibles). On
# simule le même problème : on corrompt artificiellement une partie des
# lignes d'entraînement sur plusieurs colonnes en les mettant à 0, comme le
# ferait un capteur qui échoue ou un champ laissé vide et codé "0" par
# erreur. On répète sur 10 masques de corruption différents pour ne pas
# tirer de conclusion d'un seul run bruité (le set de test ne fait que 114
# lignes).

corrupt_cols = list(range(10))  # 10 des 30 features
corrupt_fraction = 0.6

naive_accs, fixed_accs = [], []
for mask_seed in range(10):
    rng = np.random.RandomState(mask_seed)
    corrupt_mask = rng.rand(len(X_train)) < corrupt_fraction
    X_train_corrupt = X_train.copy()
    for col in corrupt_cols:
        X_train_corrupt[corrupt_mask, col] = 0.0

    # Naive : on entraîne directement sur les zéros, sans les traiter
    scaler_naive = StandardScaler().fit(X_train_corrupt)
    Ws_n, bs_n, _ = train_numpy(scaler_naive.transform(X_train_corrupt), y_train, verbose=False)
    _, _, pred_naive = forward(scaler_naive.transform(X_test), Ws_n, bs_n)
    naive_accs.append(np.mean((pred_naive > 0.5) == y_test))

    # Fixed : on remplace les zéros suspects par la médiane (hors zéros) de la colonne
    X_train_fixed = X_train_corrupt.copy()
    for col in corrupt_cols:
        col_vals = X_train_fixed[:, col]
        median = np.median(col_vals[col_vals != 0])
        X_train_fixed[:, col] = np.where(col_vals == 0, median, col_vals)
    scaler_fixed = StandardScaler().fit(X_train_fixed)
    Ws_f, bs_f, _ = train_numpy(scaler_fixed.transform(X_train_fixed), y_train, verbose=False)
    _, _, pred_fixed = forward(scaler_fixed.transform(X_test), Ws_f, bs_f)
    fixed_accs.append(np.mean((pred_fixed > 0.5) == y_test))

print(f"Baseline propre (sans corruption)   : {test_acc_np:.4f}")
print(f"Naive (zéros non traités, moy/10)   : {np.mean(naive_accs):.4f} (+/- {np.std(naive_accs):.4f})")
print(f"Fixé (médiane, moy/10)              : {np.mean(fixed_accs):.4f} (+/- {np.std(fixed_accs):.4f})")

#  les deux versions corrompues (naive et fixée) perdent
# 2 à 3 points d'accuracy par rapport à la baseline propre — corrompre
# 60% des lignes sur 10 des 30 colonnes fait donc une vraie différence.
# En revanche, entre "naive" et "fixé", l'écart mesuré est faible et
# instable d'un masque à l'autre (écarts-types qui se chevauchent) : ici la
# corruption est injectée aléatoirement, indépendamment de la classe, et
# Breast Cancer a des features très redondantes (rayon/périmètre/aire sont
# corrélés) — le réseau peut donc compenser en s'appuyant sur les 20 autres
# colonnes propres. Sur un dataset réel comme Pima Diabetes, le problème
# est systémique (zéros dans TOUTES les lignes de certaines colonnes, pas
# un sous-ensemble aléatoire) et moins de colonnes sont disponibles pour
# compenser : l'effet du nettoyage y est documenté comme bien plus net.
# Conclusion honnête : traiter les valeurs manquantes reste la bonne
# pratique par défaut (on ne le saura pas à l'avance sur un nouveau
# dataset), mais son bénéfice mesuré dépend fortement de la redondance des
# features et du mécanisme de corruption.


# ==========================
# Scénario adversarial : entrée hors distribution
# ==========================

print("\n=== Scénario adversarial : ligne extrême (99999 partout) ===")
X_extreme = np.array([[99999.0] * X.shape[1]])
X_extreme_scaled = scaler.transform(X_extreme)

_, _, pred_extreme_np = forward(X_extreme_scaled, Ws_np, bs_np)
pred_extreme_keras = model_keras.predict(X_extreme_scaled, verbose=0)[0, 0]

print(f"Prédiction numpy (proba classe 'bénin') : {pred_extreme_np[0]:.6f}")
print(f"Prédiction Keras (proba classe 'bénin') : {pred_extreme_keras:.6f}")

# Constat : une valeur de 99999 sur les 30 features est à des dizaines de
# milliers d'écarts-types de tout ce que le modèle a vu à l'entraînement
# (StandardScaler la transforme en un z-score énorme). Pourtant, les deux
# modèles ne renvoient pas une probabilité proche de 0.5 (incertitude) : le
# ReLU ne sature jamais vers le haut, donc ces valeurs énormes traversent
# le réseau et le sigmoid final finit écrasé contre 0 ou 1 — une prédiction
# ultra-confiante sur une entrée qui n'a aucun sens médical. C'est le
# signal d'alarme classique des réseaux de neurones : rien dans leur
# architecture ne les empêche d'être sûrs d'eux sur des données totalement
# hors distribution. En production, une entrée aussi extrême devrait être
# rejetée par une validation en amont, pas laissée au réseau.


# ==========================
# Bonus 1 : impact de la normalisation
# ==========================

print("\n=== Bonus : impact de StandardScaler sur la convergence numpy ===")
Ws_raw, bs_raw, losses_raw = train_numpy(X_train, y_train, lr=0.5, n_epochs=200, verbose=False)
_, _, pred_raw_test = forward(X_test, Ws_raw, bs_raw)
test_acc_raw = np.mean((pred_raw_test > 0.5) == y_test)

print(f"Avec StandardScaler    : loss finale {losses_np[-1]:.4f} | test acc {test_acc_np:.4f}")
print(f"Sans StandardScaler    : loss finale {losses_raw[-1]:.4f} | test acc {test_acc_raw:.4f}")

# Constat (mesuré) : sans normalisation, avec le même lr=0.5 qui fonctionne
# très bien sur les données mises à l'échelle, l'entraînement se bloque
# quasi immédiatement (loss et accuracy figées dès les toutes premières
# epochs, test accuracy proche du simple "toujours prédire la classe
# majoritaire"). Les features brutes de Breast Cancer vivent à des échelles
# très différentes (ex. "mean area" ~ centaines/milliers vs "mean
# smoothness" ~ 0.05-0.15) : avec un pas de descente pensé pour des données
# centrées-réduites, les colonnes à grande échelle produisent des sommes
# pondérées énormes dès la première couche, saturant/désactivant une bonne
# partie des neurones ReLU de façon quasi permanente ("dead ReLU"). La
# normalisation n'est donc pas un détail cosmétique : sans elle, le même lr
# qui convergeait proprement devient inutilisable.


# ==========================
# Bonus 2 : facteur de vitesse numpy vs Keras
# ==========================

time_per_epoch_np = time_np / 200
time_per_epoch_keras = time_keras / 50
print("\n=== Bonus : temps par epoch, numpy vs Keras ===")
print(f"Numpy  : {time_per_epoch_np * 1000:.2f} ms/epoch")
print(f"Keras  : {time_per_epoch_keras * 1000:.2f} ms/epoch")
print(f"Numpy est {time_per_epoch_keras / time_per_epoch_np:.0f}x plus rapide par epoch ici, "
      f"mais Keras converge en 4x moins d'epochs (50 vs 200).")

# Constat (mesuré) : sur ce dataset minuscule (455 lignes d'entraînement,
# 30 features), c'est numpy qui est le plus rapide PAR EPOCH — de loin
# (~0.16ms contre ~68ms pour Keras). Contre-intuitif après les phases 5-7,
# mais logique : TensorFlow paie à chaque epoch un coût fixe (dispatch
# Python -> graphe TF, découpage en mini-batches de 32, retour des
# métriques) qui ne dépend presque pas de la taille des données, alors que
# notre boucle numpy fait juste quelques multiplications de petites
# matrices en une passe, sans aucun de ces frais fixes. Sur MNIST (phase 5,
# 60000 lignes), ce coût fixe devient négligeable face au calcul réel, et
# c'est l'inverse qui domine : les noyaux BLAS/XLA de Keras écrasent une
# boucle numpy naïve. La leçon : "Keras plus rapide" n'est pas une loi
# universelle, c'est vrai à partir d'une certaine taille de problème — en
# dessous, l'overhead du framework peut coûter plus cher que le calcul
# lui-même.
