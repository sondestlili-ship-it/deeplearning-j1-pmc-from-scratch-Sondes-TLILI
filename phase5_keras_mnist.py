# -*- coding: utf-8 -*-
"""phase5_keras_mnist

Même logique que les phases 1-4 (forward → loss → backprop → update), mais
avec Keras : plus de dérivées à la main, plus de gestion manuelle des
shapes. On entraîne sur MNIST, un problème bien trop gros pour notre
numpy from-scratch des phases précédentes.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import time

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Préprocessing : flatten 28x28 → 784, normaliser entre 0 et 1
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

print(f"Train : {X_train.shape} | Test : {X_test.shape}")
print(f"Classes uniques : {np.unique(y_train)}")


def build_model(hidden_layers, input_shape=(784,), n_classes=10):
    """hidden_layers : liste de tailles de couches Dense(relu) avant la sortie softmax."""
    layers = [keras.layers.Input(shape=input_shape)]
    for size in hidden_layers:
        layers.append(keras.layers.Dense(size, activation='relu'))
    layers.append(keras.layers.Dense(n_classes, activation='softmax'))
    model = keras.Sequential(layers)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ==========================
# Scénario normal : Dense(128, relu) → Dense(64, relu) → Dense(10, softmax)
# ==========================

print("\n=== Scénario normal (epochs=5, batch_size=64) ===")
model = build_model([128, 64])
model.summary()

start = time.time()
history = model.fit(X_train, y_train,
                     epochs=5, batch_size=64,
                     validation_split=0.1, verbose=1)
elapsed = time.time() - start

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTemps d'entraînement : {elapsed:.1f}s")
print(f"Test accuracy : {test_acc:.4f}")
print(f"Test loss : {test_loss:.4f}")

# Courbes (code fourni — repose sur history.history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='train')
axes[0].plot(history.history['val_loss'], label='val')
axes[0].set_title("Loss")
axes[0].legend()
axes[1].plot(history.history['accuracy'], label='train')
axes[1].plot(history.history['val_accuracy'], label='val')
axes[1].set_title("Accuracy")
axes[1].legend()
plt.savefig("phase5_mnist_curves.png", dpi=100, bbox_inches='tight')
plt.close()
print("Courbes sauvegardées : phase5_mnist_curves.png")

# Avec Keras : 15 lignes pour construire+compiler+entraîner, contre ~80
# lignes de forward/backprop manuel en numpy (phases 3-4). Adam + les
# gradients calculés automatiquement (autodiff) convergent vers >97%
# d'accuracy en seulement 5 epochs, là où notre neurone/MLP numpy des
# phases 1-2 aurait besoin de centaines d'epochs pour approcher ce résultat
# sur un problème aussi grand (784 entrées, 10 classes, 60000 exemples).


# ==========================
# Cas limite : epochs=0
# ==========================

print("\n=== Cas limite (epochs=0) ===")
model_zero = build_model([128, 64])
try:
    history_zero = model_zero.fit(X_train, y_train,
                                   epochs=0, batch_size=64,
                                   validation_split=0.1, verbose=1)
    print("Pas d'erreur levée.")
    print(f"history.history (devrait être vide) : {history_zero.history}")
except Exception as e:
    print(f"Erreur levée : {type(e).__name__}: {e}")

# Constat : contrairement à ce qu'on pourrait imaginer, Keras ne lève pas
# d'erreur avec epochs=0. model.fit boucle simplement "0 fois" sur les
# epochs : aucun batch n'est traité, aucun poids n'est mis à jour (le
# modèle reste à son initialisation aléatoire), et history.history est un
# dictionnaire vide ({}). Pas de crash, mais un entraînement silencieusement
# nul — un piège si epochs=0 arrive par erreur (ex. variable de config mal
# calculée) : rien ne prévient qu'aucun apprentissage n'a eu lieu.


# ==========================
# Scénario adversarial : batch_size=1 (SGD pur) vs batch_size=64
# ==========================

print("\n=== Scénario adversarial : batch_size=1 vs batch_size=64 ===")

# batch_size=1 sur les 60000 exemples est très lent (60000 mises à jour de
# poids par epoch au lieu de ~938 avec batch_size=64). Pour garder un temps
# d'exécution raisonnable, on compare sur 1 epoch et un sous-ensemble des
# données d'entraînement — le contraste de vitesse et de stabilité reste
# parfaitement visible sans attendre plusieurs minutes.
n_subset = 6000
X_sub, y_sub = X_train[:n_subset], y_train[:n_subset]


class BatchLossHistory(keras.callbacks.Callback):
    """history.history ne garde que la loss moyenne par epoch. Pour voir
    l'instabilité step par step, il faut la capturer à chaque batch."""

    def on_train_begin(self, logs=None):
        self.batch_losses = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])


cb_bs64 = BatchLossHistory()
model_bs64 = build_model([128, 64])
start = time.time()
model_bs64.fit(X_sub, y_sub, epochs=1, batch_size=64, verbose=0, callbacks=[cb_bs64])
time_bs64 = time.time() - start

cb_bs1 = BatchLossHistory()
model_bs1 = build_model([128, 64])
start = time.time()
model_bs1.fit(X_sub, y_sub, epochs=1, batch_size=1, verbose=0, callbacks=[cb_bs1])
time_bs1 = time.time() - start

print(f"batch_size=64 | {n_subset} exemples, 1 epoch | temps : {time_bs64:.1f}s")
print(f"batch_size=1  | {n_subset} exemples, 1 epoch | temps : {time_bs1:.1f}s")
print(f"Ralentissement : x{time_bs1 / time_bs64:.1f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(cb_bs1.batch_losses, alpha=0.8)
axes[0].set_xlabel("Step (1 exemple / step)")
axes[0].set_ylabel("Loss (batch)")
axes[0].set_title("batch_size=1 : loss par step (bruitée)")
axes[1].plot(cb_bs64.batch_losses, alpha=0.8, color='C1')
axes[1].set_xlabel("Step (64 exemples / step)")
axes[1].set_ylabel("Loss (batch)")
axes[1].set_title("batch_size=64 : loss par step (lissée)")
plt.savefig("phase5_mnist_batchsize.png", dpi=100, bbox_inches='tight')
plt.close()

# Constat : batch_size=1 est nettement plus lent (beaucoup plus d'appels
# Python/TensorFlow pour le même nombre d'exemples, aucun bénéfice de la
# vectorisation matricielle qui fait la force de Keras/numpy sur CPU/GPU).
# La courbe de loss par step est aussi beaucoup plus bruitée/irrégulière :
# chaque mise à jour ne se base que sur 1 exemple, donc le gradient est une
# estimation à très haute variance du vrai gradient (contrairement à la
# moyenne sur 64 exemples qui lisse les mises à jour). Le coût d'un
# batch_size trop petit est donc double : plus lent ET moins stable, pour
# un gain de généralisation qui ne compense pas ici sur un aussi gros
# dataset.


# ==========================
# Pour aller plus loin : comparaison d'architectures
# ==========================

print("\n=== Bonus : comparaison d'architectures (2 epochs, sous-ensemble) ===")

# Comparaison à visée pédagogique : on utilise 2 epochs sur un
# sous-ensemble des données (au lieu de 5 epochs sur les 60000 exemples)
# pour garder un temps d'exécution raisonnable tout en illustrant l'écart
# entre architectures. Sur les 5 epochs complets, les écarts d'accuracy
# entre A/B/C resteraient du même ordre.
architectures = {
    "A : [128, 64]": [128, 64],
    "B : [256, 128, 64] (plus profonde)": [256, 128, 64],
    "C : [512] (plus large, moins profonde)": [512],
}

results = []
for name, hidden_layers in architectures.items():
    m = build_model(hidden_layers)
    n_params = m.count_params()
    start = time.time()
    h = m.fit(X_sub, y_sub, epochs=2, batch_size=64, validation_split=0.1, verbose=0)
    t = time.time() - start
    val_acc = h.history['val_accuracy'][-1]
    results.append((name, n_params, val_acc, t))
    print(f"{name:40s} | params: {n_params:7d} | val_acc: {val_acc:.4f} | temps: {t:.1f}s")

best = max(results, key=lambda r: r[2])
print(f"\nMeilleure architecture (sur ce mini-run) : {best[0]} avec val_acc={best[2]:.4f}")

# Explication attendue : sur MNIST, un problème relativement simple, les
# trois architectures obtiennent des accuracies proches. L'architecture B
# (plus profonde) a plus de paramètres et de capacité de représentation
# hiérarchique, mais peut être plus lente à converger en peu d'epochs et
# plus coûteuse en temps. L'architecture C (une seule grande couche) est
# souvent étonnamment compétitive sur un problème aussi simple que MNIST
# flatten : la profondeur n'aide vraiment que sur des problèmes où les
# caractéristiques doivent être composées hiérarchiquement (images brutes
# avec convolutions, par exemple), ce qui n'est pas vraiment le cas ici une
# fois les pixels aplatis en un vecteur de 784 valeurs.
