# -*- coding: utf-8 -*-
"""phase7_learning_rate

Même architecture, même optimizer (Adam), seul le learning rate change.
On regarde comment un LR trop petit, trop grand, ou "juste bien" se
comporte sur MNIST — puis on compare Adam à SGD au sweet spot.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import time

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

learning_rates = [1e-7, 1e-3, 1.0]
lr_labels = ['trop petit (1e-7)', 'sweet spot (1e-3)', 'trop grand (1.0)']
results = []
histories = {}


def build_model():
    return keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])


def run_experiment(label, optimizer, epochs=10, batch_size=64, seed=42):
    # Même graine avant chaque construction de modèle : seule la config
    # d'optimisation (lr ou optimizer) change entre les runs.
    tf.random.set_seed(seed)
    model = build_model()
    model.compile(optimizer=optimizer,
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    start = time.time()
    history = model.fit(X_train, y_train,
                         epochs=epochs, batch_size=batch_size,
                         validation_split=0.1, verbose=0)
    elapsed = time.time() - start

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    val_losses = history.history['val_loss']

    result = {
        'label': label,
        'val_loss_final': val_losses[-1],
        'test_accuracy': test_acc,
        'train_time_s': elapsed,
    }
    print(f"{label:24s} | val_loss={result['val_loss_final']:.4f} | "
          f"test_acc={test_acc:.4f} | temps={elapsed:.1f}s")
    return result, val_losses


# ==========================
# Scénario normal + cas limite + adversarial : 3 learning rates
# ==========================

print("=== Impact du learning rate (Adam, 10 epochs) ===")
for lr, label in zip(learning_rates, lr_labels):
    result, val_losses = run_experiment(label, keras.optimizers.Adam(learning_rate=lr))
    result['lr'] = lr
    results.append(result)
    histories[label] = val_losses


def print_table(rows):
    print("\n=== TABLEAU COMPARATIF LEARNING RATE ===")
    print(f"{'LR':8s} | {'Label':24s} | {'Val loss final':14s} | {'Test acc':10s} | {'Temps (s)':10s}")
    print("-" * 80)
    for r in rows:
        print(f"{r['lr']:.0e} | {r['label']:24s} | {r['val_loss_final']:<14.4f} | "
              f"{r['test_accuracy']:<10.4f} | {r['train_time_s']:.0f}")


print_table(results)

# Courbe superposée, échelle log : rend les extrêmes lisibles
plt.figure(figsize=(10, 5))
for label, val_losses in histories.items():
    plt.plot(range(1, 11), val_losses, label=label, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.title("Impact du learning rate sur la convergence (MNIST)")
plt.legend()
plt.yscale('log')
plt.savefig("phase7_lr_curve.png", dpi=100, bbox_inches='tight')
plt.close()
print("\nCourbe sauvegardée : phase7_lr_curve.png")

# ==========================
# Cas limite : lr=1e-7, la loss bouge-t-elle vraiment ?
# ==========================

small_lr_losses = histories['trop petit (1e-7)']
delta = small_lr_losses[0] - small_lr_losses[-1]
print(f"\n=== Cas limite : lr=1e-7 ===")
print(f"Val loss epoch 1  : {small_lr_losses[0]:.6f}")
print(f"Val loss epoch 10 : {small_lr_losses[-1]:.6f}")
print(f"Delta (epoch1 - epoch10) : {delta:.6f}")

# Constat (mesuré) : delta ≈ 0.06 sur 10 epochs — le réseau bouge
# techniquement (ce n'est pas un delta nul, la loss ne stagne pas à la
# décimale près), mais c'est dérisoire face au chemin à parcourir : il part
# de ~2.37 (quasi la loss d'une prédiction uniforme sur 10 classes) et n'a
# grappillé qu'un vingtième de point après 10 epochs, quand lr=1e-3
# atteint 0.11 sur le même nombre d'epochs. À ce rythme il faudrait des
# centaines d'epochs supplémentaires pour espérer s'approcher du sweet
# spot. En contexte réel (dataset bien plus gros, epochs bien plus
# coûteuses), ça se traduirait par des jours d'entraînement pour un
# résultat que lr=1e-3 atteint en quelques secondes.

# ==========================
# Scénario adversarial : lr=1.0, la loss autour de ln(10) ?
# ==========================

uniform_ce = -np.log(1 / 10)  # ~2.3026 : loss d'une prédiction uniforme sur 10 classes
big_lr_result = results[2]
print(f"\n=== Scénario adversarial : lr=1.0 ===")
print(f"Val loss finale : {big_lr_result['val_loss_final']:.4f} "
      f"(loss d'une prédiction uniforme sur 10 classes : {uniform_ce:.4f})")
if abs(big_lr_result['val_loss_final'] - uniform_ce) < 0.3:
    print("=> La loss oscille autour de ln(10) : le modèle n'a rien appris, "
          "il prédit essentiellement au hasard.")
else:
    print("=> La loss diverge/oscille loin de son point de départ : "
          "l'entraînement est instable mais pas figé sur la prédiction uniforme.")

# Explication : avec lr=1.0, chaque mise à jour de poids est démesurée par
# rapport à la courbure de la loss. Le modèle saute par-dessus les minima au
# lieu de s'en approcher progressivement ; les poids peuvent même diverger
# vers des valeurs extrêmes qui saturent les couches. Le résultat le plus
# fréquent est un modèle bloqué autour de la loss d'une prédiction uniforme
# (~2.30, soit -log(1/10)) : il ne fait mieux qu'une réponse au hasard sur
# les 10 classes, quel que soit le nombre d'epochs.


# ==========================
# Pour aller plus loin : Adam vs SGD au sweet spot
# ==========================

print("\n=== Bonus : Adam vs SGD ===")
optimizers_to_test = [
    ('Adam lr=1e-3', keras.optimizers.Adam(learning_rate=1e-3), 1e-3),
    ('SGD lr=1e-3', keras.optimizers.SGD(learning_rate=1e-3), 1e-3),
    ('SGD lr=1e-2', keras.optimizers.SGD(learning_rate=1e-2), 1e-2),  # SGD a souvent besoin d'un lr plus grand
]

optimizer_results = []
for label, optimizer, lr in optimizers_to_test:
    result, val_losses = run_experiment(label, optimizer)
    result['lr'] = lr
    optimizer_results.append(result)
    histories[label] = val_losses

results.extend(optimizer_results)
print_table(results)

plt.figure(figsize=(10, 5))
for label, val_losses in histories.items():
    plt.plot(range(1, 11), val_losses, label=label, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.title("Learning rate + Adam vs SGD (MNIST)")
plt.legend()
plt.yscale('log')
plt.savefig("phase7_lr_optimizers_curve.png", dpi=100, bbox_inches='tight')
plt.close()
print("Courbe étendue sauvegardée : phase7_lr_optimizers_curve.png")

adam_epochs_to_below_01 = next((i for i, v in enumerate(histories['Adam lr=1e-3'], start=1) if v < 0.1), None)
sgd1_epochs_to_below_01 = next((i for i, v in enumerate(histories['SGD lr=1e-3'], start=1) if v < 0.1), None)
sgd2_epochs_to_below_01 = next((i for i, v in enumerate(histories['SGD lr=1e-2'], start=1) if v < 0.1), None)
print(f"\nEpoch où val_loss < 0.1 -> Adam lr=1e-3: {adam_epochs_to_below_01} | "
      f"SGD lr=1e-3: {sgd1_epochs_to_below_01} | SGD lr=1e-2: {sgd2_epochs_to_below_01}")

# Constat attendu : à lr=1e-3 identique, Adam converge nettement plus vite
# que SGD (Adam adapte le pas par paramètre grâce aux moments d'ordre 1 et
# 2, ce qui accélère beaucoup la descente au tout début de l'entraînement,
# quand SGD "vanille" avance à pas constant et donc plus lentement). Monter
# le lr de SGD à 1e-2 (x10) réduit l'écart et peut rapprocher SGD d'Adam en
# val_accuracy sur 10 epochs, mais SGD reste généralement plus lent à
# franchir le seuil val_loss < 0.1 : Adam garde l'avantage de la vitesse de
# convergence, même face à un SGD dont on a compensé le lr trop prudent.

"""Résultats mesurés (run réel, 10 epochs, MNIST complet)

LR      | Label              | Val loss final | Test acc | Temps
1e-7    | trop petit         | 2.3055          | 7.1%     | 9s
1e-3    | sweet spot         | 0.1109          | 97.3%    | 9s
1.0     | trop grand         | 2.3733          | 9.8%     | 10s
1e-3    | Adam               | 0.0936          | 97.8%    | 11s
1e-3    | SGD                | 0.3854          | 88.3%    | 9s
1e-2    | SGD (10x)          | 0.1513          | 94.8%    | 8s

- trop petit (1e-7) et trop grand (1.0) finissent tous les deux proches de
  ln(10) ≈ 2.303, la loss d'une prédiction uniforme sur 10 classes : ni
  l'un ni l'autre n'apprend quoi que ce soit en 10 epochs.
- sweet spot (1e-3) est le seul des trois à converger proprement.
- Adam vs SGD au même lr=1e-3 : Adam finit à 97.8% contre 88.3% pour SGD.
  Donner à SGD un lr 10x plus grand (1e-2) referme une partie de l'écart
  (94.8%) sans le rattraper complètement.
"""
