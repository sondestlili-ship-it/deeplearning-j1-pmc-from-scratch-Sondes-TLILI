# -*- coding: utf-8 -*-
"""phase6_activations

Même architecture, même learning rate, même dataset : seule la fonction
d'activation des couches cachées change. On mesure son impact sur la
vitesse de convergence et la qualité finale du modèle.
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

activations = ['sigmoid', 'tanh', 'relu']
results = []
histories = {}


def build_model(hidden_sizes, hidden_activation, input_shape=(784,), n_classes=10):
    """hidden_activation : nom d'activation Keras, ou None pour une couche linéaire."""
    layers = [keras.layers.Input(shape=input_shape)]
    for size in hidden_sizes:
        layers.append(keras.layers.Dense(size, activation=hidden_activation))
    layers.append(keras.layers.Dense(n_classes, activation='softmax'))
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def first_epoch_below(val_losses, threshold=0.1):
    """Première epoch (1-indexée) où val_loss < threshold, sinon 'N/A'."""
    for i, v in enumerate(val_losses, start=1):
        if v < threshold:
            return i
    return "N/A"


def run_experiment(name, hidden_sizes, hidden_activation, epochs=10, batch_size=64, seed=42):
    # Même graine avant chaque construction de modèle : initialisation
    # comparable entre les runs, seule l'activation (ou l'architecture) change.
    tf.random.set_seed(seed)
    model = build_model(hidden_sizes, hidden_activation)

    start = time.time()
    history = model.fit(X_train, y_train,
                         epochs=epochs, batch_size=batch_size,
                         validation_split=0.1, verbose=0)
    elapsed = time.time() - start

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    val_losses = history.history['val_loss']

    result = {
        'activation': name,
        'val_loss_final': val_losses[-1],
        'test_accuracy': test_acc,
        'convergence_epoch_sub01': first_epoch_below(val_losses),
        'train_time_s': elapsed,
    }
    print(f"{name:12s} | val_loss={result['val_loss_final']:.4f} | "
          f"test_acc={test_acc:.4f} | temps={elapsed:.1f}s")
    return result, val_losses


# ==========================
# Scénario normal : sigmoid vs tanh vs relu
# ==========================

print("=== Comparaison des activations (2-128-64-10, 10 epochs) ===")
for activation in activations:
    result, val_losses = run_experiment(activation, (128, 64), activation)
    results.append(result)
    histories[activation] = val_losses


def print_table(rows):
    print("\n=== TABLEAU COMPARATIF ===")
    print(f"{'Activation':12s} | {'Val loss epoch 10':18s} | {'Test accuracy':14s} | "
          f"{'Epoch < 0.1 loss':16s} | {'Temps (s)':10s}")
    print("-" * 84)
    for r in rows:
        print(f"{r['activation']:12s} | {r['val_loss_final']:<18.4f} | "
              f"{r['test_accuracy']:<14.4f} | {str(r['convergence_epoch_sub01']):16s} | "
              f"{r['train_time_s']:<10.0f}")


print_table(results)

# Courbe superposée : val_loss par epoch pour les 3 activations
plt.figure(figsize=(8, 5))
for name, val_losses in histories.items():
    plt.plot(val_losses, label=name, marker='o', markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Val loss")
plt.title("Convergence selon l'activation (couches cachées)")
plt.legend()
plt.savefig("phase6_activations.png", dpi=100, bbox_inches='tight')
plt.close()
print("\nCourbe sauvegardée : phase6_activations.png")

# Constat observé (phase6_activations.png) : ReLU et tanh descendent
# nettement plus vite que sigmoid sur les toutes premières epochs (pas de
# saturation aux extrêmes, gradient non écrasé), et ReLU/tanh franchissent
# le seuil val_loss < 0.1 dès l'epoch 3 contre l'epoch 5 pour sigmoid — la
# confirmation que ReLU converge plus tôt.
# Mais sur 10 epochs complètes, la fin de course est plus nuancée : la
# val_loss de ReLU recommence à remonter après l'epoch ~4 (léger
# sur-apprentissage, plus rapide justement parce que ReLU laisse passer un
# gradient plus fort), alors que sigmoid, bien que plus lente au départ,
# continue de descendre et finit avec la meilleure val_loss des trois à
# l'epoch 10. Convergence rapide ne veut donc pas dire meilleur résultat
# final : ReLU gagne la vitesse, pas forcément l'arrivée sur ce run précis.


# ==========================
# Cas limite : pas d'activation (linéaire) sur les couches cachées
# ==========================

print("\n=== Cas limite : hidden_activation=None (linéaire) ===")
result_linear, val_losses_linear = run_experiment('linear', (128, 64), None)
results.append(result_linear)
histories['linear'] = val_losses_linear

# Explication : sans fonction d'activation, une couche Dense reste une
# transformation affine (Wx + b). Empiler deux couches "linéaires" reste
# mathématiquement équivalent à une seule transformation linéaire (le
# produit de deux matrices est encore une matrice). Le réseau perd donc
# toute sa capacité à représenter des frontières non-linéaires : il se
# comporte comme une simple régression logistique multi-classes appliquée
# aux pixels bruts. Sa val_loss finale reste nettement plus haute que celle
# de ReLU/tanh/sigmoid.


# ==========================
# Scénario adversarial : softmax en couches cachées
# ==========================

print("\n=== Scénario adversarial : hidden_activation='softmax' ===")
result_softmax, val_losses_softmax = run_experiment('softmax_hidden', (128, 64), 'softmax')
results.append(result_softmax)
histories['softmax_hidden'] = val_losses_softmax

# Explication : softmax normalise la sortie d'une couche pour qu'elle somme
# à 1 sur l'ensemble de ses neurones — c'est utile en sortie pour obtenir
# une distribution de probabilités sur les classes, mais destructeur en
# couche cachée. Les 128 (ou 64) activations d'une couche cachée softmax
# sont forcées à se "partager" une masse totale de 1 : elles perdent leur
# indépendance et leur magnitude individuelle (l'information utile pour la
# couche suivante), et deviennent presque toutes proches de 0 quand la
# couche est large. Le signal transmis aux couches suivantes est donc
# fortement appauvri, ce qui dégrade nettement val_accuracy par rapport à
# ReLU. C'est pourquoi softmax est réservé à la couche de sortie.


# ==========================
# Tableau complet (activations + cas limite + adversarial)
# ==========================

print_table(results)

plt.figure(figsize=(8, 5))
for name, val_losses in histories.items():
    plt.plot(val_losses, label=name, marker='o', markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Val loss")
plt.title("Convergence : activations valides vs cas limite / adversarial")
plt.legend()
plt.savefig("phase6_activations_extended.png", dpi=100, bbox_inches='tight')
plt.close()
print("Courbe étendue sauvegardée : phase6_activations_extended.png")


# ==========================
# Pour aller plus loin : profondeur (ReLU uniquement)
# ==========================

print("\n=== Bonus : impact de la profondeur (ReLU) ===")
depth_configs = [
    ("relu_peu_profond (1 couche)", (256,)),
    ("relu_moyen (2 couches)", (128, 64)),
    ("relu_profond (3 couches)", (128, 64, 32)),
]

depth_results = []
for name, hidden_sizes in depth_configs:
    result, _ = run_experiment(name, hidden_sizes, 'relu')
    depth_results.append(result)

results.extend(depth_results)
print_table(results)

# Constat attendu : sur MNIST flatten, ajouter de la profondeur n'améliore
# pas forcément le résultat au-delà de 2 couches — le problème (784 pixels
# bruts → 10 classes) ne nécessite pas de composer des caractéristiques
# très hiérarchiques comme le ferait un CNN sur une image structurée. Le
# réseau "moyen" ou "profond" peut même faire légèrement moins bien que le
# peu profond en seulement 10 epochs (plus de paramètres à apprendre,
# convergence plus lente), pour un coût en temps d'entraînement plus élevé.
# La profondeur a donc un coût (temps, risque de sur-paramétrisation) qui
# n'est pas toujours rentabilisé par un gain d'accuracy sur un problème
# aussi simple.
