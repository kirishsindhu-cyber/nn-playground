"""Core logic: datasets, TensorFlow model, decision-boundary plotting."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split

DATASETS = ["Moons", "Circles", "XOR", "Spiral"]


def make_data(name, noise=0.2, n=400, seed=0):
    rng = np.random.RandomState(seed)
    if name == "Moons":
        X, y = make_moons(n, noise=noise, random_state=seed)
    elif name == "Circles":
        X, y = make_circles(n, noise=noise / 2, factor=0.5, random_state=seed)
    elif name == "XOR":
        X = rng.uniform(-1, 1, (n, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        X += rng.normal(0, noise / 3, X.shape)
    else:  # Spiral
        m = n // 2
        t = np.linspace(0.3, 3.5 * np.pi, m)
        a = np.c_[t * np.cos(t), t * np.sin(t)] / 10
        b = np.c_[t * np.cos(t + np.pi), t * np.sin(t + np.pi)] / 10
        X = np.vstack([a, b]) + rng.normal(0, noise / 4, (2 * m, 2))
        y = np.r_[np.zeros(m), np.ones(m)].astype(int)
    X = (X - X.mean(0)) / X.std(0)
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def build_model(hidden_layers, neurons, activation, lr):
    tf.keras.utils.set_random_seed(1)
    layers = [tf.keras.Input(shape=(2,))]
    layers += [tf.keras.layers.Dense(neurons, activation=activation) for _ in range(hidden_layers)]
    layers += [tf.keras.layers.Dense(1, activation="sigmoid")]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_state(model, Xtr, ytr, Xte, yte, history):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    allX = np.vstack([Xtr, Xte])
    x0, x1 = allX[:, 0].min() - .5, allX[:, 0].max() + .5
    y0, y1 = allX[:, 1].min() - .5, allX[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x0, x1, 120), np.linspace(y0, y1, 120))
    zz = model(np.c_[xx.ravel(), yy.ravel()].astype("float32"), training=False).numpy().reshape(xx.shape)
    ax[0].contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.75, vmin=0, vmax=1)
    ax[0].scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap="RdBu", edgecolors="k", s=22, label="train")
    ax[0].scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap="RdBu", edgecolors="k", marker="s", s=26, label="test")
    ax[0].set_title("Decision boundary"); ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(history["loss"], label="train loss"); ax[1].plot(history["val_loss"], label="test loss")
    ax[1].set_title("Loss curve"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    return fig


def train(model, Xtr, ytr, Xte, yte, epochs, on_update=None, every=5):
    hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    class CB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, ep, logs=None):
            for k in hist: hist[k].append(logs[k])
            if on_update and ((ep + 1) % every == 0 or ep + 1 == epochs):
                on_update(ep + 1, hist)

    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs, batch_size=32, verbose=0, callbacks=[CB()])
    return hist
