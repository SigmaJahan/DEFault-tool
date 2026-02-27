import numpy as np
import tensorflow as tf


def build_model():
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(20,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_data():
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(1024, 20)).astype("float32")
    y_train = rng.integers(0, 2, size=(1024,), dtype=np.int32)
    x_val = rng.normal(size=(256, 20)).astype("float32")
    y_val = rng.integers(0, 2, size=(256,), dtype=np.int32)
    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "batch_size": 64,
    }

