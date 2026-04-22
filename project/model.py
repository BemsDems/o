from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from project.config import CFG


def build_tcn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LayerNormalization()(x_in)

    # Smaller, more regularized TCN to reduce overfitting on limited samples.
    dropout = float(CFG.get("DROPOUT", 0.3))

    for filters, dilation in [(32, 1), (32, 2), (16, 4)]:
        x = tf.keras.layers.Conv1D(
            filters,
            kernel_size=3,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
            kernel_initializer="he_normal",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(x_in, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(CFG["LR"]),
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model
