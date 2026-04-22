from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from project.config import CFG


def build_tcn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LayerNormalization()(x_in)

    # Original architecture (kept for apples-to-apples comparison).
    for filters, dilation in [(64, 1), (64, 2), (64, 4), (32, 8)]:
        x = tf.keras.layers.Conv1D(
            filters,
            kernel_size=3,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
            kernel_initializer="he_normal",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
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
