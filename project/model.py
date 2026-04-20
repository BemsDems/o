from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from project.config import CFG


class TCNBlock(tf.keras.layers.Layer):
    """One TCN block with residual connection."""

    def __init__(self, filters, kernel_size=3, dilation_rate=1, dropout=0.25, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            activation=None,
            kernel_initializer="he_normal",
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.relu = tf.keras.layers.ReLU()
        self.match_conv = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.match_conv = tf.keras.layers.Conv1D(
                self.filters,
                1,
                padding="same",
                kernel_initializer="he_normal",
            )
        super().build(input_shape)

    def call(self, x, training=False):
        residual = x
        if self.match_conv is not None:
            residual = self.match_conv(residual)

        out = self.conv(x)
        out = self.bn(out, training=training)
        out = self.relu(out)
        out = self.dropout(out, training=training)

        return out + residual


def build_tcn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LayerNormalization()(x_in)

    for filters, dilation in [(64, 1), (64, 2), (64, 4), (32, 8)]:
        x = TCNBlock(filters, kernel_size=3, dilation_rate=dilation, dropout=0.25)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(x_in, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(CFG["LR"]),
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model

