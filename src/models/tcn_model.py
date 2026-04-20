from __future__ import annotations

import tensorflow as tf
from tcn import TCN


def build_tcn_model(window: int, n_features: int, lr: float) -> tf.keras.Model:
    """TCN architecture for binary classification."""
    inp = tf.keras.Input(shape=(window, n_features))

    x = TCN(
        nb_filters=32,
        kernel_size=3,
        nb_stacks=1,
        dilations=(1, 2, 4, 8),
        padding="causal",
        use_skip_connections=True,
        dropout_rate=0.3,
        return_sequences=False,
        activation="relu",
        kernel_initializer="he_normal",
    )(inp)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )
    return model

