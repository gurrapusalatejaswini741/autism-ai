"""
model.py
--------
Model loading, building, and inference - Streamlit Cloud compatible.
Falls back gracefully to a demo mode when no trained model exists.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

GRADCAM_LAYER       = "Conv_1"
DEFAULT_MODEL_PATH  = os.path.join("models", "autism_model.h5")


# ── Model Building ─────────────────────────────────────────────────────────────

def build_model(input_shape=(224, 224, 3), trainable_base=False):
    """MobileNetV2 + custom classification head."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = trainable_base

    inputs = tf.keras.Input(shape=input_shape)
    x      = base(inputs, training=False)
    x      = layers.GlobalAveragePooling2D()(x)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(256, activation="relu")(x)
    x      = layers.Dropout(0.4)(x)
    x      = layers.Dense(128, activation="relu")(x)
    x      = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, output, name="AutismScreener_MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    logger.info(f"Model built  |  params: {model.count_params():,}")
    return model


def load_model(path=DEFAULT_MODEL_PATH):
    """
    Load trained model. Falls back to fresh (untrained) model in demo mode
    so the Streamlit Cloud app still renders without weights.
    """
    import tensorflow as tf

    if os.path.exists(path):
        logger.info(f"Loading trained model from {path}")
        return tf.keras.models.load_model(path)

    logger.warning(
        f"No model at '{path}' — running in DEMO MODE with random weights."
    )
    return build_model()


def save_model(model, path=DEFAULT_MODEL_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)
    logger.info(f"Model saved → {path}")


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_single(model, img_array) -> float:
    pred = model.predict(img_array, verbose=0)
    return float(pred[0][0])


def predict_all_images(model, image_arrays: dict) -> dict:
    predictions = {}
    for key, arr in image_arrays.items():
        if arr is not None:
            predictions[key] = predict_single(model, arr)
            logger.info(f"  [{key}] conf = {predictions[key]:.4f}")
    return predictions


def unfreeze_top_layers(model, num_layers=30):
    import tensorflow as tf
    base = model.layers[1]
    base.trainable = True
    total = len(base.layers)
    for layer in base.layers[: total - num_layers]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model
