from typing import Tuple, Optional
from tensorflow import keras
from tensorflow.keras import layers


def build_xception(input_shape: Tuple[int, int, int] = (299, 299, 3), dropout: float = 0.4, freeze_until: Optional[int] = None) -> keras.Model:
    base = keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    if freeze_until is not None:
        for layer in base.layers[:freeze_until]:
            layer.trainable = False
        for layer in base.layers[freeze_until:]:
            layer.trainable = True

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.xception.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="xception_transfer")
    return model