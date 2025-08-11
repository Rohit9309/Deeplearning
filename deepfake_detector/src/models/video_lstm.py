from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers


def build_video_lstm(
    frame_shape: Tuple[int, int, int] = (224, 224, 3),
    timesteps: int = 16,
    lstm_units: int = 128,
    dropout: float = 0.4,
) -> keras.Model:
    # CNN backbone for per-frame features
    cnn_backbone = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=frame_shape, pooling="avg"
    )
    cnn_backbone.trainable = False

    inputs = keras.Input(shape=(timesteps, *frame_shape))
    x = layers.TimeDistributed(keras.layers.Lambda(keras.applications.mobilenet_v2.preprocess_input))(inputs)
    x = layers.TimeDistributed(cnn_backbone)(x)

    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="video_cnn_lstm")
    return model