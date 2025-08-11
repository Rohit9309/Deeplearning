from __future__ import annotations
from pathlib import Path
from typing import Tuple, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_image_datasets(
    data_dir: Union[str, Path],
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 1337,
):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        seed=seed,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        seed=seed,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    # Caching and prefetching
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    # Augmentation for training
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ]
    )

    def augment(x, y):
        return data_augmentation(x), y

    return train_ds.map(augment), val_ds, test_ds