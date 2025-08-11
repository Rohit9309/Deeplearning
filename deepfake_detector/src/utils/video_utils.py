from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np
import tensorflow as tf


def load_video_as_frame_sequence(
    video_path: Union[str, Path],
    num_frames: int = 16,
    image_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    # Choose evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            # If read fails, duplicate last good frame
            if frames:
                frames.append(frames[-1].copy())
                continue
            else:
                raise ValueError(f"Failed to read frame {idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    arr = np.stack(frames, axis=0).astype("float32")
    return arr


def _gather_video_paths_and_labels(root_dir: Path) -> Tuple[List[Path], List[int]]:
    classes = ["real", "fake"]
    paths: List[Path] = []
    labels: List[int] = []
    for label, cls in enumerate(classes):
        class_dir = root_dir / cls
        if not class_dir.exists():
            continue
        for p in class_dir.rglob("*.mp4"):
            paths.append(p)
            labels.append(label)
        for p in class_dir.rglob("*.avi"):
            paths.append(p)
            labels.append(label)
        for p in class_dir.rglob("*.mov"):
            paths.append(p)
            labels.append(label)
    return paths, labels


def build_video_dataset(
    data_dir: Union[str, Path],
    split: str,
    num_frames: int = 16,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 2,
    shuffle: bool = True,
    seed: int = 1337,
) -> tf.data.Dataset:
    root = Path(data_dir) / split
    paths, labels = _gather_video_paths_and_labels(root)
    if not paths:
        raise ValueError(f"No videos found under {root}")

    ds = tf.data.Dataset.from_tensor_slices((
        [str(p) for p in paths],
        labels,
    ))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    def load_fn(path, label):
        frames = tf.numpy_function(
            func=lambda p: load_video_as_frame_sequence(p.decode("utf-8"), num_frames, image_size),
            inp=[path],
            Tout=tf.float32,
        )
        frames.set_shape((num_frames, image_size[1], image_size[0], 3))
        return frames, tf.cast(label, tf.float32)

    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds