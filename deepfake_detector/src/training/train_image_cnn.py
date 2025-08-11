from __future__ import annotations
import argparse
from pathlib import Path
import json
from tensorflow import keras
from tensorflow.keras import callbacks, optimizers, losses, metrics
from src.models import build_simple_cnn
from src.utils import build_image_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Train/Eval Simple CNN on images")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--evaluate_only", action="store_true")
    p.add_argument("--weights", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_image_datasets(args.data_dir, image_size=(224, 224), batch_size=args.batch_size)

    model = build_simple_cnn((224, 224, 3))
    model.compile(
        optimizer=optimizers.Adam(args.lr),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy(name="accuracy"), metrics.AUC(name="auc")],
    )

    ckpt_path = str(out_dir / "best.weights.h5")
    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=True),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        callbacks.CSVLogger(str(out_dir / "training_log.csv")),
    ]

    if args.weights:
        model.load_weights(args.weights)

    history = None
    if not args.evaluate_only:
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    # Evaluate
    val_metrics = model.evaluate(val_ds, return_dict=True)
    test_metrics = model.evaluate(test_ds, return_dict=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    print("Validation:", val_metrics)
    print("Test:", test_metrics)

    # Always save final weights
    model.save_weights(ckpt_path)


if __name__ == "__main__":
    main()