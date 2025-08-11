from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from tensorflow import keras
from src.models import build_simple_cnn, build_xception, build_video_lstm
from src.utils import load_video_as_frame_sequence


def parse_args():
    p = argparse.ArgumentParser(description="Predict real/fake for an image or video")
    p.add_argument("--arch", choices=["cnn", "xception", "lstm"], required=True)
    p.add_argument("--model", type=str, required=True, help="Path to weights (.weights.h5)")
    p.add_argument("--image", type=str, default="")
    p.add_argument("--video", type=str, default="")
    p.add_argument("--num_frames", type=int, default=16)
    return p.parse_args()


essential = {"cnn": (224, 224), "xception": (299, 299)}


def main():
    args = parse_args()

    if args.arch in ("cnn", "xception") and not args.image:
        raise SystemExit("--image is required for cnn/xception")
    if args.arch == "lstm" and not args.video:
        raise SystemExit("--video is required for lstm")

    if args.arch == "cnn":
        model = build_simple_cnn((224, 224, 3))
        model.load_weights(args.model)
        img = keras.utils.load_img(args.image, target_size=essential["cnn"])
        arr = keras.utils.img_to_array(img)[None, ...]
        arr = arr / 255.0
        prob = float(model.predict(arr, verbose=0)[0][0])
    elif args.arch == "xception":
        model = build_xception((299, 299, 3))
        model.load_weights(args.model)
        img = keras.utils.load_img(args.image, target_size=essential["xception"])
        arr = keras.utils.img_to_array(img)[None, ...]
        arr = keras.applications.xception.preprocess_input(arr)
        prob = float(model.predict(arr, verbose=0)[0][0])
    else:  # lstm
        model = build_video_lstm(frame_shape=(224, 224, 3), timesteps=args.num_frames)
        model.load_weights(args.model)
        frames = load_video_as_frame_sequence(args.video, num_frames=args.num_frames, image_size=(224, 224))
        frames = keras.applications.mobilenet_v2.preprocess_input(frames)
        arr = frames[None, ...]
        prob = float(model.predict(arr, verbose=0)[0][0])

    label = "fake" if prob >= 0.5 else "real"
    confidence = prob if label == "fake" else (1.0 - prob)
    print({"label": label, "confidence": round(confidence, 4), "prob_fake": round(prob, 4)})


if __name__ == "__main__":
    main()