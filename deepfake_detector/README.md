# Deepfake Detector (Images + Videos)

Detect AI-generated (fake) vs real content using three architectures:
- Simple CNN (images)
- XceptionNet transfer learning (images)
- CNN+LSTM (videos)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset layout

Place your data under `data/`:

```
data/
  images/
    train/
      real/
      fake/
    val/
      real/
      fake/
    test/
      real/
      fake/
  videos/
    train/
      real/
      fake/
    val/
      real/
      fake/
    test/
      real/
      fake/
```

- Classes are folder names: `real` and `fake`.

## Train

- Simple CNN (images, 224x224):
```bash
python -m src.training.train_image_cnn --data_dir data/images --out_dir outputs/cnn --epochs 10 --batch_size 32
```

- XceptionNet (images, 299x299):
```bash
python -m src.training.train_xception --data_dir data/images --out_dir outputs/xception --epochs 8 --batch_size 24 --freeze_until 80
```

- CNN+LSTM (videos, 224x224, 16 frames sampled):
```bash
python -m src.training.train_video_lstm --data_dir data/videos --out_dir outputs/lstm --epochs 8 --batch_size 2 --num_frames 16
```

## Evaluate

```bash
python -m src.training.train_image_cnn --data_dir data/images --out_dir outputs/cnn --evaluate_only --weights outputs/cnn/best.weights.h5
python -m src.training.train_xception  --data_dir data/images --out_dir outputs/xception --evaluate_only --weights outputs/xception/best.weights.h5
python -m src.training.train_video_lstm --data_dir data/videos --out_dir outputs/lstm --evaluate_only --weights outputs/lstm/best.weights.h5 --num_frames 16
```

Each script prints accuracy on the validation and test sets and saves a metrics JSON in the output directory.

## Inference (single image/video)

```bash
# Image with CNN
python -m src.inference.predict --arch cnn --model outputs/cnn/best.weights.h5 --image path/to/image.jpg

# Image with Xception
python -m src.inference.predict --arch xception --model outputs/xception/best.weights.h5 --image path/to/image.jpg

# Video with LSTM
python -m src.inference.predict --arch lstm --model outputs/lstm/best.weights.h5 --video path/to/video.mp4 --num_frames 16
```

Outputs a label (real/fake) and confidence.

## Notes
- Models are binary classifiers with sigmoid outputs.
- LSTM model extracts per-frame features with a CNN backbone and aggregates temporally.
- Adjust `--epochs`, `--batch_size`, and learning rate to your hardware.