"""Gujarati handwritten WORD OCR using the trained CRNN V2 checkpoint.

This recognizer was trained on isolated word images, not complete handwritten
complaint documents. See README for the experimental full-page segmentation.
"""
from pathlib import Path
import unicodedata

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

IMAGE_H = 48
IMAGE_W = 320
DEFAULT_MODEL_FILENAME = "gujarati_crnn_v2_best.pt"


def prepare_word_image(image: Image.Image, height=IMAGE_H, width=IMAGE_W, *,
                       segmented_page_crop=False):
    """Normalize a word for CRNN V2.

    Direct single-word OCR keeps the established preprocessing unchanged.
    For auto-detected page crops, add a small WHITE top/bottom margin after
    correcting the paper background but BEFORE resizing. Tight bounding
    boxes can otherwise stretch short words vertically when resized to 48px.

    The six-pixel margin was selected from a small local diagnostic on one
    complaint image. It is heuristic and needs testing on other handwriting.
    """
    image = ImageOps.exif_transpose(image).convert("L")
    if image.width == 0 or image.height == 0:
        raise ValueError("Empty image")

    # Many camera/document photos have gray paper (pixel values ~200-225),
    # whereas the training data often has almost white backgrounds.
    # Make the paper white before resizing, while keeping dark strokes.
    # Preserve already-white images (including Kaggle word images).
    gray = np.asarray(image, dtype=np.uint8)
    paper_level = float(np.percentile(gray, 90))
    if paper_level < 244.0:
        lightening = float(np.clip(255.0 - paper_level + 30.0, 0, 90))
        corrected = np.clip(gray.astype(np.float32) + lightening, 0, 255)
        image = Image.fromarray(corrected.astype(np.uint8), mode="L")

    if segmented_page_crop:
        # Padding after paper whitening avoids the artificial white border
        # interfering with percentile-based background normalization.
        vertical_margin = 6
        padded = Image.new(
            "L", (image.width, image.height + 2 * vertical_margin), 255
        )
        padded.paste(image, (0, vertical_margin))
        image = padded

    new_w = max(1, min(width, round(image.width * height / image.height)))
    image = image.resize((new_w, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width, height), color=255)
    canvas.paste(image, (0, 0))
    return np.array(canvas, dtype=np.uint8).copy()


class GujaratiCRNNv2(nn.Module):
    """Same module names, layer order, and tensor shapes as Kaggle CRNN V2."""

    def __init__(self, n_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.projection = nn.Sequential(nn.Linear(256 * 3, 256), nn.ReLU())
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        x = 1.0 - x
        x = self.cnn(x)
        batch, channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().reshape(
            batch, width, channels * height
        )
        x = self.projection(x)
        x, _ = self.rnn(x)
        return self.head(x)


def ctc_greedy_decode(logits: torch.Tensor, chars: list[str]):
    """Collapse consecutive CTC predictions and remove blank ID 0."""
    predicted_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    predictions = []
    for sequence in predicted_ids:
        previous = None
        letters = []
        for token in sequence:
            if token != 0 and token != previous:
                letters.append(chars[token - 1])
            previous = token
        predictions.append(unicodedata.normalize("NFC", "".join(letters)))
    return predictions


class GujaratiWordOCR:
    def __init__(self, checkpoint_path=None, device=None):
        path = Path(checkpoint_path) if checkpoint_path is not None else (
            Path(__file__).resolve().parent / DEFAULT_MODEL_FILENAME
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Model file missing: {path}. Copy your downloaded "
                f"{DEFAULT_MODEL_FILENAME} from Kaggle into the civic_ocr folder."
            )

        # Only open checkpoints you trained or obtained from a trusted source.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "GujaratiCRNNv2":
            raise ValueError(
                "Checkpoint is not GujaratiCRNNv2. Please use your final "
                "gujarati_crnn_v2_best.pt downloaded from Kaggle."
            )
        self.height = int(checkpoint["image_height"])
        self.width = int(checkpoint["image_width"])
        if (self.height, self.width) != (IMAGE_H, IMAGE_W):
            raise ValueError(
                f"Unexpected checkpoint image size {self.width}x{self.height}; "
                f"expected {IMAGE_W}x{IMAGE_H}."
            )
        self.chars = checkpoint["chars"]
        if len(self.chars) != 79:
            raise ValueError(
                f"Expected 79 trained Gujarati characters; got {len(self.chars)}."
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = GujaratiCRNNv2(len(self.chars) + 1)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.to(self.device).eval()
        self.checkpoint_epoch = checkpoint["epoch"]
        self.best_cer = checkpoint["best_cer"]

    @torch.inference_mode()
    def recognize_word(self, image: Image.Image) -> str:
        arr = prepare_word_image(image, self.height, self.width)
        x = (
            torch.from_numpy(arr)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .div(255.0)
            .to(self.device)
        )
        return ctc_greedy_decode(self.model(x), self.chars)[0]

    @torch.inference_mode()
    def recognize_words(self, images: list[Image.Image], batch_size=16) -> list[str]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        predictions = []
        for start in range(0, len(images), batch_size):
            group = images[start : start + batch_size]
            arrays = np.stack(
                [prepare_word_image(im, self.height, self.width,
                                    segmented_page_crop=True) for im in group]
            )
            x = torch.from_numpy(arrays).unsqueeze(1).float().div(255.0).to(self.device)
            predictions.extend(ctc_greedy_decode(self.model(x), self.chars))
        return predictions


def segment_document_roughly(image: Image.Image):
    """Find likely handwritten word crops and group them into reading lines.

    EXPERIMENTAL, not a validated handwriting-page OCR system. Spacing, skew,
    small marks, or touching words can still make word proposals incorrect.
    Returns: (list of RGB word crops, annotated RGB page, line IDs).
    """
    import cv2

    source = ImageOps.exif_transpose(image).convert("RGB")
    if min(source.size) < 40:
        return [], source, []

    # Keep enough pixels for Gujarati vowel marks on high-resolution photos.
    scale = min(1.0, 1800.0 / max(source.size))
    if scale < 1.0:
        source = source.resize(
            (max(1, round(source.width * scale)),
             max(1, round(source.height * scale))),
            Image.Resampling.LANCZOS,
        )

    rgb = np.array(source)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Separate ink from paper, including mild lighting variation.
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15,
    )

    # Unlike the previous algorithm, do not interpret short connected
    # horizontal fragments as whole lines. First find WORD candidates globally.
    # Horizontal joining is kept fairly small to avoid merging adjacent words.
    joined = cv2.dilate(
        bw, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    )
    contours, _ = cv2.findContours(
        joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ink_pixels = cv2.countNonZero(bw[y:y + h, x:x + w])
        # Ignore specks; retain short Gujarati words. These thresholds are
        # heuristics and will need tuning on actual handwritten complaints.
        if w >= 17 and h >= 12 and ink_pixels >= 25:
            candidates.append((x, y, w, h, ink_pixels))

    if not candidates:
        return [], source, []

    typical_height = float(np.median([b[3] for b in candidates]))
    minimum_height = max(12, int(round(typical_height * 0.33)))
    candidates = [b for b in candidates if b[3] >= minimum_height]
    if not candidates:
        return [], source, []

    # Group words by their vertical centers, then sort each line from left to
    # right. The old function treated many individual words as separate lines.
    max_center_difference = max(10.0, typical_height * 0.75)
    candidates.sort(key=lambda b: (b[1] + b[3] / 2.0, b[0]))
    groups = []

    for box in candidates:
        center_y = box[1] + box[3] / 2.0
        nearest = None
        nearest_distance = float("inf")
        for group in groups:
            distance = abs(center_y - group["center_y"])
            if distance < nearest_distance and distance <= max_center_difference:
                nearest = group
                nearest_distance = distance

        if nearest is None:
            groups.append({"boxes": [box], "center_y": center_y})
        else:
            nearest["boxes"].append(box)
            nearest["center_y"] = float(np.median([
                b[1] + b[3] / 2.0 for b in nearest["boxes"]
            ]))

    groups.sort(key=lambda group: group["center_y"])
    for group in groups:
        group["boxes"].sort(key=lambda b: b[0])

    debug = rgb.copy()
    word_crops = []
    line_ids = []
    margin = max(3, round(typical_height * 0.08))

    for line_idx, group in enumerate(groups):
        for x, y, w, h, _ in group["boxes"]:
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(source.width, x + w + margin)
            y1 = min(source.height, y + h + margin)
            # Crop from unthresholded RGB input; the trained model expects
            # original grayscale converted internally, not a binary mask.
            word_crops.append(source.crop((x0, y0, x1, y1)))
            line_ids.append(line_idx)
            cv2.rectangle(debug, (x0, y0), (x1, y1), (230, 55, 55), 2)

    return word_crops, Image.fromarray(debug), line_ids


def join_words_by_line(predictions: list[str], line_numbers: list[int]) -> str:
    """Assemble detected words in top-to-bottom, left-to-right order."""
    if len(predictions) != len(line_numbers):
        raise ValueError("Prediction and line count mismatch")
    lines = {}
    for prediction, line_index in zip(predictions, line_numbers):
        # Preserve words for correction, but avoid blank tokens adding spaces.
        text = prediction.strip()
        if text:
            lines.setdefault(line_index, []).append(text)
    return "\n".join(" ".join(lines[i]) for i in sorted(lines))
