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
    """Experimental page segmentation, adaptive to document resolution.

    Returns (word crops, box-overlay, line_ids), compatible with the existing
    Streamlit app. Blue ink on wrinkled gray paper is normalized locally; when
    projection reveals well-separated lines, find words within those bands.
    Otherwise, use component-based word proposals grouped by reading line.

    This is geometric segmentation, NOT trained Gujarati character detection.
    Tiny standalone marks are deliberately not sent to word OCR, because that
    would frequently turn a comma/period/noise speck into a Gujarati numeral.
    """
    import cv2

    source = ImageOps.exif_transpose(image).convert("RGB")
    if min(source.size) < 40:
        return [], source, []
    # Normalize working resolution; avoid enlarging very small photos without
    # warning users that upsampling cannot invent lost stroke details.
    factor = min(1.0, 2200 / max(source.size))
    if source.width < 1000:
        factor = min(2.8, 1050 / source.width, 2200 / max(source.size))
    if abs(factor - 1.0) > 0.001:
        source = source.resize(
            (max(1, round(source.width * factor)),
             max(1, round(source.height * factor))),
            Image.Resampling.BICUBIC if factor > 1 else Image.Resampling.LANCZOS,
        )
    rgb = np.asarray(source).copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sigma = max(11.0, min(gray.shape) * 0.04)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(gray, background, scale=255)
    _, ink_mask = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    # Remove faint paper blemishes / single-pixel noise, retaining real ink.
    connected_count, connected_labels, stats, _ = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8
    )
    clean = np.zeros_like(ink_mask)
    min_component_area = max(5, ink_mask.size // 140000)
    for k in range(1, connected_count):
        x, y, w, h, area = map(int, stats[k])
        if (area >= min_component_area and
                w < source.width * 0.35 and
                h < source.height * 0.35):
            clean[connected_labels == k] = 255

    # Horizontal ink projection gives good line boundaries on pages with a
    # few well-spaced lines, including low-resolution messaging-app photos.
    projection = (clean > 0).sum(axis=1).astype(np.float32)
    smooth = cv2.GaussianBlur(projection[:, None], (1, 13), 3.0).ravel()
    threshold = max(4.0, float(smooth.max()) * 0.035)
    active = smooth > threshold
    bands = []
    start = None
    for y, foreground in enumerate(active):
        if foreground and start is None:
            start = y
        elif not foreground and start is not None:
            if y - start >= max(6, round(source.height * .008)):
                bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, source.height))

    if bands:
        median_height = float(np.median([b - a for a, b in bands]))
        # Attach detached Gujarati upper vowel marks to their lower text line,
        # instead of treating detached accents as a whole independent line.
        merged = []
        i = 0
        while i < len(bands):
            a, b = bands[i]
            if (b - a < 0.38 * median_height and i + 1 < len(bands) and
                    bands[i + 1][0] - b <= max(5, .32 * median_height)):
                merged.append((a, bands[i + 1][1]))
                i += 2
            else:
                merged.append((a, b))
                i += 1
        bands = merged

    heights = [b - a for a, b in bands]
    separated_lines = (
        len(bands) >= 2 and len(bands) <= 40 and
        max(heights) < 1.75 * np.median(heights)
    )
    groups = []
    if separated_lines:
        for y0, y1 in bands:
            band = clean[y0:y1, :]
            line_h = y1 - y0
            kernel_width = max(7, round(line_h * 0.28))
            joined = cv2.dilate(
                band, cv2.getStructuringElement(
                    cv2.MORPH_RECT, (kernel_width, 3)
                )
            )
            contours, _ = cv2.findContours(
                joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            proposals = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                ink_count = cv2.countNonZero(clean[y0 + y:y0 + y + h, x:x + w])
                if (w >= max(10, line_h * .16) and
                        h >= max(8, line_h * .22) and
                        ink_count >= max(12, line_h * .5)):
                    proposals.append((x, y0 + y, w, h))
            proposals.sort(key=lambda b: b[0])
            if proposals:
                groups.append(proposals)
    else:
        # On dense pages, projections may join neighboring lines. In that
        # case, find word contours globally, group by vertical word centers.
        kernel_width = max(9, round(source.width * 0.007))
        joined = cv2.dilate(
            clean, cv2.getStructuringElement(
                cv2.MORPH_RECT, (kernel_width, 3)
            )
        )
        contours, _ = cv2.findContours(
            joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            foreground = cv2.countNonZero(clean[y:y + h, x:x + w])
            if w >= 17 and h >= 12 and foreground >= 25:
                candidates.append((x, y, w, h))
        if candidates:
            typical = float(np.median([b[3] for b in candidates]))
            candidates = [b for b in candidates if
                          b[3] >= max(12, typical * .33)]
            centers = []
            for item in sorted(candidates, key=lambda b: b[1] + b[3]/2):
                cy = item[1] + item[3]/2
                nearest = min(
                    centers,
                    key=lambda g: abs(g['cy'] - cy), default=None
                )
                if nearest is not None and abs(nearest['cy'] - cy) <= max(10, typical*.75):
                    nearest['boxes'].append(item)
                    nearest['cy'] = float(np.median([
                        b[1] + b[3]/2 for b in nearest['boxes']
                    ]))
                else:
                    centers.append({'cy': cy, 'boxes': [item]})
            groups = [sorted(g['boxes'], key=lambda b: b[0]) for g in
                      sorted(centers, key=lambda g: g['cy'])]

    debug = rgb.copy()
    crops, line_ids = [], []
    if not groups:
        return [], Image.fromarray(debug), []

    median_word_height = float(np.median([
        b[3] for group in groups for b in group
    ]))
    all_word_rectangles = []
    for line_number, group in enumerate(groups):
        for x, y, w, h in group:
            # A tiny detached mark (comma, colon dot, accent, background speck)
            # is not an isolated handwritten *word*. Do not send it to CRNN,
            # whose nearest classes can otherwise be digits ૧ / ૨ / ૬.
            if (h < median_word_height * .47 and
                    w < median_word_height * .48):
                cv2.rectangle(debug, (x, y), (x+w, y+h), (255, 165, 0), 2)
                continue
            margin = max(3, round(h * .10))
            left, top = max(0, x-margin), max(0, y-margin)
            right, bottom = (min(source.width, x+w+margin),
                             min(source.height, y+h+margin))
            crops.append(source.crop((left, top, right, bottom)))
            all_word_rectangles.append((left, top, right, bottom))
            line_ids.append(line_number)
            cv2.rectangle(debug, (left, top), (right, bottom), (225, 55, 55), 2)

    # Visualization of probable punctuation: orange boxes are NOT fed to the
    # word recognizer. OCR V2 was trained on words, not punctuation crops.
    # Do not confuse tiny commas/periods with Gujarati digits.
    for k in range(1, connected_count):
        x, y, w, h, area = map(int, stats[k])
        if not (min_component_area <= area <= max(22, median_word_height * 0.9)
                and w <= median_word_height * .5
                and h <= median_word_height * .55):
            continue
        cx, cy = x + w / 2, y + h / 2
        inside_word = any(
            (left <= cx <= right and top <= cy <= bottom)
            for left, top, right, bottom in all_word_rectangles
        )
        if inside_word:
            continue
        # A tiny mark is plausible punctuation only near an actual word.
        near_word = any(
            (0 <= x - right <= median_word_height * .55 and
             top - median_word_height * .35 <= cy <= bottom + median_word_height * .30)
            for left, top, right, bottom in all_word_rectangles
        )
        if near_word:
            cv2.rectangle(debug, (x, y), (x+w, y+h), (255, 165, 0), 2)

    return crops, Image.fromarray(debug), line_ids

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
