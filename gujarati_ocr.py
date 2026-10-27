"""Gujarati handwritten WORD OCR using CRNN V2/V5 checkpoints.

This recognizer is trained on isolated words. The page segmentation is
experimental, with special handling for lined notebook paper. Human review
is required for all full-document OCR results.
"""
from pathlib import Path
import unicodedata

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from gujarati_beam_search import ctc_beam_search

IMAGE_H = 48
IMAGE_W = 320
DEFAULT_MODEL_FILENAME = "gujarati_crnn_v5_control.pt"


def prepare_word_image(image: Image.Image, height=IMAGE_H, width=IMAGE_W, *,
                       segmented_page_crop=False):
    """Normalize a word for the CRNN recognizer.

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
    """Same module names, layer order, and tensor shapes as Kaggle CRNN V2/V5."""

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
                "Checkpoint architecture mismatch: expected GujaratiCRNNv2."
            )
        self.height = int(checkpoint["image_height"])
        self.width = int(checkpoint["image_width"])
        if (self.height, self.width) != (IMAGE_H, IMAGE_W):
            raise ValueError(
                f"Unexpected checkpoint image size {self.width}x{self.height}; "
                f"expected {IMAGE_W}x{IMAGE_H}."
            )
        self.chars = list(checkpoint["chars"])

        if len(self.chars) not in (79, 257):
            raise ValueError(
                f"Unsupported OCR vocabulary size: {len(self.chars)}. "
                "Expected 79 (V2) or 257 (V5)."
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = GujaratiCRNNv2(len(self.chars) + 1)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.to(self.device).eval()
        self.checkpoint_epoch = checkpoint.get(
            "epoch",
            checkpoint.get("v5_epoch", checkpoint.get("base_epoch"))
        )
        self.best_cer = checkpoint.get("best_cer")
    def _decode_logits(self, logits: torch.Tensor) -> list[str]:
        """Use validated beam decoding for V5; retain greedy for legacy V2."""
        if len(self.chars) != 257:
            return ctc_greedy_decode(logits, self.chars)
        log_probs = logits.log_softmax(dim=2).detach().cpu().numpy().astype(np.float64)
        return [
            ctc_beam_search(item, self.chars, beam_width=10, top_k=16)
            for item in log_probs
        ]

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
        return self._decode_logits(self.model(x))[0]

    @torch.inference_mode()
    def recognize_words(self, images: list[Image.Image], batch_size=16) -> list[str]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        predictions = []
        for start in range(0, len(images), batch_size):
            group = images[start:start + batch_size]
            arrays = np.stack([
                prepare_word_image(im, self.height, self.width,
                                   segmented_page_crop=True)
                for im in group
            ])
            x = torch.from_numpy(arrays).unsqueeze(1).float().div(255.0).to(self.device)
            predictions.extend(self._decode_logits(self.model(x)))
        return predictions


def _segment_document_legacy(image: Image.Image):
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
    # word recognizer. This OCR is trained on words, not punctuation crops.
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



def _ruled_page_ink(rgb):
    """Produce a foreground mask on a photographed ruled paper page.

    Only used for page segmentation, not as ground-truth text or model output.
    """
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sigma = max(10.0, min(gray.shape) * 0.035)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    normalized = cv2.divide(gray, bg, scale=255)
    mask = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 39, 11
    )

    # Dark table outside the paper should not count as ink. The red/brown
    # finger visible in the sample should not count as handwriting either.
    local_brightness = cv2.GaussianBlur(gray, (0, 0), sigmaX=12)
    paper = np.uint8(local_brightness > 100) * 255
    paper = cv2.morphologyEx(
        paper, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    )
    paper = cv2.erode(
        paper, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )
    r, g, b = (rgb[:, :, j].astype(np.float32) for j in range(3))
    skin = ((r > 90) & (r - g > 20) & (r - b > 27) & (g > 40))
    mask[(paper == 0) | skin] = 0
    mask[:3, :] = 0
    mask[-3:, :] = 0
    mask[:, :3] = 0
    mask[:, -3:] = 0
    return mask


def _ruled_line_angle(mask):
    """Return deskew angle if the image has many parallel ruled lines."""
    import cv2
    h, w = mask.shape
    segments = cv2.HoughLinesP(
        mask, 1, np.pi / 180,
        threshold=max(50, int(w * 0.095)),
        minLineLength=int(w * 0.23),
        maxLineGap=int(w * 0.045),
    )
    if segments is None:
        return None
    angles, positions = [], []
    # HoughLinesP may return (N, 1, 4) or (N, 4) across OpenCV builds.
    segments = np.asarray(segments).reshape(-1, 4)
    for item in segments:
        x1, y1, x2, y2 = (int(v) for v in item)
        dx = x2 - x1
        if abs(dx) < w * 0.23:
            continue
        theta = float(np.degrees(np.arctan2(y2 - y1, dx)))
        if abs(theta) > 7:
            continue
        angles.append(theta)
        positions.append(y1 + (w * 0.5 - x1) * (y2 - y1) / dx)
    if len(angles) < 12:
        return None
    if len(set(round(y / max(4, h * 0.017)) for y in positions)) < 7:
        return None
    return float(np.median(angles))


def _ruled_baselines(mask):
    """Locate the actual printed notebook rules, not characters/words."""
    import cv2
    h, w = mask.shape
    segments = cv2.HoughLinesP(
        mask, 1, np.pi / 180,
        threshold=max(50, int(w * 0.09)),
        minLineLength=int(w * 0.19),
        maxLineGap=int(w * 0.04),
    )
    if segments is None:
        return []

    votes = np.zeros(h, dtype=np.float32)
    # HoughLinesP may return (N, 1, 4) or (N, 4) across OpenCV builds.
    segments = np.asarray(segments).reshape(-1, 4)
    for item in segments:
        x1, y1, x2, y2 = (int(v) for v in item)
        width = abs(x2 - x1)
        theta = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if width < w * 0.19 or abs(theta) > 1.3:
            continue
        y = int(round((y1 + y2) / 2))
        if 0 <= y < h:
            votes[y] += width
    votes = cv2.GaussianBlur(votes[:, None], (1, 11), 2.1).ravel()
    if votes.max() <= 0:
        return []

    threshold = max(90.0, float(votes.max()) * 0.10)
    candidates = [
        (votes[y], y) for y in range(5, h - 5)
        if votes[y] >= votes[y - 1] and votes[y] > votes[y + 1]
        and votes[y] > threshold
    ]
    centers = []
    for _, y in sorted(candidates, reverse=True):
        if all(abs(y - other) > w * 0.025 for other in centers):
            centers.append(y)
    return sorted(centers)


def _segment_ruled_page(image: Image.Image):
    """Attempt line-guided segmentation on a ruled-paper photo.

    Return None when detection is unreliable so the unchanged legacy
    segmentation can handle unruled documents.
    """
    import cv2
    source = ImageOps.exif_transpose(image).convert('RGB')
    if min(source.size) < 80:
        return None

    factor = min(2.2, 1050 / source.width, 1800 / source.height)
    if factor > 1.01:
        source = source.resize(
            (round(source.width * factor), round(source.height * factor)),
            Image.Resampling.BICUBIC,
        )
    rgb = np.asarray(source).copy()
    height, width = rgb.shape[:2]
    mask = _ruled_page_ink(rgb)
    angle = _ruled_line_angle(mask)
    if angle is None:
        return None

    if abs(angle) > 0.25:
        transform = cv2.getRotationMatrix2D(
            (width / 2, height / 2), angle, 1.0
        )
        rgb = cv2.warpAffine(
            rgb, transform, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )
        mask = _ruled_page_ink(rgb)

    lines = _ruled_baselines(mask)
    # At least 12 regular rules are required to enter this special mode.
    # This prevents a short 5-line unruled complaint from switching modes.
    if len(lines) < 12:
        return None
    spacings = np.diff(lines)
    pitch = float(np.median(spacings))
    if pitch < 17 or pitch > height * 0.075:
        return None

    # Lines can be partially occluded by writing/fingers; recover a regular
    # grid from the neighboring stable ruled lines.
    extras = []
    for a, b in zip(lines, lines[1:]):
        count = int(round((b - a) / pitch))
        if 2 <= count <= 4:
            extras.extend(
                round(a + (b - a) * k / count)
                for k in range(1, count)
            )
    lines = sorted(set(lines + extras))
    while lines and lines[0] - pitch > 55:
        lines.insert(0, round(lines[0] - pitch))
    while lines and lines[-1] + pitch < height - 30:
        lines.append(round(lines[-1] + pitch))

    # Suppress long horizontal paper rules for grouping, while leaving the
    # original photo available for human inspection.
    ruling = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(80, width // 8), 1)
        ),
    )
    ruling = cv2.dilate(
        ruling, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    )
    clean = cv2.subtract(mask, ruling)
    for y in lines:
        cv2.rectangle(
            clean, (4, max(0, y - 1)),
            (width - 5, min(height - 1, y + 1)), 0, -1
        )

    # Brighten the paper and gently strengthen faint handwriting in word
    # images. Do not alter the CRNN's 48x320 preprocessing or model weights.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=max(10, min(gray.shape) * 0.035)
    )
    flat = cv2.divide(gray, bg, scale=255)
    enhanced = np.uint8(
        np.clip(255.0 - (255.0 - flat.astype(np.float32)) * 1.8, 0, 255)
    )

    crops, line_ids = [], []
    overlay = rgb.copy()
    line_number = 0
    for prev, rule in zip(lines[:-1], lines[1:]):
        if not 15 <= rule - prev <= 1.6 * pitch:
            continue
        top = max(0, prev + 3)
        bottom = min(height, rule + 6)
        band = clean[top:bottom, :]
        if band.size == 0:
            continue

        # Keep components from an individual writing line together without
        # allowing the notebook lines to join adjacent text lines.
        kernel_width = max(11, min(28, round(pitch * 0.50)))
        joined = cv2.dilate(
            band,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (kernel_width, 3)
            ),
        )
        contours, _ = cv2.findContours(
            joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for contour in contours:
            x, yy, ww, hh = cv2.boundingRect(contour)
            foreground = cv2.countNonZero(
                band[yy:yy + hh, x:x + ww]
            )
            if (x < width * .04 or x + ww > width * .985 or
                ww < max(9, pitch * .30) or
                hh < max(9, pitch * .28) or
                foreground < max(12, pitch * .75)):
                continue
            boxes.append((x, top + yy, ww, hh))
        if not boxes:
            continue

        for x, y, ww, hh in sorted(boxes, key=lambda b: b[0]):
            margin = max(3, round(hh * .10))
            left, top_y = max(0, x - margin), max(0, y - margin)
            right, bottom_y = (
                min(width, x + ww + margin),
                min(height, y + hh + margin),
            )
            crops.append(
                Image.fromarray(
                    enhanced[top_y:bottom_y, left:right].copy(),
                    mode='L',
                )
            )
            line_ids.append(line_number)
            cv2.rectangle(
                overlay, (left, top_y), (right, bottom_y),
                (225, 55, 55), 2
            )
        line_number += 1

    if not crops:
        return None
    return crops, Image.fromarray(overlay), line_ids


def segment_document_roughly(image: Image.Image):
    """Streamlit-compatible (word crops, bounding-box image, line IDs).

    Use rule-guided segmentation only on verified lined-paper documents,
    otherwise preserve the original tested segmentation method.
    """
    special = _segment_ruled_page(image)
    if special is not None:
        return special
    return _segment_document_legacy(image)

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
