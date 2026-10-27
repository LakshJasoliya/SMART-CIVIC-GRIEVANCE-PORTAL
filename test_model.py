"""Verify CRNN V2 checkpoint loading, optionally recognize one cropped Gujarati word.

Usage: python test_model.py [path_to_cropped_word.png]
"""
import sys
from pathlib import Path
from PIL import Image
from gujarati_ocr import GujaratiWordOCR, DEFAULT_MODEL_FILENAME

checkpoint = Path(__file__).resolve().parent / DEFAULT_MODEL_FILENAME

try:
    ocr = GujaratiWordOCR(checkpoint)
except Exception as exc:
    print("FAILED TO LOAD OCR MODEL:", exc)
    raise SystemExit(1)

print("MODEL LOADED SUCCESSFULLY")
print("Architecture: GujaratiCRNNv2")
print("Epoch:", ocr.checkpoint_epoch)
print("Image size:", f"{ocr.width} x {ocr.height}")
print("Characters:", len(ocr.chars))
print("Device:", ocr.device)
print("Best validation CER:", round(ocr.best_cer * 100, 2), "%")

if len(sys.argv) > 1:
    image_path = Path(sys.argv[1])
    with Image.open(image_path) as image:
        print("Predicted Gujarati word:", ocr.recognize_word(image))
else:
    print("To recognize a word, pass a path to a tightly cropped word image.")
