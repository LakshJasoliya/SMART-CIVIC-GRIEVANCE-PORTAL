"""Quick diagnostic: python test_segmentation.py path_to_photo.jpg"""
from pathlib import Path
import sys
from PIL import Image
from gujarati_ocr import segment_document_roughly

if len(sys.argv) != 2:
    raise SystemExit('Usage: python test_segmentation.py "path_to_photo.jpg"')

path = Path(sys.argv[1])
with Image.open(path) as photo:
    print(f'Original resolution: {photo.width}x{photo.height}')
    crops, overlay, line_ids = segment_document_roughly(photo)

out = path.with_name(path.stem + '_word_boxes.png')
overlay.save(out)
print(f'Detected {len(crops)} word regions in {len(set(line_ids))} approximate lines')
print('Saved annotated preview:', out)
print('Red = likely word; orange = tiny mark (not passed to word OCR)')
