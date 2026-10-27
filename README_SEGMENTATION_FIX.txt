SMART CIVIC GRIEVANCE PORTAL — ADAPTIVE SEGMENTATION EXPERIMENT

IMPORTANT
- This is an incremental improvement to RED WORD BOX proposals, not a guaranteed full-page OCR solution.
- Retains your existing Gujarati CRNN V2 checkpoint and your Ollama correction feature.
- The trained model recognizes CROPPED WORDS and has not learned punctuation or full-page layout.
- Low-resolution/compressed handwriting may still be incorrectly recognized, even when boxes look right.

INSTALL IN YOUR EXISTING D:\civic_ocr FOLDER
1. Stop Streamlit with Ctrl+C.
2. Back up your current app_trained_ocr.py and gujarati_ocr.py.
3. Copy the NEW app_trained_ocr.py and gujarati_ocr.py into D:\civic_ocr and confirm replacement.
4. (Optional) Copy test_segmentation.py into D:\civic_ocr.
5. DO NOT replace gujarati_crnn_v2_best.pt or your .venv folder.
6. In Command Prompt:
       cd /d D:\civic_ocr
       .venv\Scripts\activate
       python -m streamlit run app_trained_ocr.py
7. Use "Gujarati Letter (Experimental)" and upload the same photo.
8. Click "Extract Gujarati Text (Experimental)" and inspect word boxes BEFORE using AI corrections.
9. Never confirm a garbled OCR transcript as correct; manually edit words from the photo.

OPTIONAL BOUNDING BOX DIAGNOSTIC
  python test_segmentation.py "C:\path\to\my_photo.jpg"
  Look beside the input for my_photo_word_boxes.png.

HOW THIS VERSION DIFFERS
- Normalizes uneven lighting and paper texture before finding ink strokes.
- Adjusts working resolution across small and large uploads.
- Uses horizontal ink-profile line bands when line spacing is distinct.
- Falls back to component-based grouping for dense text lines.
- Suppresses small detached blobs from the word OCR path.
- Annotates candidate words in red and likely tiny non-word marks in orange.
- Warns when image resolution is low, including 500x209 WhatsApp-style images.

LOCAL TESTS (NOT A GUARANTEE)
- High-resolution example (1547x1017): 102 candidate word regions, 17 lines.
- Low-resolution 500x209 example: 36 candidate word regions, 5 lines.
- Old application detected 105/17 and 61/12 respectively.
- On a synthetic set of resized images, high-resolution example yielded 17 lines
  at widths 360-2400, and low-resolution example yielded 5 lines at widths 360-1600.
- The low-resolution example STILL had very poor WORD RECOGNITION when tested
  with your actual Epoch-20 model. This fix alone cannot solve model generalization.

NEXT REAL IMPROVEMENTS
- Collect genuine photos with varied handwriting and phone lighting;
  annotate ground-truth word crops for a held-out test set.
- Fine-tune/retrain with realistic data augmentations and supervised word labels.
- For high-stakes complaint intake, require citizen confirmation or a
  separate verified Gujarati document OCR service/model.
- Punctuation remains experimental. The app deliberately does NOT guess
  unreadable isolated marks as Gujarati numerals or punctuation.
