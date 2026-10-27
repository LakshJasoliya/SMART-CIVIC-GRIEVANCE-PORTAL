Gujarati OCR V2 - Paper Background Normalization (Oct 2026)
==============================================================

FIX:
Blue handwritten ink on gray photographed paper was causing CRNN V2 to
predict almost only Gujarati digits and the Gujarati rupee sign.

Changes to the ONLY included Python file, gujarati_ocr.py:
- Before usual image resizing, it whitens dark/gray paper background.
- It preserves nearly white inputs.
- Keeps your existing trained GujaratiCRNNv2 architecture, checkpoint format,
  word grouping, and the model trained at epoch 20 unchanged.

IMPORTANT LIMITS:
This is a diagnostic improvement tested on one uploaded complaint photo.
Some words on the page may still be wrong and must be corrected manually.
Please do not consider this full-page OCR production ready.

INSTALL (Windows):
1) In Command Prompt where Streamlit runs: Ctrl+C to stop.
2) Backup D:\civic_ocr\gujarati_ocr.py (copy or rename).
3) Extract this ZIP and replace D:\civic_ocr\gujarati_ocr.py.
4) Keep D:\civic_ocr\app_trained_ocr.py and gujarati_crnn_v2_best.pt unchanged.
5) Run from your activated virtual environment:
       python -m streamlit run app_trained_ocr.py
6) Upload the SAME handwritten page and click Extract Gujarati Text.
7) Review bounding boxes and transcript and do not confirm unless corrected.
