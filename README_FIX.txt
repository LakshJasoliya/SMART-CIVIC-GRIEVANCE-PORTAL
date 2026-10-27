GUJARATI OCR / OLLAMA PATCH — TWO ISSUES
======================================

What changed:
- Replaces old page line grouping (which split a page into far too many lines).
- Adds debug view of first 16 detected word crops.
- Requires manual transcription review and detects obviously unreadable OCR text.
- Makes Ollama return a much smaller JSON answer with a bounded token budget.
- Explains the Ollama token repeat limit error instead of silently continuing.
- Preserves the reviewed text verbatim and uses example category-to-department mappings.

IMPORTANT LIMITATION
The trained model is a SINGLE-WORD OCR recognizer. Even after the line
segmentation patch, a real/generated page can still produce unreadable OCR.
A meaningful full-page OCR solution requires training/validation on page word
crops, better segmentation, or a proven full-page Gujarati handwriting model.

WINDOWS INSTALL (one step at a time)
1. Press Ctrl+C in the Command Prompt running Streamlit.
2. Make a backup of D:\civic_ocr\gujarati_ocr.py and app_trained_ocr.py.
3. Replace both Python files in D:\civic_ocr with the files in this ZIP.
4. Keep gujarati_crnn_v2_best.pt unchanged. Do NOT retrain.
5. Restart in D:\civic_ocr:
   .venv\Scripts\activate
   python -m streamlit run app_trained_ocr.py
6. Upload the same Gujarati letter; Extract Gujarati Text (Experimental).
7. Inspect boxes and first 16 word crops, correct the transcript manually.
8. Tick the confirmation checkbox before analysis. Unreadable text is blocked.

If the Ollama token repeat error still occurs on CLEAN typed Gujarati/English
complaints, this is a separate Ollama/model problem, not an OCR issue.
Check: ollama list
Try: ollama run llama3.2

NOTE: Not tested against a running Ollama service in this environment.
