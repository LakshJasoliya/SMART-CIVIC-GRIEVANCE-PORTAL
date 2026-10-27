# Smart Civic Grievance Portal — Gujarati CRNN V2 implementation

This folder upgrades your ORIGINAL starter application to use your **trained CRNN V2 checkpoint** from Kaggle (Epoch 20).

## What is included

- `gujarati_ocr.py`: exact V2 architecture, preprocessing (48 x 320), checkpoint loader, word predictions, experimental page segmentation.
- `app_trained_ocr.py`: Streamlit app for typed complaints, Gujarati word images, experimental page OCR, English images, and local Ollama complaint analysis.
- `requirements.txt`: local Python dependencies.
- `test_model.py`: local checkpoint loader / one-word prediction verification.
- `Smart_Civic_Gujarati_OCR_Kaggle_OLD_V1.ipynb`: the original baseline notebook from your ZIP, preserved for reference ONLY. It is **not** the Kaggle notebook you used for successful CRNN V2 training. Your saved Kaggle Version #2 has the actual final training run.

**IMPORTANT:** The trained `.pt` checkpoint is **not included**, because you didn't upload your downloaded model weights with the ZIP. Copy the file yourself in step 1.

## Step 1 — Put your trained model in this folder

Your folder should look like:

```text
D:\civic_ocr\
  app_trained_ocr.py
  gujarati_ocr.py
  gujarati_crnn_v2_best.pt       <-- copy YOUR downloaded Epoch 20 file here
  gujarati_crnn_v2_latest.pt     <-- optional backup; NOT needed for inference
  requirements.txt
  test_model.py
  README.md
```

We recommend backing up your original `D:\civic_ocr` folder before replacing these files.

## Step 2 — Install Python dependencies in PowerShell

```powershell
cd D:\civic_ocr
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The above commands avoid Windows PowerShell's virtual-environment activation restrictions.

## Step 3 — Verify trained model LOADS

```powershell
.\.venv\Scripts\python.exe test_model.py
```

You should see model architecture `GujaratiCRNNv2`, Epoch `20`, and image size `320 x 48`. The model can run on **CPU**, so a local GPU is not required.

To test a cropped single handwritten word image:

```powershell
.\.venv\Scripts\python.exe test_model.py "D:\path\to\my_word.png"
```

Use a tightly cropped **single word**, not a whole handwritten page.

## Step 4 — Ollama and Streamlit

Install and run Ollama locally: https://ollama.com/

```powershell
ollama pull llama3.2
ollama pull llava
.\.venv\Scripts\python.exe -m streamlit run app_trained_ocr.py
```

Make sure the Ollama server is running before using complaint classification and the English photo tab. Gujarati recognition of one word does **not** require Ollama.

## Caveats and results

- Kaggle independent test: **78.45% exact word accuracy** and **4.51% character error rate** on **16,490 isolated handwritten Gujarati word images**.
- These metrics **do not** measure full handwritten document transcription. The `Gujarati Letter (Experimental)` tab uses only rough OpenCV segmentation; it can split/merge words or misread lines. The user MUST review/correct the transcript before submission.
- Real-world municipal complaint photos may differ from the training dataset; quality must be tested before deployment. No accuracy is guaranteed on phone-camera pages.
- The department list is only a demo mapping; replace with real local authorities. `Session Tracking` is in-memory and not a shared database.
- The included Streamlit app is a **separate prototype**. If your actual Smart Civic Grievance Portal app lives in another `app.py`, it was **not** part of the uploaded ZIP and has not been integrated yet.
- Never feed unreviewed text from uncertain OCR directly into a final official action.
