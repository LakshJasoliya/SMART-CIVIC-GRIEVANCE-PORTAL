# Smart Civic Grievance Portal

An offline-first AI/ML prototype that helps process handwritten Gujarati
civic complaints. It combines a CRNN-based Gujarati handwriting OCR
model with OpenCV document segmentation and a locally running LLM to
support complaint analysis.

> **Project status:** Experimental prototype. OCR output can contain
> errors, especially with faint handwriting, ruled paper, tilted
> photographs, and tightly spaced words. Review the extracted text
> before relying on the AI-generated complaint analysis.

## Overview

Citizens may submit civic complaints as handwritten Gujarati notes. This
project explores a local workflow for extracting the text and organizing
the complaint information for review.

The application uses: 1. **OpenCV-based image processing and
segmentation** to propose text/word crops from a photographed complaint.
2. **CRNN + CTC OCR** to recognize Gujarati handwriting. 3. **CTC prefix
beam search** to decode model predictions and improve on greedy decoding
in evaluation. 4. **Ollama with Llama 3.2** to analyze the extracted
complaint text and suggest relevant civic information such as
category/department, severity, and location. 5. **Streamlit** for an
interactive local interface where users can inspect the extracted text
and analysis.

The application is intended as a prototype for experimentation and
human-reviewed civic complaint triage---not as an official municipal
filing system.

## Features

-   Gujarati handwritten word recognition using a custom-trained CRNN
    model.
-   Image preprocessing and experimental text/word segmentation with
    OpenCV.
-   CTC beam-search decoding (beam width 10, top-k 16 in the evaluated
    configuration).
-   Local complaint analysis using Ollama and Llama 3.2.
-   Streamlit interface for uploading complaint images and reviewing the
    OCR output.
-   Local-first inference: the OCR model and LLM can run on the user's
    own machine without a cloud AI API.

## Technology Stack

-   **Language:** Python
-   **OCR / Deep Learning:** PyTorch, CRNN, Connectionist Temporal
    Classification (CTC)
-   **Decoding:** CTC prefix beam search
-   **Image processing:** OpenCV, Pillow, NumPy
-   **Local LLM:** Ollama, Llama 3.2
-   **Interface:** Streamlit

## OCR Evaluation

The V5 control model was evaluated on a held-out split of **30,000
isolated Gujarati handwritten-word samples**. The reported results were:

  -----------------------------------------------------------------------
  Metric                        Greedy decoding      Beam-search decoding
  ------------------- ------------------------- -------------------------
  Exact word accuracy  55.52% (16,656 / 30,000)  56.21% (16,864 / 30,000)

  Character Error                        20.80%                    20.46%
  Rate (CER)                                    
  -----------------------------------------------------------------------

Beam search improved exact word accuracy by 0.69 percentage points and
reduced CER by 0.34 percentage points on this evaluation.

**How to interpret these results:** these are isolated-word dataset
metrics, not end-to-end accuracy on complete photographed complaint
letters. Full-page results also depend on image quality, line/word
segmentation, spacing, handwriting style, and punctuation. Some Gujarati
words may be misread, omitted, or split into incorrect crops, so human
verification is required.

## Requirements

-   Python 3.10 or newer (Python 3.11 is suitable)
-   Git
-   Ollama installed locally
-   The trained model checkpoint: `gujarati_crnn_v5_control.pt`
-   The project dependencies listed in `requirements.txt` (if included
    in this repository)

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/LakshJasoliya/SMART-CIVIC-GRIEVANCE-PORTAL.git
cd SMART-CIVIC-GRIEVANCE-PORTAL
```

### 2. Create and activate a virtual environment

**Windows (Command Prompt):**

``` bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

If `requirements.txt` is present:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If it is not present, install the packages required by the application
and add them to a `requirements.txt` file before sharing the project.
Core packages include `torch`, `streamlit`, `ollama`, `opencv-python`,
`Pillow`, and `numpy`.

### 4. Add the trained OCR checkpoint

Ensure `gujarati_crnn_v5_control.pt` is available at the path expected
by `app_trained_ocr.py` (normally the project root, beside the
application file). The checkpoint is a trained model artifact and is not
recreated by simply launching the app.

### 5. Set up the local LLM

Install Ollama, then download the model:

``` bash
ollama pull llama3.2
```

Make sure the Ollama service is running before using complaint analysis.
If the model name in your local Ollama installation differs, update the
model setting in the application accordingly.

### 6. Launch the application

``` bash
streamlit run app_trained_ocr.py
```

Streamlit will print a local URL---usually
`http://localhost:8501`---that you can open in your browser.

## How to Use

1.  Start the Streamlit application.
2.  Upload a clear photograph or scan of a Gujarati handwritten civic
    complaint.
3.  Run the experimental text-extraction step.
4.  Review and correct the OCR transcript manually.
5.  Submit the corrected text for local AI analysis.
6.  Review the suggested complaint category, department, severity, and
    location before using the result.

For best results, use a sharply focused image, good lighting, minimal
shadows, a page photographed as straight-on as possible, and handwriting
with visible spacing between words.

## Core Project Files

The main application files used by the current workflow are:

-   `app_trained_ocr.py` --- Streamlit user interface and
    complaint-analysis workflow.
-   `gujarati_ocr.py` --- OCR model, preprocessing, and experimental
    document segmentation.
-   `gujarati_beam_search.py` --- CTC prefix beam-search decoder.
-   `gujarati_crnn_v5_control.pt` --- trained OCR checkpoint.
-   `requirements.txt` --- Python dependencies, if included.

Your repository may contain additional scripts, notebooks, or supporting
files.

## Limitations

-   OCR performance is not yet reliable enough for unsupervised or
    official use.
-   Handwriting styles, low-resolution photos, shadows, skew, ruled
    lines, and punctuation can substantially affect results.
-   Document segmentation is heuristic and may split a word, merge
    nearby words, or include background artifacts.
-   The local LLM can make mistakes when the OCR transcript is incorrect
    or incomplete.
-   Suggested categories, severity levels, and locations must be
    reviewed by a person.
-   The project does not itself establish an official connection to a
    municipal corporation or guarantee complaint submission or
    resolution.

## Future Improvements

-   Improve ruled-paper removal, deskewing, and line/word segmentation.
-   Evaluate performance across different writers, cameras, and document
    conditions.
-   Explore line-level Gujarati handwriting recognition and
    language-aware decoding.
-   Add confidence indicators and manual correction tools.
-   Build a larger end-to-end evaluation set of complete complaint
    documents.
-   Improve structured complaint extraction and validation before
    classification.

## Privacy Note

The intended workflow can run locally using the OCR checkpoint and
Ollama. Only use complaint images you have permission to process, and
avoid committing personal information, real addresses, phone numbers, or
private complaint images to a public repository.

## License

Add a `LICENSE` file before specifying a license for this project. Until
a license is included, do not assume that others have permission to
reuse or redistribute the code.
