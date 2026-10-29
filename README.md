# SmartRAG — Mini RAG-Based AI Chatbot

A local, no-paid-API-key Python project for answering questions from an uploaded PDF/TXT file.

## Architecture

1. Streamlit receives one or more uploaded PDF/TXT files.
2. PyMuPDF extracts PDF text page-by-page; TXT files are split on form-feed (`\f`) characters if page boundaries are desired.
3. Text from all uploaded documents is split into overlapping chunks, preserving filename and page number.
4. Sentence Transformers (`all-MiniLM-L6-v2`) turns chunks and the question into embeddings.
5. FAISS retrieves the top-k most similar chunks.
6. Ollama runs a local LLM and generates an answer from retrieved context.
7. Streamlit displays the answer and the source chunks/pages.

## Prerequisites

- Python 3.10 or newer recommended.
- Ollama installed and running.
- Enough RAM/disk for the selected local model and embedding model.
- Internet may be needed once to download Python packages and model weights. Inference runs locally after downloads.

## Windows setup (PowerShell)

```powershell
cd SmartRAG_Project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install Ollama from its official website, then in PowerShell:
```powershell
ollama pull llama3.2
ollama list
```

Start the app:
```powershell
streamlit run app.py
```
Open the local URL Streamlit prints (normally `http://localhost:8501`).

## Ubuntu/Linux setup

```bash
cd SmartRAG_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Ollama using its official installation instructions, then:
```bash
ollama pull llama3.2
ollama list
streamlit run app.py
```

## How to use

1. Start Ollama and confirm `ollama list` shows `llama3.2`.
2. Start Streamlit with `streamlit run app.py`.
3. Upload one or more text-based PDF/TXT documents. The app searches across all uploaded documents together.
4. Ask a question in the chat input.
5. Expand **Sources used to generate this answer** to inspect the retrieved text and page numbers.
6. Try the out-of-document question in `test_questions.md`.

## Troubleshooting

- **Ollama connection error:** ensure Ollama is running and its local API is reachable at `http://localhost:11434`.
- **Model not found:** run `ollama pull llama3.2`; ensure the sidebar model name matches `ollama list`.
- **PDF has no text:** it may be scanned. OCR it first or use a text-based PDF.
- **FAISS installation error on Windows:** use a supported 64-bit Python version and update pip; if installation still fails, use a compatible Python environment or swap FAISS for ChromaDB.
- **Slow first run:** Sentence Transformers downloads its embedding model the first time; later runs can reuse the cached model.
- **Memory constraints:** use a smaller Ollama model compatible with your machine and change the sidebar model name to match `ollama list`.

## Important limitations

- This is a teaching/demo implementation, not a production security-reviewed service.
- The index is rebuilt when a document is uploaded; it is not persisted between sessions.
- Scanned PDFs require OCR.
- Similarity scores are ranking signals, not calibrated confidence values.
- The prompt helps reduce hallucinations but does not guarantee perfect grounding. Always test the out-of-document question.

## Assignment checklist

- [ ] Python source code
- [ ] Sample PDF/TXT document with at least two pages (the included TXT uses form-feed page separators)
- [ ] Screenshots of the running chatbot
- [ ] Five test questions and actual generated answers, including one unanswerable question
