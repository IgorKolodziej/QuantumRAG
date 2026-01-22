# Grover-Enhanced RAG for QA

This repository is a research prototype that combines dense retrieval with a Grover-inspired top-k selector and compares LLM answers with and without context. It includes a Streamlit demo and evaluation scripts using the SQuAD 1.1 dataset.

## Key Features
- Dense retrieval with `mixedbread-ai/mxbai-embed-large-v1` embeddings and FAISS cosine search.
- GroverTopK selection implemented with Qiskit Aer.
- Multi-model comparisons via Hugging Face Inference (llama-3-8b, mixtral-8x7b, phi-3.5 model keys).
- Benchmark scripts that export CSV/JSON results; plots are generated from the notebooks in `tests/evaluation/`.

## Architecture Overview
1. A user question is prefixed with the retrieval prompt and embedded by the transformer model.
2. `ContextRetriever` searches a FAISS `IndexFlatIP` index over SQuAD contexts.
3. `GroverTopK` selects the top-k contexts from the top-10 candidates using a dynamic threshold and Grover iterations.
4. `AgentHandler` queries Hugging Face Inference for each model and returns answers with and without context.

## Repository Layout
- `app.py`: Streamlit entrypoint for the interactive demo.
- `src/components/`: core logic (retrieval, Grover selection, LLM client).
- `src/utils/`: embedding and dataset helpers.
- `tests/evaluation/`: benchmark runners, artifacts, and plots.
- `squad_dataset/`: SQuAD 1.1 data used by the evaluation scripts.
- `saved_embeddings/`: cached embeddings and document lists for retrieval.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_API_KEY=your_hf_token
streamlit run app.py
```

The demo loads embeddings from `saved_embeddings/` and builds a FAISS index on startup. If you change the dataset or embedding model, regenerate embeddings using the `ContextRetriever` class and update the file paths in `app.py`.

## Configuration
- LLM models are defined in `src/components/AgentHandler.py` (model keys map to Hugging Face model IDs).
- Retrieval defaults are in `app.py`:
  - `MODEL_NAME` and `QUERY_PREFIX` for embeddings.
  - `TOP_K_FIRST` (default 10) and `TOP_K_FINAL` (default 3).
- `GroverTopK` parameters (threshold, shots, top_k) are configurable in `src/components/GroverTopK.py`.

## Evaluation and Benchmarks
Run the quality and timing experiments (requires `HF_API_KEY` and network access):
```bash
python tests/evaluation/tests_runner.py
python tests/evaluation/time_test_runner.py
```

Outputs:
- Quality: `tests/test_results.csv`, `tests/test_results.json`
- Timing: `tests/time_test_result.csv`, `tests/time_test_results.json`

The repository also includes prior results and plots under `tests/evaluation/` and a narrative summary in `tests/tests_summary.md`.

## Data and Artifacts
- `squad_dataset/train-v1.1.json` is the SQuAD 1.1 dataset used by the evaluation scripts.
- `saved_embeddings/` contains embeddings and document lists for the default model and dataset.
- Plots and UI screenshots live in `tests/evaluation/results_images/` and `tests/evaluation/GUI_images/`.

## Notes and Limitations
- LLM calls use the Hugging Face Inference API and are subject to rate limits and model availability.
- GroverTopK runs on the Qiskit Aer simulator rather than quantum hardware.
- Results are research-oriented and not intended as production benchmarks.

## Acknowledgements
This project builds on SQuAD, Qiskit, FAISS, and Hugging Face tooling.

## License
No license file is currently included in this repository.
