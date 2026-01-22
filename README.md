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

## Results (Included Artifacts)
The plots and numbers below are from the included evaluation summary (`tests/tests_summary.md`) and the saved artifacts in `tests/evaluation/`.

### Test 1: Answer Quality (SQuAD 1.1, 63 questions)
- Objective: compare Grover vs. classic context selection across LLMs and context variants (`no_context`, `top1`, `top3`).
- Metrics: word overlap (with vs. without context) and cosine similarity vs. the ideal answer.
- Key insights from the recorded summary:
  - Top-3 contexts outperform top-1; `llama-3-8b` reaches the highest cosine similarity (0.80).
  - Grover and classic selection are nearly identical (quality differences <0.5%).
  - No-context variants show a significant quality drop across models.

![Cosine Similarity Comparison](tests/evaluation/results_images/cosine_summary_plot.png)

### Test 2: End-to-End Timing (SQuAD 1.1, 56 questions)
Reported averages (end-to-end timing, including context selection):
| Component | Time |
| --- | --- |
| Context retrieval (top-10) | 0.297 s |
| Grover selection (top-3) | 0.030 s |
| Answer generation (`mixtral-8x7b`) | 1.18 s |
| Answer generation (`llama-3-8b`) | 2.56 s |
| Answer generation (`phi-3.5`) | 2.65 s |

Additional findings from the recorded summary:
- Context consistency: 99.11% match between Grover and classic selection; a single discrepancy occurred when Grover’s dynamic threshold excluded all contexts.
- Quality validation: `top3` > `top1` > `no_context`; `llama-3-8b` again achieved the highest cosine similarity (0.80).

![Timing vs Quality](tests/evaluation/results_images/cosine_summary_plot2.png)

### Key Conclusions (Summary)
- Grover adds ~30 ms latency vs. classic selection while maintaining selection quality.
- Context is critical: three contexts substantially improve accuracy vs. no context.
- Model tradeoff: `mixtral-8x7b` is fastest, `llama-3-8b` is most accurate on this benchmark.

## UI Demo
The Streamlit interface supports model comparison, context inspection, and collapsible answers:

| Screen | Preview |
| --- | --- |
| Home | ![Home](tests/evaluation/GUI_images/home_page.png) |
| Answers | ![Answers](tests/evaluation/GUI_images/answer_page2.png) |
| Top Contexts | ![Contexts](tests/evaluation/GUI_images/top3.png) |

## Data and Artifacts
- `squad_dataset/train-v1.1.json` is the SQuAD 1.1 dataset used by the evaluation scripts.
- `saved_embeddings/` contains embeddings and document lists for the default model and dataset.
- Plots and UI screenshots live in `tests/evaluation/results_images/` and `tests/evaluation/GUI_images/`.

## Notes and Limitations
- LLM calls use the Hugging Face Inference API and are subject to rate limits and model availability.
- GroverTopK runs on the Qiskit Aer simulator rather than quantum hardware.
- Results are research-oriented and not intended as production benchmarks.

## Why Grover Here?
Grover's algorithm is designed for unstructured search with a marked-item oracle. In this project it is used as a quantum-inspired top-k selector on a small, already ranked candidate set (top-10 from FAISS), mainly to explore the integration of quantum techniques in a RAG pipeline. Because the selection runs on a simulator and the search space is small, it does not offer a practical speedup over classical selection. For production retrieval quality or efficiency, classical approaches (e.g., improved embeddings, larger ANN indices, or reranking) are typically more impactful.

## GroverTopK Mechanics
The Grover selector in `src/components/GroverTopK.py` works as follows:
1. **Thresholding**: a dynamic threshold is adjusted to target roughly `k` items above the similarity cutoff.
2. **Marked indices**: all candidates exceeding the threshold are marked.
3. **Oracle**: a Grover oracle is built by applying multi-controlled X gates (MCX) to mark the target bitstrings.
4. **Diffusion**: a standard diffusion operator is applied to amplify marked states.
5. **Sampling**: the circuit runs on the Qiskit Aer simulator for a fixed number of shots.
6. **Top-k extraction**: the most frequent bitstrings are mapped back to candidate indices to return the top contexts.

## Acknowledgements
This project builds on SQuAD, Qiskit, FAISS, and Hugging Face tooling.

## License
MIT License. See `LICENSE`.
