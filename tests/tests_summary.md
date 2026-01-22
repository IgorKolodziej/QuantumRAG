# Comparative Evaluation: Grover + LLM

## Test 1: Answer Quality
The test was run on 63 questions for each model and variant. Questions and "ideal answers" used for comparison were taken from the SQuAD 1.1 dataset.

### Goal
Assess the impact of:
- Grover vs. classic context selection,
- different LLMs (llama-3-8b, mixtral-8x7b, phi-3.5),
- context variants (no_context, top1, top3).

### Metrics
- Word overlap (between answers with and without context).
- Cosine similarity (between generated answers and the "ideal" answers).

### Test Parameters
- Number of SQuAD 1.1 questions: **63**
- Models:
  - `llama-3-8b`
  - `mixtral-8x7b`
  - `phi-3.5`
- Context variants:
  - `no_context`
  - `classic_top1`
  - `classic_top3`
  - `grover_top1`
  - `grover_top3`

### How the Tests Were Run
The runner used for Test 1 is in `evaluation/tests_runner.py`.

### Results
Full results (metrics and answers for each variant) are in `evaluation/test_results.csv`
and `evaluation/test_results.json`.

### Summary Plots
Summary of all metrics for each model and variant:
![Summary of all metrics](evaluation/results_images/all_metrics_summary.png)

Table comparing models and variants by average cosine similarity to the "ideal" answer:
![Average cosine similarity table](evaluation/results_images/cosine_summary_table.png)

Plot of average cosine similarity to the "ideal" answers:
![Average cosine similarity plot](evaluation/results_images/cosine_summary_plot.png)

---

## Test 2: End-to-End Timing
This test repeats the evaluation in an extended form to include the full runtime of the QA system,
not only answer generation but also context selection and analysis. The test was run on 56 questions
for each model and variant. Questions and "ideal answers" were taken from SQuAD 1.1.

### Goal
Analyze end-to-end performance across different context selection variants and LLMs, including:
- time to retrieve top-10 contexts,
- time to select top-k contexts (classic and Grover),
- agreement between contexts selected by classic vs. Grover,
- answer quality (cosine similarity and word overlap).

### Test Parameters
- Number of SQuAD 1.1 questions: **56**
- Models:
  - `llama-3-8b`
  - `mixtral-8x7b`
  - `phi-3.5`
- Context variants:
  - `no_context`
  - `classic_top1`
  - `classic_top3`
  - `grover_top1`
  - `grover_top3`

### How the Tests Were Run
The runner used for Test 2 is in `evaluation/time_test_runner.py`.

### Results
Full results (metrics and answers for each variant) are in `evaluation/time_test_results.csv`
and `evaluation/time_test_results.json`.

#### 1. Average Answer Generation Time (seconds)
| Model        | Average time |
|--------------|--------------|
| llama-3-8b   | 2.56 s       |
| mixtral-8x7b | 1.18 s       |
| phi-3.5      | 2.65 s       |

---

#### 2. Average Top-k Context Selection Time (seconds)
| Variant      | Average time |
|--------------|--------------|
| grover_top3  | 0.0304 s     |
| grover_top1  | 0.0289 s     |
| classic_top3 | 0.0000 s     |
| classic_top1 | 0.0000 s     |
| no_context   | 0.0000 s     |

---

#### 3. Average Top-10 Context Retrieval Time
Constant time independent of variant:
**0.297321 s**

---

#### 4. Context Agreement (Grover vs. Classic)
**99.11%** of cases used the same contexts. There is one case where contexts differed because
Grover did not return any contexts. The probable reason is that none of the candidates exceeded
the dynamically adjusted similarity threshold.

---

#### 5. Average Cosine Similarity and Word Overlap
| Model      | Variant      | Cosine Similarity | Word Overlap (%) |
|------------|--------------|-------------------|------------------|
| llama-3-8b | classic_top1 | 0.7530            | 38.77            |
|            | classic_top3 | 0.8006            | 46.40            |
|            | grover_top1  | 0.7565            | 38.77            |
|            | grover_top3  | 0.8026            | 46.48            |
|            | no_context   | 0.5788            | 5.68             |
| mixtral-8x7b | classic_top1 | 0.5951          | 7.74             |
|            | classic_top3 | 0.6185            | 9.00             |
|            | grover_top1  | 0.5962            | 7.74             |
|            | grover_top3  | 0.6312            | 10.68            |
|            | no_context   | 0.5452            | 2.05             |
| phi-3.5    | classic_top1 | 0.6176            | 7.62             |
|            | classic_top3 | 0.6277            | 7.99             |
|            | grover_top1  | 0.6172            | 7.62             |
|            | grover_top3  | 0.6256            | 8.60             |
|            | no_context   | 0.5181            | 1.10             |

![Average cosine similarity table](evaluation/results_images/cosine_summary_table2.png)

Plot of average cosine similarity to the "ideal" answers:
![Average cosine similarity plot 2](evaluation/results_images/cosine_summary_plot2.png)

---

## Results Summary
The results confirm:
- very high agreement between Grover and classic context selection,
- very low runtime overhead for Grover,
- a clear drop in quality without context (`no_context`) across all models,
- the highest cosine similarity with three contexts and `llama-3-8b`,
- `mixtral-8x7b` generates answers fastest and `llama-3-8b` the slowest,
- selecting three contexts with Grover is only slightly slower than selecting one, and
  three contexts yield better answers.

# GUI Verification

## Home Screen
![Home screen](evaluation/GUI_images/home_page.png)

## Loading Screen
![Loading screen](evaluation/GUI_images/loading_page.png)

## Answers
![Answers](evaluation/GUI_images/answer_page.png)

![More answers](evaluation/GUI_images/answer_page2.png)

With collapsed answers (expand per model):
![Collapsed answers](evaluation/GUI_images/hidden_answers.png)

Top 3 contexts selected by Grover:
![Top 3 contexts](evaluation/GUI_images/top3.png)

Top 10 contexts selected by the classic method:
![Top 10 contexts 1](evaluation/GUI_images/top4.png)
![Top 10 contexts 2](evaluation/GUI_images/top8.png)
![Top 10 contexts 3](evaluation/GUI_images/top10.png)
