# NLP Embeddings, Classification, and Retrieval

**Internship project:** This work was completed by Atanas Vitanov as part of the Loka Machine Learning Internship. References to "Week X", task numbers, and labs follow the original internship curriculum.

This project covers the Week 6 LLM fundamentals work: tokenization analysis, Word2Vec from scratch with NumPy, static vs contextual embeddings, DistilBERT text classification, GPT-2 generation and attention visualization, and semantic search with FAISS. The notebooks save reusable outputs into `results/`, and the Streamlit frontend reads those saved files without retraining models.

## Recorded comparison results

| Experiment | Baseline | Alternative | Finding |
|---|---|---|---|
| IMDB test accuracy (1,000 examples) | TF-IDF + Logistic Regression: 86.3% | DistilBERT: 87.0% | A 0.7 percentage-point accuracy gain in this experiment. |
| AG News mean precision@5 (six queries) | Keyword search: 0.90 | FAISS semantic search: 0.90 | Equal averages, with differences by query. |

The saved training times are approximately 1.43 seconds for TF-IDF and 371.85 seconds
for DistilBERT. These are measurements of the recorded runs, not a controlled
hardware-efficiency benchmark. The retrieval index contains 7,588 documents;
relevance is based on coarse category matches and only six queries, limiting
conclusions about general retrieval quality.

The collection also includes tokenization analysis, NumPy Word2Vec, static versus
contextual embeddings, and GPT-2 decoding experiments. No model weights or fitted
model artifacts are included. Saved plots, tables, summaries, and notebook outputs
remain available, and the Streamlit viewer reads those results without retraining.

## Environment Setup

Use `uv` from this directory:

```bash
cd nlp-embeddings-and-retrieval
uv sync
```

Install the notebook kernel:

```bash
uv run python -m ipykernel install --user --name nlp-embeddings-and-retrieval
```

Run JupyterLab:

```bash
uv run jupyter lab
```

Run the Streamlit frontend:

```bash
uv run streamlit run frontend/app.py
```

If port `8501` is already in use:

```bash
uv run streamlit run frontend/app.py --server.port 8502
```

## Notebook Order

Run the notebooks in this order because later tasks use saved data or results from earlier tasks:

1. `notebooks/task0_text_preprocessing.ipynb`
2. `notebooks/task1_tokenization.ipynb`
3. `notebooks/task2_word2vec.ipynb`
4. `notebooks/task3_static_contextual_embeddings.ipynb`
5. `notebooks/task4_bert_classification.ipynb`
6. `notebooks/task5_gpt2_generation.ipynb`
7. `notebooks/task6_semantic_search.ipynb`

## Main Findings

GPT-2 was the most token-efficient tokenizer on the IMDB sample, but the edge cases showed that tokenizer choice should be tested on real user text, especially for code and multilingual input. The NumPy Word2Vec model learned real movie-review associations such as `movie` near `film` and `actor` near `actress`, while noisy analogies showed how much pretrained models hide about corpus size and vocabulary choices. DistilBERT only slightly beat the TF-IDF baseline on this IMDB setup while taking far longer to train, so the simple baseline is the more practical default unless the product needs the extra quality. GPT-2 generation depended heavily on decoding strategy: top-k and top-p were more useful than greedy or low-temperature decoding because they balanced coherence and variety. In semantic search, FAISS embeddings helped on meaning-based queries, while TF-IDF stayed competitive when exact topical words appeared in the articles.

## Saved Results

The notebooks save outputs into:

- `results/metadata/` for JSON configs, timings, and summaries.
- `results/plots/` for PNG visualizations.
- `results/tables/` for CSV comparison tables.
- `results/examples/` for generated text, search examples, and explanations.

The frontend reads from these folders and does not rerun expensive model training.

## CPU And Subsampling Notes

This project is designed to be CPU-friendly. DistilBERT uses documented subsets: 5,000 training rows, 1,000 validation rows, and 1,000 final test rows, with `max_length=256`, `batch_size=16`, and 3 epochs. Word2Vec trains on 2,000 cleaned IMDB reviews with a 3,000-word vocabulary. Tokenization uses 1,000 IMDB reviews. Semantic search uses the full preprocessed AG News test split as the search corpus. Some notebooks may run faster on GPU when available, but the code and saved configuration keep the subset sizes suitable for CPU development.

For better DistilBERT fine-tuning performance, run Task 4 on a GPU. My Task 4 experiment was run on Google Colab, and the notebook includes a hardcoded Google Drive/project path used during that run. If you rerun `notebooks/task4_bert_classification.ipynb` locally or from a different Colab account, update that path first so it points to your own project directory.

## Known Limitations

The Word2Vec model is intentionally small and trained on movie reviews, so analogy quality is limited. DistilBERT was evaluated on a subset rather than the full IMDB dataset. Semantic-search relevance uses coarse AG News category labels, not detailed human relevance judgments. GPT-2 generation examples depend on the chosen prompt and decoding settings, so they should be treated as qualitative examples rather than a benchmark.
