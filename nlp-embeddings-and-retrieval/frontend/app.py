from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
METADATA_DIR = RESULTS_DIR / "metadata"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
EXAMPLES_DIR = RESULTS_DIR / "examples"


def load_json(relative_path: str) -> Any:
    """
    Load a JSON file from the results directory.

    Args:
        relative_path (str): Path relative to the results directory.

    Returns:
        Any: Parsed JSON content, or None when the file is missing.
    """
    path = RESULTS_DIR / relative_path
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_markdown(relative_path: str) -> str:
    """
    Load a markdown file from the results directory.

    Args:
        relative_path (str): Path relative to the results directory.

    Returns:
        str: Markdown text, or an empty string when the file is missing.
    """
    path = RESULTS_DIR / relative_path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_table(relative_path: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV table from the results directory.

    Args:
        relative_path (str): Path relative to the results directory.

    Returns:
        Optional[pd.DataFrame]: Loaded table, or None when the file is missing.
    """
    path = RESULTS_DIR / relative_path
    if not path.exists():
        return None
    return pd.read_csv(path)


def show_plot(relative_path: str, caption: str) -> None:
    """
    Show a saved plot if it exists.

    Args:
        relative_path (str): Path relative to the plots directory.
        caption (str): Caption displayed under the image.

    Returns:
        None: This function writes directly to Streamlit.
    """
    path = PLOTS_DIR / relative_path
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


def show_table(
    relative_path: str,
    title: str,
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Show a saved CSV table if it exists.

    Args:
        relative_path (str): Path relative to the tables directory.
        title (str): Display title for the table.
        max_rows (Optional[int]): Maximum number of rows to show.
            Defaults to None.

    Returns:
        Optional[pd.DataFrame]: Loaded table, or None when the file is missing.
    """
    df = load_table(f"tables/{relative_path}")
    if df is None:
        return None
    st.markdown(f"**{title}**")
    visible_df = df.head(max_rows) if max_rows else df
    st.dataframe(visible_df, width="stretch", hide_index=True)
    return df


def metric_value(value: Any, decimals: int = 3, suffix: str = "") -> str:
    """
    Format a metric value for display.

    Args:
        value (Any): Value to format.
        decimals (int): Number of decimals for numeric values. Defaults to 3.
        suffix (str): Text appended to the formatted value. Defaults to "".

    Returns:
        str: Display-ready metric value.
    """
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{decimals}f}{suffix}"
    return f"{value}{suffix}"


def answer(question: str, text: str) -> None:
    """
    Render a deliverable question and its answer.

    Args:
        question (str): Question from the deliverables.
        text (str): Result-based answer to display.

    Returns:
        None: This function writes directly to Streamlit.
    """
    st.markdown(
        f"""
        <div class="answer-block">
            <p class="question">{question}</p>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_generated_examples(examples: List[Dict[str, Any]]) -> None:
    """
    Render GPT-2 generated examples grouped by strategy.

    Args:
        examples (List[Dict[str, Any]]): Generated text records.

    Returns:
        None: This function writes directly to Streamlit.
    """
    if not examples:
        return
    tabs = st.tabs([str(item.get("strategy", "strategy")) for item in examples])
    for tab, item in zip(tabs, examples):
        with tab:
            st.markdown(f"**Settings:** `{item.get('settings', {})}`")
            st.write(item.get("generated_text", ""))


def render_search_examples(examples: List[Dict[str, Any]]) -> None:
    """
    Render one semantic-search query with semantic and keyword results.

    Args:
        examples (List[Dict[str, Any]]): Query example records.

    Returns:
        None: This function writes directly to Streamlit.
    """
    if not examples:
        return
    labels = [item.get("query", f"Query {index + 1}") for index, item in enumerate(examples)]
    selected = st.selectbox("Search example", labels, index=min(4, len(labels) - 1))
    item = examples[labels.index(selected)]
    st.markdown(f"**Intent:** {item.get('intent', 'n/a')}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Semantic top results**")
        semantic_rows = [
            {
                "rank": row.get("rank"),
                "label": row.get("label_name"),
                "relevant": row.get("is_relevant"),
                "text": row.get("text"),
            }
            for row in item.get("semantic_top_5", [])
        ]
        st.dataframe(pd.DataFrame(semantic_rows), width="stretch", hide_index=True)
    with right:
        st.markdown("**Keyword top results**")
        keyword_rows = [
            {
                "rank": row.get("rank"),
                "label": row.get("label_name"),
                "relevant": row.get("is_relevant"),
                "text": row.get("text"),
            }
            for row in item.get("keyword_top_5", [])
        ]
        st.dataframe(pd.DataFrame(keyword_rows), width="stretch", hide_index=True)


st.set_page_config(
    page_title="Week 6 LLM Fundamentals",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17202a;
        --muted: #5f6b77;
        --paper: #fbfaf7;
        --line: #e3ded5;
        --accent: #2f6f73;
        --accent-soft: #e7f0ef;
        --note: #fff6df;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
    }

    section[data-testid="stSidebar"] {
        background: #f1eee8;
        border-right: 1px solid var(--line);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    h1 {
        font-size: 3rem;
        line-height: 1.05;
        margin-bottom: 0.4rem;
    }

    h2 {
        border-top: 1px solid var(--line);
        padding-top: 2rem;
        margin-top: 2.5rem;
    }

    p, li {
        font-size: 1.03rem;
        line-height: 1.65;
    }

    .deck {
        color: var(--muted);
        font-size: 1.18rem;
        line-height: 1.7;
        max-width: 820px;
        margin-bottom: 1.5rem;
    }

    .kicker {
        color: var(--accent);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .answer-block {
        background: #ffffff;
        border-left: 5px solid var(--accent);
        border-radius: 4px;
        box-shadow: 0 1px 0 rgba(0, 0, 0, 0.04);
        margin: 1rem 0 1.25rem 0;
        padding: 1rem 1.15rem;
    }

    .answer-block p {
        margin: 0.35rem 0;
    }

    .question {
        color: var(--accent);
        font-weight: 700;
    }

    .callout {
        background: var(--note);
        border: 1px solid #eedaa8;
        border-radius: 4px;
        margin: 1.2rem 0;
        padding: 1rem 1.15rem;
    }

    .byline {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 0.75rem 0.9rem;
    }

    header[data-testid="stHeader"],
    #MainMenu,
    header [data-testid="stToolbar"],
    header [data-testid="stDeployButton"],
    header [data-testid="stDecoration"],
    header [data-testid="stStatusWidget"],
    .stDeployButton {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

task0 = load_json("metadata/task0_text_preprocessing_summary.json") or {}
task1 = load_json("metadata/task1_tokenization_summary.json") or {}
task2 = load_json("metadata/task2_word2vec_training_summary.json") or {}
task3 = load_json("metadata/task3_static_contextual_summary.json") or {}
task4 = load_json("metadata/task4_metrics.json") or {}
task5 = load_json("metadata/task5_generation_summary.json") or {}
task6 = load_json("metadata/task6_index_summary.json") or {}
generated_examples = load_json("examples/task5_generated_text_examples.json") or []
search_examples = load_json("examples/task6_query_examples.json") or []
encoder_decoder_text = load_markdown("examples/task5_encoder_decoder_explanation.md")

st.sidebar.title("Week 6 Report")
st.sidebar.caption("All content is loaded from saved files in `results/`.")
section = st.sidebar.radio(
    "Read section",
    [
        "Full Blog",
        "Data Preparation",
        "Task 1 Tokenization",
        "Task 2 Word2Vec",
        "Task 3 Embeddings",
        "Task 4 Classification",
        "Task 5 Generation",
        "Task 6 Search",
        "Findings",
        "Conclusion",
    ],
)

st.markdown('<p class="kicker">LLM fundamentals notebook results</p>', unsafe_allow_html=True)
st.title("What the Week 6 notebooks showed")
st.markdown(
    """
    <p class="deck">
    This report turns the saved notebook outputs into a blog-style walkthrough:
    tokenization economics, from-scratch Word2Vec, static versus contextual embeddings,
    DistilBERT classification, GPT-2 generation, attention patterns, semantic search,
    and final practitioner findings.
    </p>
    """,
    unsafe_allow_html=True,
)

def should_show(name: str) -> bool:
    """
    Decide whether to render a section.

    Args:
        name (str): Section name from the sidebar.

    Returns:
        bool: True when the current sidebar selection should show the section.
    """
    return section in ("Full Blog", name)


if should_show("Data Preparation"):
    st.header("Data Preparation")
    imdb_clean = next(
        (item for item in task0.get("clean_summaries", []) if item.get("dataset") == "IMDB"),
        {},
    )
    ag_clean = next(
        (item for item in task0.get("clean_summaries", []) if item.get("dataset") == "AG News"),
        {},
    )
    removed = {item.get("dataset"): item for item in task0.get("removal_summaries", [])}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean IMDB rows", metric_value(imdb_clean.get("row_count"), 0))
    c2.metric("Clean AG News rows", metric_value(ag_clean.get("row_count"), 0))
    c3.metric("IMDB rows removed", metric_value(removed.get("IMDB", {}).get("total_removed_rows"), 0))
    c4.metric("AG News rows removed", metric_value(removed.get("AG News", {}).get("total_removed_rows"), 0))

    st.write(
        "The preprocessing notebook cleaned HTML/entities/control characters, removed duplicate or conflicting rows, "
        "and saved stable IMDB and AG News datasets. This matters because later model comparisons are only meaningful "
        "if train/test leakage and duplicate examples are controlled."
    )

if should_show("Task 1 Tokenization"):
    st.header("Task 1: Text And Tokenization")
    length_summary = task1.get("length_summary", {})
    tokenizer_rows = task1.get("tokenizer_comparison", [])
    best_tokenizer = tokenizer_rows[0].get("tokenizer", "GPT-2") if tokenizer_rows else "GPT-2"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample size", metric_value(task1.get("sample_size"), 0))
    c2.metric("Avg words/review", metric_value(length_summary.get("average_word_count"), 2))
    c3.metric("Avg chars/review", metric_value(length_summary.get("average_character_count"), 2))
    c4.metric("Most economical", best_tokenizer)

    left, right = st.columns(2)
    with left:
        show_plot("task1_review_length_histogram.png", "Review length distribution")
    with right:
        show_plot("task1_tokenizer_count_histogram.png", "Token count distributions by tokenizer")

    show_table("task1_tokenizer_comparison.csv", "Tokenizer comparison")
    show_table("task1_edge_case_tokens.csv", "Edge-case tokenization", max_rows=18)

    answer(
        "Which tokenizer is the most economical, and why?",
        task1.get(
            "interpretation",
            "GPT-2 produced the fewest average tokens in this sample, while BERT and DistilBERT were nearly identical because they use the same WordPiece vocabulary family.",
        ),
    )
    answer(
        "If paying per token, which tokenizer would you prefer? Does it change for non-English text or code?",
        "For this IMDB sample, GPT-2 is the cheapest choice because it averaged fewer tokens per review. "
        "The answer can change for non-English text or code because BPE and WordPiece split rare words, punctuation, accents, and code symbols differently, so the best production choice should be tested on the real user text.",
    )

if should_show("Task 2 Word2Vec"):
    st.header("Task 2: Word2Vec From Scratch")
    st.write(
        "This section trains a small skip-gram Word2Vec model with negative sampling using NumPy only. "
        "It uses the cleaned IMDB train split, tokenizes the selected reviews into lowercase word tokens, "
        "builds a fixed vocabulary, and prepares numeric center-context pairs before training."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training reviews", metric_value(task2.get("sample_size"), 0))
    c2.metric("Vocabulary", metric_value(task2.get("vocab_size"), 0))
    c3.metric("Training pairs", metric_value(task2.get("training_pair_count"), 0))
    c4.metric("Final loss", metric_value(task2.get("final_epoch_loss"), 3))

    st.markdown("**Notebook settings**")
    st.write(
        "The main experiment used 2,000 cleaned IMDB reviews, a 3,000-word vocabulary, `min_count=2`, "
        "`window_size=2`, 100-dimensional embeddings, 5 negative samples, learning rate 0.01, 10 epochs, "
        "and seed 42. The dimension comparison retrained 50-, 100-, and 200-dimensional models for 5 epochs "
        "on the same prepared pairs."
    )
    st.markdown("**Data preparation observations**")
    st.write(
        "The selected reviews were not truncated. The average selected review had about 233 tokens, while the "
        "longest selected review had 1,285 tokens, so longer reviews contributed many more training pairs. "
        "The vocabulary kept the most frequent 3,000 words; many rare words were grouped into `<UNK>`, with "
        "61,894 token positions mapped to that token. This keeps the model small but loses rare-word detail."
    )
    answer(
        "How many training pairs were generated?",
        "The notebook generated 1,850,888 center-context pairs from the 2,000 selected IMDB reviews. "
        "Each pair is one learning example: the model sees a center word and one nearby context word. "
        "With `window_size = 2`, most words can be paired with up to two words on the left and two words on the right.",
    )

    show_plot("task2_training_loss.png", "Skip-gram negative-sampling training loss")
    answer(
        "Did training improve the model?",
        "Yes. The average loss decreased from 2.7287 in epoch 1 to 2.3041 in epoch 10. "
        "This means the model became better at recognizing real nearby words compared with sampled negative words. "
        "The loss is not a direct human-quality score, but the downward trend shows that learning happened.",
    )
    answer(
        "What embeddings did we extract after training?",
        "After training, the notebook extracted `W_in`, the input embedding matrix, with shape (3000, 100). "
        "That means every vocabulary item has one learned vector with 100 numbers, and those vectors are used "
        "for nearest-neighbor search and analogy checks.",
    )
    show_table("task2_nearest_neighbors.csv", "Nearest neighbors learned from IMDB")
    show_table("task2_analogies.csv", "Analogy attempts")

    answer(
        "Do the nearest neighbors make sense?",
        "Mostly yes, with some noise. `movie` is closest to `film`, `mess`, `picture`, `flick`, and `documentary`, "
        "which are all review-related. `actor` is closest to `actress`, `role`, and `person`, which also makes sense. "
        "`bad` is close to words like `terrible`, `lame`, and `horrible`. The noisy part is visible for `good`, "
        "where words like `crappy` and `creepy` appear; that happens because this small model learns from nearby "
        "review language, not perfect dictionary meaning.",
    )
    answer(
        "Do the analogy tasks work?",
        "Some worked in a simple movie-review way. `man - woman + actor` returned `actress` first, and "
        "`movie - film + actor` also returned `actress` first. `good - bad + great` returned positive words like "
        "`fine`, `wonderful`, and `excellent`, but it did not produce a clean textbook analogy. This is expected "
        "because the model only saw 2,000 movie reviews, not a large general-purpose corpus.",
    )
    answer(
        "How did embedding dimension affect quality?",
        "Increasing the vector size did not clearly improve the result in this run. The 100-dimensional model had "
        "the lowest final loss (2.3609), but the difference from 50 and 200 dimensions was very small. All three "
        "models kept `film` as the first neighbor for `movie`, and the rest of the neighbors stayed movie-related "
        "(`documentary`, `picture`, `mess`, `flick`, or `show`). A concrete takeaway is that more dimensions did "
        "not automatically make better embeddings for this small IMDB corpus.",
    )
    st.markdown("**Conclusion**")
    st.write(
        "This experiment shows how a computer can learn word meaning by looking at nearby words in movie reviews. "
        "The model learned useful patterns: `movie` moved close to `film`, and `actor` moved close to `actress`. "
        "It also learned some sentiment patterns, but not perfectly, because words in reviews can appear in messy "
        "or sarcastic contexts. The analogy examples show that this small model can catch a few simple relationships, "
        "but it is not broad enough to understand language like a large pretrained model."
    )

if should_show("Task 3 Embeddings"):
    st.header("Task 3: Static Vs Contextual Embeddings")
    st.write(
        "This section compares static GloVe embeddings with contextual BERT embeddings. Static embeddings give one "
        "vector per word no matter which sentence it appears in. Contextual embeddings can change when the same word "
        "appears in a different sentence."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Static bank similarity", metric_value(task3.get("bank_static_similarity"), 3))
    c2.metric("BERT bank similarity", metric_value(task3.get("bank_bert_similarity"), 3))
    c3.metric("Projection word", task3.get("projection_word", "light"))

    st.markdown("**Notebook settings**")
    st.write(
        "The notebook used `glove-wiki-gigaword-100` for static embeddings and `bert-base-uncased` for contextual "
        "embeddings. Static nearest neighbors were checked for `bank`, `apple`, and `light`, and the analogy test "
        "used `king - man + woman`."
    )
    show_table("task3_glove_neighbors.csv", "GloVe nearest neighbors")
    st.write(
        "This table shows one fixed neighborhood for each word. For example, `bank` has one list of neighbors even "
        "though it can mean a river edge or a financial company."
    )
    show_table("task3_similarity_comparison.csv", "Same word in different contexts")
    st.write(
        "The static similarity for `bank` is exactly the same word compared with itself, so it is 1.0. The BERT "
        "similarity compares two sentence-specific vectors, so it can show that river `bank` and money `bank` are "
        "not used the same way. The same idea appears for `light` and `match`, where BERT gives lower similarity "
        "when the sentence changes the meaning."
    )
    show_plot("task3_contextual_embedding_projection.png", "Contextual embeddings projected to 2D")
    answer(
        "Do the senses cluster?",
        "Mostly yes. In the PCA plot, the brightness uses of `light` mostly appear on the right side, while the "
        "weight uses mostly appear on the left side. The split is not perfect because this is only a small set of "
        "ten sentences, but the plot does show that BERT changes the word vector when `light` means brightness "
        "versus when it means low weight.",
    )

    answer(
        "When choose static embeddings over contextual ones, and vice versa?",
        "Use static embeddings when cost, speed, and a simple fixed vocabulary are enough. "
        f"In these results, static 'bank' compared with itself has cosine similarity {metric_value(task3.get('bank_static_similarity'), 3)}, "
        f"while BERT's river-bank versus money-bank contexts dropped to {metric_value(task3.get('bank_bert_similarity'), 3)}. "
        "That makes BERT better when the surrounding sentence changes the meaning, and GloVe/Word2Vec better when cheap broad similarity is sufficient.",
    )
    st.markdown("**Final interpretation**")
    st.write(
        "Static embeddings are useful when the task is simple, cost matters, and one general meaning per word is enough. "
        "Contextual embeddings are better when sentence meaning matters, such as with words like `bank`, `light`, or "
        "`match`. In this experiment, GloVe gives one fixed answer for each word, while BERT lets the same word move "
        "depending on how it is used."
    )

if should_show("Task 4 Classification"):
    st.header("Task 4: BERT Text Classification")
    st.write(
        "This section compares a simple TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT "
        "sentiment classifier on IMDB reviews. The baseline gives the transformer a practical number to beat "
        "before paying the cost of fine-tuning."
    )
    baseline = task4.get("baseline", {})
    distilbert = task4.get("distilbert", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TF-IDF accuracy", metric_value(baseline.get("accuracy"), 3))
    c2.metric("DistilBERT accuracy", metric_value(distilbert.get("test_accuracy"), 3))
    c3.metric("TF-IDF train time", metric_value(baseline.get("training_time_seconds"), 2, "s"))
    c4.metric("DistilBERT train time", metric_value(distilbert.get("training_time_seconds"), 2, "s"))

    st.markdown("**Notebook settings**")
    st.write(
        "The run used the preprocessed IMDB data with 5,000 training rows, 1,000 validation rows, and 1,000 final "
        "test rows. TF-IDF used up to 10,000 features. DistilBERT used `distilbert-base-uncased`, `max_length=256`, "
        "`batch_size=16`, learning rate `2e-5`, and 3 epochs. The notebook used CPU-friendly subset sizes and "
        "hyperparameters; validation rows came from the IMDB training split, while the IMDB test split was held out "
        "for final evaluation only."
    )
    st.markdown("**Split observation**")
    st.write(
        "The split table showed balanced labels in every split: train had 2,496 negative and 2,504 positive reviews, "
        "validation had 499 negative and 501 positive reviews, and test had 497 negative and 503 positive reviews. "
        "Keeping validation and test separate matters because model choices can be guided by validation accuracy "
        "without leaking into final test evaluation."
    )
    answer(
        "How strong was the TF-IDF baseline?",
        "The TF-IDF baseline reached 0.863 accuracy and 0.863 macro F1 on the held-out test subset, with training "
        "and evaluation taking about 1.43 seconds. This is a strong and very fast baseline, so DistilBERT needs a "
        "meaningful test-set improvement to justify its extra compute cost.",
    )
    answer(
        "What did the DistilBERT dataloaders contain?",
        "Each batch had `input_ids`, `attention_mask`, and `labels`. The token tensors had shape `(batch_size, 256)`, "
        "and labels had shape `(batch_size)`, so each review was represented with 256 token positions after padding "
        "or truncation.",
    )
    show_table("task4_model_comparison.csv", "Model comparison")
    left, right = st.columns(2)
    with left:
        show_plot("task4_training_validation_curves.png", "Training loss and validation accuracy")
    with right:
        show_plot("task4_confusion_matrix.png", "DistilBERT normalized confusion matrix")
    answer(
        "What did the training history show?",
        "DistilBERT fit the training data strongly: loss dropped from 0.374 in epoch 1 to 0.087 in epoch 3. "
        "Validation accuracy improved only slightly from 0.878 to 0.883, so later epochs reduced training loss "
        "much more than they improved validation performance.",
    )
    answer(
        "How did DistilBERT perform on the final test subset?",
        "DistilBERT reached 0.870 accuracy and 0.870 macro F1 on the final held-out test subset. This was below "
        "the final validation accuracy of 0.883, which suggests the validation subset was slightly easier or that "
        "the model did not generalize quite as well to the test rows.",
    )
    st.write(
        "The confusion matrix is normalized by the true class, so each row shows the share of that class predicted "
        "as negative or positive. The diagonal cells are correct predictions, which makes the plot useful for checking "
        "whether the model is balanced across sentiment classes."
    )
    show_table("task4_misclassified_examples.csv", "Misclassified examples", max_rows=5)
    st.write(
        "The displayed mistakes include several positive reviews predicted as negative and one negative review "
        "predicted as positive with high confidence. The positive reviews often contain criticism, caveats, or "
        "genre-specific language before the overall positive judgment, while the negative example contains "
        "positive-sounding wording such as interest and plausibility."
    )
    answer(
        "What is the model-comparison takeaway?",
        "The comparison table shows that DistilBERT scored slightly higher than TF-IDF on the test set: 0.870 "
        "accuracy versus 0.863, an improvement of about 0.7 percentage points. The tradeoff is compute cost: "
        "DistilBERT took about 372 seconds in this notebook run, while TF-IDF took about 1.43 seconds.",
    )

    answer(
        "Is BERT's advantage large enough to justify the added complexity and compute cost?",
        task4.get(
            "interpretation",
            "DistilBERT improved accuracy only slightly over TF-IDF while taking much longer, so the baseline is the better practical choice for this setup.",
        ),
    )
    answer(
        "Would the answer change with 10x more or 10x less data?",
        "With 10x less data, TF-IDF is even more attractive because it is fast and less operationally complex. "
        "With 10x more data, both should be evaluated again: DistilBERT may benefit more from scale, but the current result says not to assume the extra compute is worth it.",
    )

if should_show("Task 5 Generation"):
    st.header("Task 5: GPT-2 Text Generation")
    st.write(
        "This section explores GPT-2 generation strategies and attention heatmaps. GPT-2 is a decoder model, so it "
        "predicts the next token from the tokens before it."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", task5.get("model_name", "gpt2"))
    c2.metric("Strategies", metric_value(len(task5.get("strategies", [])), 0))
    c3.metric("Prompt tokens", "same prompt")
    c4.metric("Best observed", task5.get("best_strategy_observed", "top-k/top-p"))

    st.markdown("**Notebook settings**")
    st.write(
        "The notebook used the small `gpt2` model, the same prompt for every decoding strategy, `max_new_tokens=45`, "
        "`attention_max_tokens=32`, and seed 42. Only the decoding settings changed across generated examples."
    )
    st.markdown("**Prompt**")
    st.info(task5.get("prompt", "The most surprising thing about artificial intelligence is"))
    render_generated_examples(generated_examples)
    st.markdown("**Strategy quality observation**")
    st.markdown(
        """
        - `greedy`: coherent at first, but repeated the same rich/poor sentence pattern.
        - `temperature_0_3`: controlled, but repeated the phrase about being "a very different animal."
        - `temperature_1_0`: more surprising and creative, but drifted into a less clear classroom example.
        - `top_k`: more coherent than the high-temperature sample and less repetitive than greedy.
        - `top_p`: similar to top-k, with a natural continuation about theories and evidence.
        """
    )

    left, right = st.columns(2)
    with left:
        show_plot("task5_attention_head_1.png", "Layer 0 Head 0 attention")
    with right:
        show_plot("task5_attention_head_2.png", "Layer 6 Head 3 attention")
    st.write(
        "The attention examples were produced with `model.generate(..., output_attentions=True, return_dict_in_generate=True)`. "
        "The heatmaps use the generated text as the inspected sequence. The first heatmap is from an early layer, "
        "and the second is from a later layer, making it easier to compare different attention patterns."
    )

    answer(
        "How did decoding strategy affect output quality?",
        task5.get(
            "main_result",
            "Greedy and low-temperature decoding were repetitive, while top-k and top-p gave a better balance of coherence and variety.",
        ),
    )
    answer(
        "What does the attention pattern reveal about what the model is looking at?",
        task5.get(
            "attention_interpretation",
            "Layer 0 Head 0 focused mostly on the first token, while Layer 6 Head 3 spread attention across more previous tokens and used broader context.",
        ),
    )
    st.markdown("**Final answer**")
    st.write(
        "The best decoding strategy in this run was either `top_k` or `top_p`, because both gave more varied text "
        "than greedy decoding without drifting as much as temperature `1.0`. Greedy and low-temperature decoding "
        "were coherent but repetitive. The attention plots show that GPT-2 generates by looking backward at earlier "
        "tokens: Layer 0 Head 0 mostly anchors on the first token, while Layer 6 Head 3 uses a broader spread of "
        "previous tokens."
    )
    st.markdown("### Encoder Vs Decoder")
    if encoder_decoder_text:
        st.markdown(encoder_decoder_text)
    else:
        st.markdown(task5.get("encoder_decoder_explanation", ""))

if should_show("Task 6 Search"):
    st.header("Task 6: Semantic Search With FAISS")
    st.write(
        "This section builds a semantic search engine over AG News with sentence-transformer embeddings and FAISS. "
        "It also compares semantic retrieval against a TF-IDF keyword baseline on the same corpus and query set."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Corpus size", metric_value(task6.get("corpus_size"), 0))
    c2.metric("Embedding dim", metric_value(task6.get("embedding_dimension"), 0))
    c3.metric("Semantic mean P@5", metric_value(task6.get("semantic_mean_precision_at_5"), 3))
    c4.metric("Keyword mean P@5", metric_value(task6.get("keyword_mean_precision_at_5"), 3))

    st.markdown("**Notebook settings**")
    st.write(
        "The notebook used the preprocessed AG News test split as the document corpus, with no additional corpus "
        "subsample. It embedded articles with `sentence-transformers/all-MiniLM-L6-v2`, used `top_k=5`, built a "
        "`faiss.IndexFlatIP` index, and compared against a TF-IDF baseline with up to 50,000 features."
    )
    st.write(
        "The embedding vectors were L2-normalized before adding them to FAISS, so inner product behaves like cosine "
        "similarity. Relevance was evaluated with a coarse manual query label: each query targets one AG News category, "
        "and a retrieved article counts as relevant when its dataset label matches that intended category."
    )
    show_plot("task6_precision_comparison.png", "Semantic vs keyword precision@5")
    comparison = show_table("task6_semantic_vs_keyword.csv", "Semantic vs keyword comparison")
    st.write(
        "The comparison table reports the query, expected AG News category, semantic precision@5, keyword precision@5, "
        "and which method was better for that query. The chart visualizes the same comparison so it is easier to see "
        "where semantic search, keyword search, or a tie happened."
    )
    show_table("task6_search_results.csv", "All retrieved results", max_rows=20)
    st.write(
        "The full result table shows the query text, search method, rank, score, retrieved article label, expected "
        "label, relevance flag, and retrieved article text. The `is_relevant` column is the rough relevance assessment "
        "used for precision@5."
    )
    render_search_examples(search_examples)

    answer(
        "When does semantic search shine, and when does keyword search suffice?",
        "Semantic search shines when the query describes a concept rather than relying on exact words. In this notebook, "
        "`technology company releases new security software` was the clearest semantic win: semantic search got P@5 = 1.0, "
        "while keyword search got P@5 = 0.8 because keyword search was distracted by exact words like `company` and "
        "`releases`, returning one Business article about AIG press releases. Keyword search suffices when strong exact "
        "topic words appear in relevant articles. `new satellite rocket launch space mission` was a tie with P@5 = 1.0 "
        "for both methods, while `stock market and company earnings update` favored keyword search at 0.6 versus "
        "semantic search at 0.4.",
    )
    st.markdown("**Final answer**")
    st.write(
        "Task 6 builds an exact FAISS `IndexFlatIP` over normalized MiniLM sentence embeddings and compares it against "
        "a TF-IDF keyword baseline on the same AG News corpus and queries. In this run, both methods averaged P@5 = 0.9, "
        "so semantic search was not universally better. It was most useful for paraphrased or meaning-based queries, "
        "while keyword search remained a strong, cheaper baseline for precise lexical queries."
    )
    if comparison is not None:
        semantic_wins = comparison[comparison["better_method"] == "semantic"]
        keyword_wins = comparison[comparison["better_method"] == "keyword"]
        if not semantic_wins.empty or not keyword_wins.empty:
            st.markdown("**Concrete examples from the comparison table**")
            example_lines = []
            if not semantic_wins.empty:
                row = semantic_wins.iloc[0]
                example_lines.append(
                    f"Semantic win: `{row['query']}` reached P@5={row['semantic_precision_at_5']} "
                    f"versus keyword P@5={row['keyword_precision_at_5']}."
                )
            if not keyword_wins.empty:
                row = keyword_wins.iloc[0]
                example_lines.append(
                    f"Keyword win: `{row['query']}` reached P@5={row['keyword_precision_at_5']} "
                    f"versus semantic P@5={row['semantic_precision_at_5']}."
                )
            st.write(" ".join(example_lines))

if should_show("Findings"):
    st.header("Findings Summary")
    st.write(
        "These five findings focus on the most useful project takeaways, not a task-by-task recap."
    )

    findings = [
        {
            "title": "1 - Tokenization & Economics",
            "point": (
                "Tokenizer choice changes the bill and the context window before the model sees anything. "
                "In the IMDB sample, GPT-2 was the most token-efficient, but the edge-case examples showed why "
                "that result should not be blindly reused for code or multilingual text. A real app should test "
                "tokenizers on the text its users actually write."
            ),
        },
        {
            "title": "2 - Embeddings Insight",
            "point": (
                "Building Word2Vec from scratch made the hidden tradeoff visible: the model learned useful movie-review "
                "neighbors like `movie` near `film` and `actor` near `actress`, but rare words and analogies stayed noisy. "
                "Pretrained embeddings hide how much the corpus size, vocabulary cutoff, and training objective shape what "
                "the vectors can know."
            ),
        },
        {
            "title": "3 - Model Comparison Highlight",
            "point": (
                "The surprising result was not that DistilBERT won; it was that TF-IDF was close enough to make the win "
                "hard to justify. DistilBERT only fixed a small number of extra reviews per hundred, while training took "
                "minutes instead of seconds. For this dataset, the simple baseline is the engineering default unless the "
                "product needs the last bit of quality."
            ),
        },
        {
            "title": "4 - Semantic Search in Action",
            "point": (
                "Semantic search helped most when the query described meaning instead of repeating document words. "
                "For the security-software query, embeddings kept the results in the technology topic, while keyword search "
                "was partly pulled toward business press-release wording. But exact terms still matter: finance and space "
                "queries were easy for keyword search because the important words appeared directly in the articles."
            ),
        },
        {
            "title": "5 - Conclusion",
            "point": (
                "A practitioner should start with cheap baselines and small, honest evaluations before reaching for larger "
                "models. Next time, I would design the evaluation examples earlier: token edge cases, hard sentiment reviews, "
                "and retrieval queries should be chosen before training so the model choice is driven by the application, "
                "not by the most impressive architecture."
            ),
        },
    ]

    for finding in findings:
        st.subheader(finding["title"])
        st.write(finding["point"])

if should_show("Conclusion"):
    st.header("Conclusion")
    st.markdown(
        """
        <div class="callout">
        <p><strong>Main takeaway:</strong> the notebooks showed that simple methods are still strong baselines, and transformer methods are most valuable when context, generation, or semantic matching changes the problem.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "In Task 4, DistilBERT beat TF-IDF only slightly: 0.870 versus 0.863 accuracy, but it took about "
        "372 seconds compared with 1.43 seconds for TF-IDF. For this sentiment setup, the extra compute cost is hard "
        "to justify unless future data or requirements show a larger quality gap."
    )
    st.write(
        "In Task 5, GPT-2 generation quality depended heavily on decoding strategy, not only on the model itself. "
        "Greedy and low-temperature decoding became repetitive, while top-k and top-p were more useful because they "
        "kept enough randomness to avoid looping without becoming completely uncontrolled."
    )
    st.write(
        "Across all tasks, the unusual lesson is that the model is only one part of the system. Tokenizer choice changes cost, "
        "training data size changes whether a transformer is worth it, decoding settings change GPT-2 behavior, and retrieval "
        "quality depends on whether the query needs meaning-based matching or exact keywords."
    )
