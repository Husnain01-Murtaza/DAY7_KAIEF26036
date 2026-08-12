"""
sentiment_classifier.py
------------------------
IMDB-style movie review sentiment classifier.

Pipeline:
  1. Load data (synthetic realistic corpus - see data_generation.py)
  2. Preprocess text (lowercase, strip HTML/punctuation, remove stopwords)
  3. Vectorize with TF-IDF (unigrams + bigrams)
  4. Train a Logistic Regression classifier
  5. Evaluate with accuracy, precision, recall, F1, confusion matrix
  6. Extract & visualize the most predictive words for each class

Run:
    python3 sentiment_classifier.py
"""

import re
import string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from data_generation import generate_dataset

# ---------------------------------------------------------------------------
# 1. Stopwords (standard English list, hardcoded since NLTK data can't be
#    downloaded in this environment)
# ---------------------------------------------------------------------------

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

# Negation words carry sentiment information, so we deliberately keep them
# in the vocabulary rather than stripping them out.
NEGATIONS_TO_KEEP = {"no", "not", "nor", "don't", "isn't", "wasn't",
                      "aren't", "wasn't", "won't", "can't", "couldn't"}
STOPWORDS = STOPWORDS - NEGATIONS_TO_KEEP


def preprocess(text: str) -> str:
    """Lowercase, strip HTML tags/punctuation/digits, remove stopwords."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)                       # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)             # URLs
    text = text.translate(str.maketrans("", "", string.punctuation.replace("'", "")))
    text = re.sub(r"\d+", " ", text)                          # digits
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def main():
    print("=" * 70)
    print("STEP 1: Loading data")
    print("=" * 70)
    texts, labels = generate_dataset(n_per_class=1500)
    print(f"Total reviews: {len(texts)}  (pos: {sum(labels)}, neg: {len(labels) - sum(labels)})")
    print("\nExample raw review:")
    print(f"  [{labels[0]}] {texts[0]}")

    print("\n" + "=" * 70)
    print("STEP 2: Preprocessing")
    print("=" * 70)
    processed = [preprocess(t) for t in texts]
    print("Example after preprocessing:")
    print(f"  {processed[0]}")

    print("\n" + "=" * 70)
    print("STEP 3: Train/test split + TF-IDF vectorization")
    print("=" * 70)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train size: {len(X_train_text)}, Test size: {len(X_test_text)}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"TF-IDF matrix shape (train): {X_train.shape}")

    print("\n" + "=" * 70)
    print("STEP 4: Training Logistic Regression")
    print("=" * 70)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    print("Model trained.")

    print("\n" + "=" * 70)
    print("STEP 5: Evaluation")
    print("=" * 70)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print("               pred_neg  pred_pos")
    print(f"actual_neg     {cm[0][0]:<9d} {cm[0][1]}")
    print(f"actual_pos     {cm[1][0]:<9d} {cm[1][1]}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["negative", "positive"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["negative", "positive"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (F1 = {f1:.3f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("/home/claude/nlp_systems/outputs/confusion_matrix.png", dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("STEP 6: Most predictive words")
    print("=" * 70)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]

    top_n = 20
    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_n]

    print(f"\nTop {top_n} words/phrases predictive of POSITIVE sentiment:")
    for idx in top_pos_idx:
        print(f"  {feature_names[idx]:<30s} coef = {coefs[idx]:+.3f}")

    print(f"\nTop {top_n} words/phrases predictive of NEGATIVE sentiment:")
    for idx in top_neg_idx:
        print(f"  {feature_names[idx]:<30s} coef = {coefs[idx]:+.3f}")

    # --- Visualization: horizontal bar chart of top predictive words ---
    fig, ax = plt.subplots(figsize=(9, 9))

    pos_words = feature_names[top_pos_idx][::-1]
    pos_vals = coefs[top_pos_idx][::-1]
    neg_words = feature_names[top_neg_idx][::-1]
    neg_vals = coefs[top_neg_idx][::-1]

    all_words = list(neg_words) + list(pos_words)
    all_vals = list(neg_vals) + list(pos_vals)
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in all_vals]

    y_pos = np.arange(len(all_words))
    ax.barh(y_pos, all_vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_words, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient (TF-IDF weight)")
    ax.set_title(f"Top {top_n} Predictive Words — Positive (green) vs Negative (red) Sentiment")
    plt.tight_layout()
    plt.savefig("/home/claude/nlp_systems/outputs/top_predictive_words.png", dpi=150)
    plt.close()

    print("\nSaved plots:")
    print("  outputs/confusion_matrix.png")
    print("  outputs/top_predictive_words.png")

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
    }


if __name__ == "__main__":
    main()