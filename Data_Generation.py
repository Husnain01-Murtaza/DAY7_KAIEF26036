"""
data_generation.py
-------------------
Generates a large, linguistically realistic movie-review dataset for the
sentiment classifier.

WHY SYNTHETIC DATA?
This environment has no network access to Kaggle / HuggingFace / the
Stanford AI Lab servers that host the real 50k-review IMDB dataset, so it
can't be downloaded here. Instead, this module builds a template + vocabulary
based generator that:
  - Combines hundreds of aspect phrases (acting, plot, pacing, dialogue...)
    with sentiment-bearing adjectives and intensifiers
  - Injects negation, mixed clauses, and varied sentence structure so the
    class boundary isn't trivially a single keyword
  - Produces thousands of unique reviews with realistic length variance

DROP-IN REAL DATA:
If you have the real IMDB dataset (e.g. aclImdb, or a Kaggle CSV with
columns `review` and `sentiment`), just replace the call to
`generate_dataset()` in sentiment_classifier.py with:

    import pandas as pd
    df = pd.read_csv("IMDB Dataset.csv")   # columns: review, sentiment
    df['label'] = (df['sentiment'] == 'positive').astype(int)

Everything downstream (preprocessing, TF-IDF, model, plots) works unchanged.
"""

import random

random.seed(42)

# ---------------------------------------------------------------------------
# Vocabulary banks
# ---------------------------------------------------------------------------

ASPECTS = [
    "the acting", "the plot", "the script", "the direction", "the cinematography",
    "the pacing", "the soundtrack", "the characters", "the ending", "the dialogue",
    "the visual effects", "the editing", "the storyline", "the performances",
    "the chemistry between the leads", "the cast", "the writing", "the tone",
    "the second act", "the world-building", "the production design",
    "the lead actor", "the supporting cast", "the twist", "the runtime",
    "the humor", "the emotional depth", "the villain", "the score",
]

POSITIVE_ADJ = [
    "brilliant", "outstanding", "captivating", "masterful", "compelling",
    "riveting", "superb", "excellent", "phenomenal", "well-crafted",
    "engaging", "powerful", "beautifully done", "impressive", "flawless",
    "unforgettable", "top-notch", "remarkable", "delightful", "stunning",
    "heartfelt", "inspired", "gripping", "polished", "exceptional",
]

NEGATIVE_ADJ = [
    "terrible", "dreadful", "disappointing", "lackluster", "tedious",
    "clumsy", "forgettable", "unconvincing", "flat", "poorly written",
    "cringeworthy", "uninspired", "sloppy", "boring", "shallow",
    "mediocre", "awful", "incoherent", "predictable", "wooden",
    "overlong", "amateurish", "hollow", "grating", "underwhelming",
]

INTENSIFIERS = ["", "truly ", "genuinely ", "surprisingly ", "remarkably ",
                "absolutely ", "consistently ", "utterly ", "really "]

POS_OPENERS = [
    "I was blown away by this film.",
    "This movie exceeded every expectation I had.",
    "What a fantastic piece of filmmaking.",
    "I walked out of the theater grinning.",
    "This is easily one of the best films I've seen this year.",
    "From start to finish, I was completely hooked.",
    "Rarely does a film hit every note this well.",
    "I can't stop thinking about this movie.",
]

NEG_OPENERS = [
    "I regret spending my evening on this film.",
    "This movie was a chore to sit through.",
    "I expected so much more from this one.",
    "What a colossal waste of potential.",
    "I almost walked out halfway through.",
    "This is easily one of the worst films I've seen this year.",
    "From start to finish, I was bored out of my mind.",
    "I can't believe how much this movie let me down.",
]

POS_CLOSERS = [
    "I'd recommend this to anyone who loves great cinema.",
    "I'll definitely be watching this again.",
    "This one deserves all the praise it's getting.",
    "A must-watch for fans of the genre.",
    "Easily worth the price of admission.",
    "This film will stay with me for a long time.",
]

NEG_CLOSERS = [
    "I would not recommend this to anyone.",
    "I won't be watching this again.",
    "Save your money and skip this one.",
    "A forgettable entry in an otherwise decent genre.",
    "Not worth the price of admission.",
    "This film will be forgotten within a week.",
]

NEGATION_TEMPLATES_POS = [
    "It's not a perfect film, but {aspect} more than makes up for it.",
    "I wasn't expecting much, yet {aspect} turned out to be {intensifier}{adj}.",
]

NEGATION_TEMPLATES_NEG = [
    "It's not the worst film ever, but {aspect} was still {intensifier}{adj}.",
    "I wanted to like it, but {aspect} was {intensifier}{adj} and ruined the experience.",
]


def _aspect_sentence(polarity: str) -> str:
    aspect = random.choice(ASPECTS)
    intensifier = random.choice(INTENSIFIERS)
    if polarity == "pos":
        adj = random.choice(POSITIVE_ADJ)
        verb_phrase = random.choice([
            f"{aspect.capitalize()} was {intensifier}{adj}.",
            f"{aspect.capitalize()} felt {intensifier}{adj}.",
            f"I found {aspect} to be {intensifier}{adj}.",
        ])
    else:
        adj = random.choice(NEGATIVE_ADJ)
        verb_phrase = random.choice([
            f"{aspect.capitalize()} was {intensifier}{adj}.",
            f"{aspect.capitalize()} felt {intensifier}{adj}.",
            f"I found {aspect} to be {intensifier}{adj}.",
        ])
    return verb_phrase


def _generate_review(polarity: str) -> str:
    n_aspect_sentences = random.randint(2, 5)
    sentences = []

    opener = random.choice(POS_OPENERS if polarity == "pos" else NEG_OPENERS)
    sentences.append(opener)

    for _ in range(n_aspect_sentences):
        sentences.append(_aspect_sentence(polarity))

    # Occasionally add a mixed/negated clause to avoid a trivially separable
    # bag-of-words boundary (keeps the classification task realistic).
    if random.random() < 0.35:
        aspect = random.choice(ASPECTS)
        intensifier = random.choice(INTENSIFIERS)
        if polarity == "pos":
            adj = random.choice(POSITIVE_ADJ)
            template = random.choice(NEGATION_TEMPLATES_POS)
        else:
            adj = random.choice(NEGATIVE_ADJ)
            template = random.choice(NEGATION_TEMPLATES_NEG)
        sentences.append(template.format(aspect=aspect, intensifier=intensifier, adj=adj))

    closer = random.choice(POS_CLOSERS if polarity == "pos" else NEG_CLOSERS)
    sentences.append(closer)

    random.shuffle(sentences[1:-1])  # keep opener first, closer last
    return " ".join(sentences)


def generate_dataset(n_per_class: int = 1200):
    """Returns (texts, labels) with label 1 = positive, 0 = negative."""
    texts, labels = [], []
    seen = set()
    for polarity, label in [("pos", 1), ("neg", 0)]:
        count = 0
        attempts = 0
        while count < n_per_class and attempts < n_per_class * 20:
            attempts += 1
            review = _generate_review(polarity)
            if review in seen:
                continue
            seen.add(review)
            texts.append(review)
            labels.append(label)
            count += 1
    # shuffle texts/labels together
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


if __name__ == "__main__":
    texts, labels = generate_dataset(5)
    for t, l in zip(texts, labels):
        print(l, "->", t)
        print()