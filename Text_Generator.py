"""
text_generator.py
------------------
Bigram and trigram language models built from scratch (no external LM
libraries), trained on a real multi-sentence English corpus.

Pipeline:
  1. Load & tokenize corpus into sentences -> word lists, with <s>/</s>
     boundary tokens
  2. Split into train / held-out test sentences
  3. Build unigram, bigram, and trigram counts from the training set
  4. Apply Laplace (add-1) smoothing for probability estimation
  5. Generate new text by sampling from each model
  6. Evaluate model quality via perplexity on the held-out test set

Run:
    python3 text_generator.py
"""

import re
import random
import math
from collections import defaultdict, Counter

from corpus import get_corpus

random.seed(7)

START = "<s>"
END = "</s>"


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def sentence_tokenize(text: str):
    """Very small sentence splitter: split on ., !, ? followed by space/EOL."""
    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def word_tokenize(sentence: str):
    """Lowercase word tokenizer that keeps contractions like "don't" intact."""
    sentence = sentence.lower()
    words = re.findall(r"[a-z']+", sentence)
    return words


def build_sentences(text: str):
    """Returns a list of token lists, each wrapped with <s> ... </s>.
    Trigram model gets two <s> tokens at the start so P(w3 | w1, w2) is
    always well-defined for the first real word.
    """
    raw_sentences = sentence_tokenize(text)
    tokenized = []
    for s in raw_sentences:
        words = word_tokenize(s)
        if len(words) < 2:
            continue
        tokenized.append([START, START] + words + [END])
    return tokenized


# ---------------------------------------------------------------------------
# N-gram model
# ---------------------------------------------------------------------------

class NgramModel:
    """A Laplace-smoothed n-gram language model (n = 2 for bigram, 3 for trigram)."""

    def __init__(self, n: int):
        assert n in (2, 3), "This implementation supports bigram (2) or trigram (3)."
        self.n = n
        self.context_counts = defaultdict(Counter)   # context tuple -> Counter(next_word)
        self.vocab = set()

    def fit(self, tokenized_sentences):
        for sent in tokenized_sentences:
            # For bigram: context = single previous word (skip the extra leading <s>)
            # For trigram: context = previous two words
            start_offset = 1 if self.n == 2 else 0
            for i in range(start_offset, len(sent) - 1):
                context = tuple(sent[i - (self.n - 1) + 1:i + 1]) if self.n == 3 else (sent[i],)
                next_word = sent[i + 1]
                self.context_counts[context][next_word] += 1
                self.vocab.add(next_word)
        self.vocab.add(END)
        self.V = len(self.vocab)

    def prob(self, context, word):
        """Laplace (add-1) smoothed probability of `word` given `context`."""
        counts = self.context_counts.get(context, Counter())
        total = sum(counts.values())
        return (counts[word] + 1) / (total + self.V)

    def _get_context(self, sent, i):
        if self.n == 2:
            return (sent[i],)
        else:
            return tuple(sent[i - 1:i + 1])

    def generate(self, max_len: int = 25) -> str:
        if self.n == 2:
            context = (START,)
        else:
            context = (START, START)

        output = []
        for _ in range(max_len):
            candidates = self.context_counts.get(context)
            if not candidates:
                # backoff: pick a random known context to avoid dead ends
                context = random.choice(list(self.context_counts.keys()))
                candidates = self.context_counts[context]

            words, counts = zip(*candidates.items())
            weights = [c + 1 for c in counts]  # smoothed sampling weights
            next_word = random.choices(words, weights=weights, k=1)[0]

            if next_word == END:
                break
            output.append(next_word)

            if self.n == 2:
                context = (next_word,)
            else:
                context = (context[-1], next_word)

        return " ".join(output)

    def sentence_log_prob(self, sent):
        """Sum of log2 probabilities for a tokenized (with <s>/</s>) sentence."""
        log_prob = 0.0
        start_offset = 1 if self.n == 2 else 0
        n_predictions = 0
        for i in range(start_offset, len(sent) - 1):
            context = self._get_context(sent, i)
            word = sent[i + 1]
            p = self.prob(context, word)
            log_prob += math.log2(p)
            n_predictions += 1
        return log_prob, n_predictions

    def perplexity(self, tokenized_sentences):
        total_log_prob = 0.0
        total_words = 0
        for sent in tokenized_sentences:
            lp, n = self.sentence_log_prob(sent)
            total_log_prob += lp
            total_words += n
        avg_log_prob = total_log_prob / total_words
        return 2 ** (-avg_log_prob)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 1: Loading & tokenizing corpus")
    print("=" * 70)
    text = get_corpus()
    sentences = build_sentences(text)
    total_words = sum(len(s) - 3 for s in sentences)  # exclude <s><s> and </s>
    print(f"Total sentences: {len(sentences)}")
    print(f"Total words (excluding boundary tokens): {total_words}")

    random.shuffle(sentences)
    split = int(len(sentences) * 0.85)
    train_sents, test_sents = sentences[:split], sentences[split:]
    print(f"Train sentences: {len(train_sents)}, Test sentences: {len(test_sents)}")

    print("\n" + "=" * 70)
    print("STEP 2: Building bigram and trigram models (Laplace smoothing)")
    print("=" * 70)
    bigram = NgramModel(n=2)
    bigram.fit(train_sents)
    print(f"Bigram model: {len(bigram.context_counts)} unique contexts, vocab size {bigram.V}")

    trigram = NgramModel(n=3)
    trigram.fit(train_sents)
    print(f"Trigram model: {len(trigram.context_counts)} unique contexts, vocab size {trigram.V}")

    print("\n" + "=" * 70)
    print("STEP 3: Text generation")
    print("=" * 70)
    print("\n--- Bigram-generated sentences ---")
    for _ in range(5):
        print(" *", bigram.generate())

    print("\n--- Trigram-generated sentences ---")
    for _ in range(5):
        print(" *", trigram.generate())

    print("\n" + "=" * 70)
    print("STEP 4: Perplexity on held-out test set")
    print("=" * 70)
    bigram_ppl = bigram.perplexity(test_sents)
    trigram_ppl = trigram.perplexity(test_sents)
    print(f"Bigram perplexity:  {bigram_ppl:.2f}")
    print(f"Trigram perplexity: {trigram_ppl:.2f}")

    better = "Trigram" if trigram_ppl < bigram_ppl else "Bigram"
    print(f"""
Interpretation:
  Lower perplexity = the model is, on average, less "surprised" by the
  held-out text = better fit. Here the {better.lower()} model scored lower.

  With a corpus this small (~1,400 training words), both models are
  fighting the same core problem: sparsity. Laplace (add-1) smoothing
  reassigns a lot of probability mass to unseen n-grams, and since a
  trigram model has roughly V times more possible contexts than a bigram
  model (V = vocabulary size), its counts are spread thinner. Whether the
  bigram or trigram wins on a small held-out set can go either way run to
  run: trigrams predict better when the test context was actually seen
  in training (sharper, more confident distributions), but degrade harder
  than bigrams when it wasn't (falling back to near-uniform smoothing over
  a larger context space). At real-world scale (millions of words), this
  trade-off usually favors trigrams+ once enough contexts have been
  observed to make the added specificity pay off.
""")

    return {"bigram_perplexity": bigram_ppl, "trigram_perplexity": trigram_ppl}


if __name__ == "__main__":
    main()