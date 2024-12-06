from collections import defaultdict, Counter
import random
from nltk.util import ngrams


class BackoffNGramModel:
    def __init__(self, max_n=4):
        """
        Initialize the Backoff N-Gram Model.
        :param max_n: Maximum order of N-grams to consider.
        """
        self.max_n = max_n
        self.models = {n: defaultdict(Counter) for n in range(1, max_n + 1)}

    def build_model(self, tokenized_data):
        """
        Build N-gram models for the given tokenized data.
        :param tokenized_data: List of tokenized sentences (list of lists).
        """
        for sentence in tokenized_data:
            for n in range(1, self.max_n + 1):
                n_grams = ngrams(sentence, n)
                for gram in n_grams:
                    prefix = tuple(gram[:-1])  # Prefix (n-1 words)
                    suffix = gram[-1]         # Suffix (last word)
                    self.models[n][prefix][suffix] += 1

        # Convert counts to probabilities
        for n in self.models:
            for prefix, suffix_counts in self.models[n].items():
                total = sum(suffix_counts.values())
                for suffix in suffix_counts:
                    suffix_counts[suffix] /= total

    def generate_text(self, seed, length):
        # Choose a random seed from the highest-order model if no seed is provided
        if seed is None:
            seed = random.choice(list(self.models[self.max_n].keys()))
        text = list(seed)

        for _ in range(length - len(seed)):
            for n in range(self.max_n, 0, -1):  # Back off through N-grams
                prefix = tuple(text[-(n-1):]) if n > 1 else ()  # Prefix for current N-gram
                if prefix in self.models[n]:
                    # Sample the next word based on probabilities
                    next_word = random.choices(
                        list(self.models[n][prefix].keys()),
                        weights=self.models[n][prefix].values()
                    )[0]
                    text.append(next_word)
                    break
            else:
                # If no match is found, choose a random word (default to unigram)
                next_word = random.choice(list(self.models[1][()].keys()))
                text.append(next_word)

        return ' '.join(text)