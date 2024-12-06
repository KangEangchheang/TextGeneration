from collections import Counter, defaultdict
import random


class InterpolationModel:
    def __init__(self, max_n=4, lambdas=None):
        """
        Initialize the Interpolation Model.

        :param max_n: Maximum order of N-grams to use.
        :param lambdas: List of weights for N-gram probabilities. If None, weights are uniform.
        """
        self.max_n = max_n
        self.models = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
        self.lambdas = lambdas or [1 / max_n] * max_n  # Default uniform weights

        # Ensure the weights sum to 1
        weight_sum = sum(self.lambdas)
        self.lambdas = [l / weight_sum for l in self.lambdas]

    def build_model(self, tokenized_sentences):
        """
        Build N-gram models from tokenized sentences.

        :param tokenized_sentences: List of tokenized sentences.
        """
        for sentence in tokenized_sentences:
            for n in range(1, self.max_n + 1):
                for i in range(len(sentence) - n + 1):
                    ngram = tuple(sentence[i:i + n - 1])
                    next_word = sentence[i + n - 1]
                    self.models[n][ngram][next_word] += 1

        # Normalize counts to probabilities
        for n in range(1, self.max_n + 1):
            for context, counter in self.models[n].items():
                total = sum(counter.values())
                self.models[n][context] = Counter({word: count / total for word, count in counter.items()})

    def generate_text(self, seed=None, length=10):
        """
        Generate text using the interpolated N-gram model.

        :param seed: Seed context as a list of words. Random if None.
        :param length: Number of words to generate.
        :return: Generated text as a list of words.
        """
        if seed is None:
            # Pick a random starting context from the highest-order N-gram model
            seed = random.choice(list(self.models[self.max_n].keys()))
        else:
            seed = tuple(seed[-(self.max_n - 1):])  # Trim seed to match max_n - 1

        generated = list(seed)
        for _ in range(length):
            next_word = self._generate_next_word(tuple(generated[-(self.max_n - 1):]))
            if next_word is None:  # Fallback to a random word if no valid continuation
                next_word = random.choice(list(self.models[1][()].keys()))
            generated.append(next_word)

        return ' '.join(generated)  # Instead of returning a list of words

    def _generate_next_word(self, context):
        """
        Generate the next word based on the interpolated probabilities.

        :param context: Tuple representing the current context.
        :return: The next word as a string.
        """
        word_probs = Counter()

        # Aggregate probabilities from all N-gram levels
        for n in range(1, self.max_n + 1):
            relevant_context = context[-(n - 1):] if n > 1 else ()
            ngram_probs = self.models[n].get(relevant_context, {})
            for word, prob in ngram_probs.items():
                word_probs[word] += self.lambdas[n - 1] * prob

        if not word_probs:
            return None  # No valid words found

        # Sample the next word based on interpolated probabilities
        total = sum(word_probs.values())
        rnd = random.uniform(0, total)
        cumulative = 0
        for word, prob in word_probs.items():
            cumulative += prob
            if rnd <= cumulative:
                return word

        return None
