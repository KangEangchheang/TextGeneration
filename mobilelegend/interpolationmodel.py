from collections import Counter


class InterpolationModel:
    def __init__(self, ngrams, k=0.1):
        self.k = k
        self.counts = Counter(ngrams)

    def generate(self, context, n=4):
        context = tuple(context[-(n-1):])
        candidates = [(k, v + self.k) for k, v in self.counts.items() if k[:n-1] == context]
        if not candidates:
            return "<UNK>"
        return max(candidates, key=lambda x: x[1])[0][-1]
