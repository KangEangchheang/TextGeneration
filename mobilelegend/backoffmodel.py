from collections import Counter


class BackoffModel:
    def __init__(self, ngrams):
        self.counts = Counter(ngrams)

    def generate(self, context, n=4):
        context = tuple(context[-(n-1):])
        candidates = [(k, v) for k, v in self.counts.items() if k[:n-1] == context]
        if not candidates:
            return "<UNK>"
        return max(candidates, key=lambda x: x[1])[0][-1]
