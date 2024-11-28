from collections import Counter


class InterpolationModel:
    def __init__(self, ngrams, k=0.1):
        self.ngrams = ngrams
        self.k = k  # Add-k smoothing parameter
    
    def get_ngram_prob(self, ngram, n):
        """ Calculate the probability of a n-gram (with add-k smoothing) """
        count = self.ngrams[n].count(ngram)
        vocab_size = len(set([ngram[-1] for ngram in self.ngrams[n]]))  # Total vocabulary size
        total_count = sum([c[1] for c in Counter(self.ngrams[n]).items()])  # Total counts of all n-grams
        return (count + self.k) / (total_count + self.k * vocab_size)
    
    def generate(self, context):
        context_tuple = tuple(context)
        
        # Interpolation (combining the probabilities of different n-grams)
        prob_4gram = self.get_ngram_prob(context_tuple + ('',), 4)
        prob_3gram = self.get_ngram_prob(context_tuple[1:] + ('',), 3)
        prob_2gram = self.get_ngram_prob(context_tuple[2:] + ('',), 2)
        prob_1gram = self.get_ngram_prob(context_tuple[3:] + ('',), 1)
        
        return prob_4gram + prob_3gram + prob_2gram + prob_1gram
