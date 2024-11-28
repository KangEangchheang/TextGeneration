import random


class BackoffModel:
    def __init__(self, ngram_model):
        self.ngram_model = ngram_model  # This will be a dictionary of 4-gram
    # Function to generate text using the backoff model (from 4-grams)
    def generate(self,ngram_model, context, num_words):
        generated_words = []
        for _ in range(num_words):
            # Ensure the context has exactly 3 words
            if len(context) < 3:
                return ["<Context must contain at least 3 words.>"]
            context = tuple(context[-3:])  # Last 3 words as the current context
            # print(f"Checking context: {context}") #this is for example to print the context

            if context in ngram_model:
                # Randomly choose a next word from the n-gram model
                next_word = random.choice(ngram_model[context])
                generated_words.append(next_word)

                # Update the context with the new word (convert tuple to list)
                context = list(context)  # Convert tuple to list
                context.append(next_word)  # Append the new word
            else:
                # No valid next word found, stop generation
                break

        return generated_words
