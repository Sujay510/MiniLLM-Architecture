import torch

class Simple_Tokenizer:
    def __init__(self,text):
        words = text.split()
        vocab = sorted(list(set(words)))
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}

    def encode(self,text):
        words = text.split()
        return [self.word_to_id[word] for word in words]
    def decode(self,ids):
        return [self.id_to_word[id] for id in ids]
    
text = "the cat sat on the mat"
tokenizer = Simple_Tokenizer(text)

encoded = tokenizer.encode("the cat sat")
print(encoded)        # list of ids

decoded = tokenizer.decode(encoded)
print(decoded)        # ["the", "cat", "sat"]