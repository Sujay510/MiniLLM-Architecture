def pair_counts(vocab):
    pairs = {}
    for word,freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pair = (symbols[i],symbols[i+1])
            pairs[pair] = pairs.get(pair,0) + freq
    return pairs
def initial_vocab(text):
    vocab = {}
    words = text.split()
    for word in words:
        spaced = " ".join(list(word)) + " _"
        vocab[spaced] = vocab.get(spaced, 0) + 1
    return vocab

def merge(vocab,pair):
    new = {}
    find = " ".join(pair)
    replace = "".join(pair)
    for word,freq in vocab.items():
        new_word = word.replace(find,replace)
        new[new_word] = freq
    return new

def train_bpe(text,num_merges):
    vocab = initial_vocab(text)
    merges = []
    for i in range(num_merges):
        pairs = pair_counts(vocab)
        frequent = max(pairs,key = pairs.get)
        vocab = merge(vocab,frequent)
        merges.append(frequent)
    return merges

def encode(text,merges):
    words = text.split()
    tokens = [] 
    for word in words: 
        spaced = " ".join(str(word)) + " _"
        for pair in merges:
            find = " ".join(pair)
            replace = "".join(pair)
            spaced = spaced.replace(find,replace)
        tokens.extend(spaced.split())
    return tokens

def decode(tokens):
    text = "".join(tokens)
    text = text.replace("_"," ")
    return text.strip()

merges = train_bpe("hello world hello", num_merges=5)
print(merges)

tokens = encode("hello world hello", merges)
print(tokens)
print(decode(tokens))
