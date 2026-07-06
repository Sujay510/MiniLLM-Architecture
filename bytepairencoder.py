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
    for word,freq in vocab.items():
        symbols = word.split()
        new_symbols = []
        i = 0
        while i<len(symbols):
            if(i+1<len(symbols)):    
                if(symbols[i] == pair[0] and symbols[i+1]==pair[1]):
                    s = symbols[i]+symbols[i+1]
                    new_symbols.append(s)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            else: 
                new_symbols.append(symbols[i])
                i += 1
        new_word = " ".join(new_symbols)
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
        vocab = {}
        for pair in merges:
            vocab[spaced] = 1
            vocab = merge(vocab,pair)
            new_space = list(vocab.keys())
            spaced = "".join(new_space)
        tokens.extend(spaced.split())
    return tokens

def decode(tokens):
    text = "".join(tokens)
    text = text.replace("_"," ")
    return text.strip()
if __name__ == "__main__":
    merges = train_bpe("hello world hello", num_merges=5)
    print(merges)

    tokens = encode("hello world hello", merges)
    print(tokens)
    print(decode(tokens))
