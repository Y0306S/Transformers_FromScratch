"""
Helper functions for the project.
"""
import re # for regex operations
import numpy as np # representing matrix and thier operations
np.random.seed(42) # for reproducibility
from collections import Counter, defaultdict # for counting the frequency of tokens and storing the vocab corpus

def old_tokenize(text, vocab):
    """
    Use counter to split the text into tokens in this case into different words and mapping them into the different ids 
    Also adding the special tokens like "<PAD>, <SOS>, <EOS>, <UNK>" to the corpus and assign it a unique id as well
    This function will return a list of tokens ids for each word in the text
    Just to add on we ensured that the text is homogenous by converting all the words to lowercase before processing them
    However, it may result in the lost of some infromation for example 'Apple' can mean the brand while 'apple may only refer to the fruit.
    """
    res = []
    for word in text.split():
        lowered_word = word.lower() # ensuring that that the text is homogenous 
        if lowered_word not in vocab:
            vocab[lowered_word] = len(vocab) # assigning a unique id to each word
        res.append(vocab[lowered_word])
    return res    

def get_stats(vocab_dict):
    """
    in build_tokenizer the words are spilt into their tokens by placing a space between them eg {"he l l o </w>" : 11}
    we would use this to count the freq of the pairs of tokens
    """
    pairs = defaultdict(int)
    for word, freq in vocab_dict.items():
        # we split the words by space to extract tokens
        tokens = word.split()
        for i in range(len(tokens)-1):
            pairs[tokens[i], tokens[i+1]] += freq
    return pairs

def merge_vocab(best_pair, vocab_dict):
    """
    removes whitespace and merge the best pair of tokens across the entire dictionary
    """
    new_vocab = {}
    bigram = re.escape(" ".join(best_pair)) # add .escape ensures that special char are treated as strings
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') # regex to find the bigram in the vocab dict
    replace_string = "".join(best_pair)
    for word in vocab_dict:
        w_out = p.sub(replace_string, word)
        new_vocab[w_out] = vocab_dict[word]
    return new_vocab

def byte_pair_tokenizer(text):
    """
    Now we would want to apply the byte pair encoding which was mentioned in the paper
    We would to be safe just add in all the lowercase, upppercase letters, numbers, basic punctuation into the vocab corpus.
    We must then somehow need to find out the common pairs of tokens in the text which sit next to each other and add them into the vocab corpus
    We would iterativey do this process until we have hit the  deisred vocab size.
    One of the small improvemnts we must make is to add an end of word token to prevent the model from learning the wrong tokens 
    """
    raw_word_counts = Counter(text.split()) 
    vocab_dict = {}
    for word, freq in raw_word_counts.items():
        spaced_word = " ".join (list(word)) + "</w>" # add space to each token and special token at the end
        vocab_dict[spaced_word] = freq
    vocab = set(["<PAD>", "<SOS>", "<EOS>", "<UNK>"]) # initialize our corpus
    for word in vocab_dict:
        vocab.update(word.split()) # add the tokens into the corpus
    target_vocab_size = 32000
    merges_needed = target_vocab_size - len(vocab)
    merges_done = []
    for i in range(merges_needed):
        pairs = get_stats(vocab_dict)
        best_pair = max(pairs, key=pairs.get())
        vocab_dict = merge_vocab(best_pair, vocab_dict)
        merges_done.append(best_pair)
    return vocab, merges_done
    

def get_embedding_table(len_vocab):
    d_model = 512 # dimension of embeddings
    embedding = np.random.randn(len_vocab, d_model)
    return embedding

def update_embedding_table(embedding, len_embed, len_vocab):
    """
    This function will update the embedding table with the new vocab
    """
    for i in range(len_embed, len_vocab):
        new_embedding = np.random.randn(1, 512) # generate a random embedding for the new word
        embedding = np.vstack((embedding, new_embedding)) # add the new embedding to the table
    return embedding

def get_embeddings(tokens, embedding):
    """
    This function will return the embeddings for the tokens.
    For the first run we would first use random embeddings for the tokens
    In the future we can use program it such that semantically, similar words have similar embeddings
    """
    return embedding[tokens]

def get_positional_encoding(pos):
    """
    Get the positional encoding for the given position
    """
    res = []
    for i in range(512):
        if i%2 == 0:
            res.append(np.sin(pos / 10000 ** (2 * i / 512)))
        else:
            res.append(np.cos(pos / 10000 ** (2 * i / 512)))
    return res

def attention(query, key, value, masked=False, d_k=64):
    """
    Compute the attention scores with the option to mask the fuutre tokens
    """
    scores = np.dot(query, key.T) / np.sqrt(d_k) # compute the attention score
    if masked:
        for i in range(len(scores)):
            for j in range(i+1, len(scores)):
                scores[i][j] = float('-inf') # mask the future tokens
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) # softmax 
    res = np.dot(attention_weights, value)
    return res
    

def multi_head_attention(query, key, value, masked=False, weights=None):
    """
    This function will return the multi head attention, with the option to mask the future tokens in the decoder during training.
    We will use random weight for the W matrices for now. 
    The W matrices help to shrink down the dimension of the Q, K, V matrices to reduce computational overhead
    Stored all of the weight matrices in a dictionary for easier retrival
    """
    num_heads = 8 # number of attention heads
    d_k = 512 // num_heads # dimension of values 
    d_h = 512 // num_heads # dimension of queries and keys
    attention_heads = []
    for i in range(num_heads):
        W_q_i = weights[W_q][i] # weight matrix for query at head i
        W_k_i = weights[W_k][i] # weight matrix for key at head i
        W_v_i = weights[W_v][i] 
        query_i = np.dot(query, W_q_i) # query for head i
        key_i = np.dot(key, W_k_i) # key for head i
        value_i = np.dot(value, W_v_i) # value for head i
        attention_i = attention(query_i, key_i, value_i, masked, d_k) # attention for head i
        attention_heads.append(attention_i)
    attention_f = np.concatenate(attention_heads, axis=1) # concatenate attention from the heads
    W_o = weights[W_o] # weight matrix
    res = np.dot(attention_f, W_o) 
    return res

def layer_norm(x):
    epsilon = 1e-6
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    normalized = (x-mean) / np.sqrt(variance + epsilon)
    return normalized

def ffn(x, weights):
    W_1 = weights[W_1]
    b_1 = weights[b_1]
    W_2 = weights[W_2]
    b_2 = weights[b_2]
    return np.dot(np.maximum(0, np.dot(x, W_1) + b_1), W_2) + b_2

def linear(x, embedding):
    """
    we cannot just search the values of the embeddings table and return it as the vector output will not match the value in the embedding table exactly,
    so we will need to use the dot proudct to see how close it is to each word
    """
    return np.dot(x, embedding.T)



