from helper import tokenize, get_embedding_table, get_embeddings, get_positional_encoding, multi_head_attention, ffn, layer_norm, linear, update_embedding_table
import numpy as np

vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3} # special tokens
embedding = get_embedding_table(len(vocab)) # embedding table for the vocab
weights= {} # weights for multi head attention and ffn

def preprocess(sentences, vocab):
    """
    We will tokenize the sentences, get their embeddings and add positional encoding before feeding them into the encoder or decoder
    """
    tokenized_sentences = [tokenize(sentence, vocab) for sentence in sentences] # tokenising everyt sentnces
    embedding = update_embedding_table(embedding, len(vocab), len(vocab)) # update the embedding table with the new vocab
    embedded_sentences = [get_embeddings(tokenized_sentence, embedding) for tokenized_sentence in tokenized_sentences]
    positional_encodings = [get_positional_encoding(i) for i in range(len(embedded_sentences[0]))] # get the positional encoding for each position in the sentence
    preprocessed_sentences = [embedded_sentence + positional_encodings for embedded_sentence in embedded_sentences] # add the positional encoding to the embeddings
    return preprocessed_sentences

def build_weights(len(vocab)):
    """
    This function will build the weights and will contain differen values list for W_q, W_k, W_v
    and an matrix for W_o, W_1, W_2, b_1, b_2
    """
    num_heads = 8
    d_model = 512
    d_k = d_model // num_heads
    d_v = d_model // num_heads
    d_ff = 2048

    weights[W_q] = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
    weights[W_k] = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
    weights[W_v] = [np.random.randn(d_model, d_v) for _ in range(num_heads)]
    weights[W_o] = np.random.randn(num_heads * d_v, d_model)
    weights[W_1] = np.random.randn(d_model, d_ff)
    weights[W_2] = np.random.randn(d_ff, d_model)
    weights[b_1] = np.random.randn(d_ff)
    weights[b_2] = np.random.randn(d_model)


def encoder(preprocessed_sentences, weights):
    """
    We would perform multihead attention and feed forward network, along with the add and norm 8 times
    We would then return the key and value matrices to be used in the decoder
    """
    curr_output = preprocessed_sentences
    for i in range(8):
        attn_output = [multi_head_attention(x, x, x, masked=False, weights=weights) for x in curr_output] # multi head attention
        add_norm_output = [layer_norm(x + attn_output[i]) for i, x in enumerate(curr_output)]
        ffn_output = [ffn(x, weights) for x in add_norm_output]
        add_norm_output_2 = [layer_norm(add_norm_output[i] + ffn_output[i]) for i in range(len(add_norm_output))]
        curr_output = add_norm_output_2
    return curr_output, curr_output # return the key and value matrices

def decoder(preprocessed_sentences, encoder_output, weights):
    """
    We would first perform a masked self attention, then perform multihead attention with the encoder outputs
    """
    encoder_key, encoder_value = encoder_output
    curr_output = preprocessed_sentences
    for i in range(8):
        masked_attn_output = [multi_head_attention(x, x, x, masked = True, weights=weights) for x in curr_output]
        add_norm_output = [layer_norm(x + masked_attn_output[i]) for i, x in enumerate(curr_output)]
        attn_output = multi_head_attention(add_norm_output, encoder_key, encoder_value, masked=False, weights=weights)
        add_norm_output_2 = [layer_norm(add_norm_output[i] + attn_output[i]) for i in range(len(add_norm_output))]
        ffn_output = [ffn(x, weights) for x in add_norm_output_2]
        add_norm_output_3 = [layer_norm(add_norm_output_2[i] + ffn_output[i]) for i in range(len(add_norm_output_2))]
        curr_output = add_norm_output_3
    return curr_output

def post_decoder(decoder_output):
    """
    we would perform the linear transformation to get the output logits, then softmax to get the probabilities
    """
    


    
