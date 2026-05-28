# Transformer Model From Scratch without AI

Recapping what is in a transfomer model by outlining its strucutre and listing my thought process on how I am going to approach this.

The transformer model is split into the encoder and decoder.

**Pre-Encoder and Decoder**

The inputs and outputs would be in the form of <SOS>[text]<EOS>. Then the input/output would be tokenised and input/output embedding would take place where the each token in mapped to a vector of 512 dimensions. Then positional encoding will be applied where a vector would be added which add represention of the position of the token in the passage. For even dimensions we would use the sin function and for the odd dimensions we would use the cos function.

**Encoder**

The encoder would take in the vector of (num of tokens)*152 vector and it would undergo multi-head attention.

Multi-Head Attention consists of many attention 'blocks' happening simultaneously.
MultiHead(Q,K,V) = Concat(head1,...,headh)W^o
where headi = Attention(QWi^Q,KWi^K,VWi^V) and all the W vectors are tunable parameters
then the add and norm pocess, where the attention vectors are added to the inputs and are normalized using calculating (xi-mean)/sqrt(var+e) where e is a small number if variance is zero. The it is fed to a FFN, where the output is FFN(x) = max(0,xW1 + b1)W2 + b2 and this output is once again added to the output of the previous process and normilized using the same process again. The output of K and V would be fed to the process in the decoder for its multihead attention process.

**Decoder**

The decoder would aslso take in the inputs with the output embedding and positional encoding.
Firstly it would go through masked multihead attention where in this case the word can only 'see' and get context from the words before it so after getting the attention values from the multihead attention, the values on the upper triangular portion would be set to -inf so when calculating the softmax, its value would be zero, then we add and normalize, Then we would take its query value from the output and the key and value from the input and perform one more round of multihead attention and add&norm. Then it is fed to the FFN and add&norm. Then it we would implement a linear function which converts the vectors to a single number. The output is in the form of logits. Then all the logits are passed through the softmax function. We would then pick the token with the highest probability.

**Training**

For exmaple training the model using English and French. The english text would be fed into the model in the form of <SOS>[text]<EOS> into the input path, for the French text would be the same and it would be fed into the output path.

**Inference**

For inference after the model is being trained, the input in the form of english words would be fed into the input and the output in the form of <SOS> with no text is fed into the output path after running through the whole process, it would output the first word which has the highest probability. Since all the values of the input path is already calculated, now we would feed <SOS>{first_word} into the path. and the process repeats until <EOS> is produced.

Plan to build this helper.py which contains all the helper functions, encoder.py which contains the encoder code, decoder.py which contains the decoder code.

helper.py should contain the

- input/output embedding
- positional encoding
- multi-head attention (including masked version)
- add & norm
- FFN
- linear

After creating the rough structure we would need to design the training and inference scripts which go along with the script as well.
