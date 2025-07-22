# Importing the libraries and modules
import torch
import torch.nn as nn
import torch.optim as optim
import math


# The fundamental building blocks of tranformer model includes:
# 1. Multi-Head Attention
# 2. Position-wise Feed-Forward Networks
# 3. Positional Encoding

# Following are the steps to implement the transformer model from scratch:
# 1. Importing the necessary libraries
# 2.  Defining the basic building blocks - Multi-head Attention, Position-Wise Feed-Forward Networks, Positional Encoding
# 3. Builing the Encoder block
# 4. Building the Decoder block
# 5. Building the Transformer model- combining the Encoder and Decoder blocks

# Multi-Head Attention
# The Multi-Head Attention mechanism allows the model to jointly attend to information from different representation subspaces at different positions.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Initializing the parameters
        self.d_model = d_model
        self.num_heads = num_heads

        # Dimension of the model must be divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Dimentions of each head's querym key and value vectors
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model) # Query weights
        self.wk = nn.Linear(d_model, d_model) # Key weights
        self.wv = nn.Linear(d_model, d_model) # Value weights
        
        self.dense = nn.Linear(d_model, d_model) # Output linear layer

        # Calculate the attention scores
        def scaled_dot_product_attention(Q, K, V, mask = None):
            # Attention Scores: represents the relevance of one elt in a sequence to another elt
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.depth)
            
            # Masking otu irrelvant scores
            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask==0, -1e9)

            # Apply softmax to get the attention probabilities
            attention_probs = torch.softmax(attention_scores, dim=-1)
            
            # Multiplying the prob values to obtain the output
            output = torch.matmul(attention_probs)
        
        # Split the input tensor into multiple heads
        def split_heads(self, x):
            batch_size, seq_length, d_model = x.size()
            return x.view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)
        
        # Combine the heads back into a single tensor
        def combine_heads(self, x):
            batch_size, num_heads, seq_length, depth = x.size()
            return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
        # Linear transformation for the output
        def forward(self, Q, K, V, mask = None):
            Q = self.split_heads(self.wq(Q))
            K = self.split_heads(self.wk(K))
            V = self.split_heads(self.wv(V))
        
            # Perform scaled dot-product attention
            attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
            # Combine the heads and apply the final linear transformation
            output = self.W_o(self.combine_heads(attention_output))
            return output
    

# Position-wise Feed-Forward Networks
# The Position-wise Feed-Forward Networks consist of two linear transformations with a ReLU activation in between.

# ReLU (Rectified Linear Unit): The function introduces non-linearity into the model, allowing it to learn complex patterns in the data, and the network capable of modeling non-linear relationships.
# ReLU applied between two linear layers within the FFN block of each encoder and decoder laye

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        # Initializing the parameters
        self.linear1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.linear2 = nn.Linear(d_ff, d_model)  # Second linear layer
        self.relu = nn.ReLU()  # ReLU activation

    def forward(self, x):
       return self.linear2(self.relu(self.linear1(x)))  # Apply the two linear transformations with ReLU activation in between.
    

# Positional Encoding
# Positional Encoding is used to inject information about the relative or absolute position of tokens in the embedding sequence.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        # Initialize positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)  
        position = torch.arrange(0, max_seq_length, dtype=torch.float).unsqueeze(1)  # Create a position tensor
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # Calculate the division term

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine function to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine function to odd indices

        self.register_buffer('pe', pe.unsqueeze(0))  # Register the positional encoding as a buffer

        def forward(self, x):
            # Add positional encoding to the input tensor
            return x + self.pe[:, :x.size(1)]
        
 
# Encoder Layer:
# One of the fundamental unit of the transformer model's encoding process.
# The key components of the Encoder Layer include:
# 1. Multi-Head Self-Attention: Allows the model to focus on different parts of the input sequence simultaneously.
# 2. Position-wise Feed-Forward Network: Applies a feed-forward network to each position independently.
# 3. Layer Normalization: Normalizes the output of the attention and feed-forward layers to stabilize training.
# 4. Residual Connections: Adds the input of each sub-layer to its output, helping to mitigate the vanishing gradient problem.
# 5. Dropout: Regularization technique to prevent overfitting.

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # Initializing the parameters
        self.self_attention = MultiHeadAttention(d_model, num_heads)  # Multi-Head Self-Attention
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)  # Position-wise Feed-Forward Network
        self.norm1 = nn.LayerNorm(d_model)  # Layer Normalization after attention
        self.norm2 = nn.LayerNorm(d_model)  # Layer Normalization after feed-forward network
        self.dropout = nn.Dropout(dropout)  # Dropout after feed-forward network

    def forward(self, x, mask):
        # Apply multi-head self-attention and add residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Apply position-wise feed-forward network and add residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Decoder Layer:
# The decoder layer is a curcial building blaock of the transformer model. It plays a dual role:
# 1. Decoding Output Sequences: It generates the output sequence based on the encoded input sequence.
# 2. Attending to Relevant Information: Paying attention to both the previously generated output tokens and the encoded input sequence.

# Key components of the Decoder Layer include:
# 1. Self-Attention: Allows the decoder to attend to its own previous outputs.
# 2. Encoder-Decoder Attention: Enables the decoder to focus on relevant parts of the encoded input sequence.
# 3. Position-wise Feed-Forward Network: Applies a feed-forward network to each position.
# 4. Layer Normalization: Normalizes the output of the attention and feed-forward layers.

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        
        # Initializing the parameters
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)  # Encoder-Decoder Attention
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)  # Layer Normalization after self-attention
        self.norm2 = nn.LayerNorm(d_model)  # Layer Normalization after cross-attention
        self.norm3 = nn.LayerNorm(d_model)  # Layer Normalization after feed-forward network
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attention_output = self.self_attention(x, x, x, tgt_mask)  # Self-Attention
        x = self.norm1(x + self.dropout(attention_output))  # Residual connection
        attention_output = self.cross_attention(x, enc_output, enc_output, src_mask)  # Encoder-Decoder Attention
        x = self.norm2(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # Residual connection
        return x
    
# Transformer Model:
# The Transformer model is a neural network architecture designed for sequence-to-sequence tasks, such as machine translation.
# It consists of an encoder and a decoder, each composed of multiple layers.

# Key components of the Transformer model include:


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, dropout=0.1):
        super(Transformer, self).__init__()
        # Initializing the parameters
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)  # Final linear layer for output
        self.dropout = nn.Dropout(dropout)
    
    def generate_mask(self, src, tgt):
        # Create source mask
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # Create target mask
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        # Create causal mask for target sequence
        nopeak_mask = (1-torch.triu(torch.ones(seq_length, seq_length), diagonal=1)).bool( )
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        # Generate masks for source and target sequences
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # Embed source and target sequences
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        # Embed target sequences
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        encoding_output = src_embedded
        # Pass through encoder layers
        for encoding_layer in self.encoder_layers:
            encoding_output = encoding_layer(encoding_output, src_mask)
        
        # Pass through decoder layers
        decoding_output = tgt_embedded
        for decoding_layer in self.decoder_layers:
            decoding_output = decoding_layer(decoding_output, encoding_output, src_mask, tgt_mask)

        output = self.fc(decoding_output)  # Final linear layer for output
        return output