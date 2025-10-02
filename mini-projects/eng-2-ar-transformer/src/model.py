import torch
import torch.nn as nn
import math
from tokenizers import Tokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length= 512, dropout= 0.1):
        super().__init__()
        self.dropout= nn.Dropout(p= dropout)

        # Create positional encoding matrix
        position= torch.arange(max_seq_length).unsqueeze(1) # (max_seq_length, 1)
        div_term= torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe= torch.zeros(max_seq_length, d_model)
        pe[:, 0::2]= torch.sin(position * div_term) # Even positions: sin
        pe[:, 1::2]= torch.cos(position * div_term) # Odd positions: cos

        # Register as buffer (non-learnable param)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        # Add positional encoding to input
        x= x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length= 512, dropout= 0.1, padding_idx= 1):
        super().__init__()
        self.embedding= nn.Embedding(vocab_size, d_model, padding_idx= padding_idx)
        self.positional_encoding= PositionalEncoding(d_model, max_seq_length, dropout)
        self.d_model= d_model
    
    def forward(self, x):
        # Scale embeddings by sqrt(d_model) as mentioned in the paper
        embeddings= self.embedding(x) * math.sqrt(self.d_model)
        return self.positional_encoding(embeddings)
    

def scaled_dot_product_attention(query, key, value, mask= None, dropout= None):
    """
    Compute scaled dot-product attention.
    
    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
        key: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
        value: Value tensor of shape (batch_size, num_heads, seq_len, d_k)
        mask: Mask tensor broadcastable to (batch_size, num_heads, seq_len, seq_len)
        dropout: Dropout layer
    
    Returns:
        attn_output: Attention output tensor of shape (batch_size, num_heads, seq_len, d_k)
        attn_weights: Attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len)
    """
    d_k= query.size(-1)
    scores= torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores= scores.masked_fill(mask == 0, -1e9)
    attn_weights= scores.softmax(dim= -1)

    if dropout is not None:
        attn_weights= dropout(attn_weights)
    attn_output= torch.matmul(attn_weights, value)
    return attn_output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout= 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model= d_model
        self.num_heads= num_heads
        self.d_k= d_model // num_heads

        # Linear projections for query, key and value
        self.w_q= nn.Linear(d_model, d_model)
        self.w_k= nn.Linear(d_model, d_model)
        self.w_v= nn.Linear(d_model, d_model)
        self.w_o= nn.Linear(d_model, d_model)

        self.dropout= nn.Dropout(dropout)

    def forward(self, query, key, value, mask= None):
        """
        Args:
            query: Tensor of shape (batch_size, seq_len, d_model)
            key: Tensor of shape (batch_size, seq_len, d_model)
            value: Tensor of shape (batch_size, seq_len, d_model)
            mask: Mask tensor broadcastable to (batch_size, seq_len, seq_len) or (batch_size, num_heads, seq_len, seq_len)
        
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """

        batch_size= query.size(0)

        # Linear projections and split into heads
        Q= self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K= self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V= self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        # Apply attention
        attn_output, attn_weights= scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        # Concat heads and apply final linear
        attn_output= attn_output.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        output= self.w_o(attn_output)
        return output

def generate_causal_mask(seq_len):
    """
    Generate a causal mask for decoder self-attention.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        mask: Tensor of shape (1, 1, seq_len, seq_len) with zeros for masked positions
    """
    mask= torch.triu(torch.ones(seq_len, seq_len), diagonal= 1).bool()
    return mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)
def generate_padding_mask(seq, pad_id= 1):
    """
    Generate a padding mask for sequences.
    
    Args:
        seq: Tensor of shape (batch_size, seq_len) containing token indices
        pad_id: Padding token ID
    
    Returns:
        mask: Tensor of shape (batch_size, 1, 1, seq_len) with zeros for padded positions
    """
    mask= (seq != pad_id).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
    return mask

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout= 0.1):
        super().__init__()
        self.self_attn= MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward= nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1= nn.LayerNorm(d_model)
        self.norm2= nn.LayerNorm(d_model)
        self.dropout1= nn.Dropout(dropout)
        self.dropout2= nn.Dropout(dropout)

    def forward(self, x, mask= None):
        # Self-attention with residual connection
        attn_output= self.self_attn(x, x, x, mask)
        x= x + self.dropout1(attn_output)
        x= self.norm1(x)

        # Feed-forward with residual connection
        ff_output= self.feed_forward(x)
        x= x + self.dropout2(ff_output)
        x= self.norm2(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout= 0.1):
        super().__init__()
        self.layers= nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x, mask= None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model) - embedded input
            mask: Optional mask for padding
        
        Returns:
            Tensor of same shape as input
        """
        for layer in self.layers:
            x= layer(x, mask)
        return x
        
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout= 0.1):
        super().__init__()
        # Masked self-attention (for decoder's own sequence)
        self.self_attn= MultiHeadAttention(d_model, num_heads, dropout)
        # Cross-attention (encoder outputs -> decoder)
        self.cross_attn= MultiHeadAttention(d_model, num_heads, dropout)
        # Feed-forward network
        self.feed_forward= nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        # Normalization layers
        self.norm1= nn.LayerNorm(d_model)
        self.norm2= nn.LayerNorm(d_model)
        self.norm3= nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout1= nn.Dropout(dropout)
        self.dropout2= nn.Dropout(dropout)
        self.dropout3= nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask= None, tgt_mask= None):
        """
        Args:
            x: Decoder input (target sequence) of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Source mask (for padding) of shape (batch_size, 1, 1, src_seq_len)
            tgt_mask: Target mask (causal mask) of shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        # 1. Masked self-attention with residual connection
        self_attn_output= self.self_attn(x, x, x, mask= tgt_mask)
        x= x + self.dropout1(self_attn_output)
        x= self.norm1(x)
        # 2. Cross-attention with residual connection
        # Query: decoder state, Key/Value: encoder outputs
        cross_attn_output= self.cross_attn(x, encoder_output, encoder_output, mask= src_mask)
        x= x + self.dropout2(cross_attn_output)
        x= self.norm2(x)

        # FFN with res conn
        ff_output= self.feed_forward(x)
        x= x + self.dropout3(ff_output)
        x= self.norm3(x)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input (target sequence) of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Source mask for padding of shape (batch_size, 1, 1, src_seq_len)
            tgt_mask: Target causal mask of shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
    
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 num_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 d_ff,
                 max_seq_length= 512,
                 dropout= 0.1):
        super().__init__()

        # Embedding layers
        self.src_embed= TransformerEmbeddings(src_vocab_size, d_model, max_seq_length, dropout)
        self.tgt_embed= TransformerEmbeddings(tgt_vocab_size, d_model, max_seq_length, dropout)

        # Encoder and Decoder stacks
        self.encoder= TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder= TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)

        # Final output proj
        self.output_projection= nn.Linear(d_model, tgt_vocab_size)

        # Initialize params
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask= None, tgt_mask= None):
        """
        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            tgt: Target token indices of shape (batch_size, tgt_seq_len)
            src_mask: Source padding mask of shape (batch_size, src_seq_len)
            tgt_mask: Target causal mask of shape (batch_size, tgt_seq_len, tgt_seq_len)
        """
        # Embed source and tgt seq
        src_embedded= self.src_embed(src)
        tgt_embedded= self.tgt_embed(tgt)

        # Encode source sequence
        encoder_output= self.encoder(src_embedded, src_mask)
        decoder_output= self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)

        # Project to target vocab
        output= self.output_projection(decoder_output)
        return output
    def encode(self, src, src_mask= None):
        """Encode source seq (useful for inference)"""
        src_embedded= self.src_embed(src)
        return self.encoder(src_embedded, src_mask)
    def decode(self, tgt, encoder_output, src_mask= None, tgt_mask= None):
        """Decode target seq given encoder output (useful for inference)"""
        tgt_embedded= self.tgt_embed(tgt)
        decoder_output= self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        
        return self.output_projection(decoder_output)