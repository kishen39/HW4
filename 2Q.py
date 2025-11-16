import numpy as np
import math
import matplotlib.pyplot as plt

# =========================================================
# 1. Small toy dataset
# =========================================================
sentences = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "a small cat chased a big mouse",
    "deep learning models use attention",
    "transformers encode context",
    "attention helps capture long dependencies",
    "we build a tiny transformer encoder",
    "each word attends to others",
    "this is a simple example",
    "nlp models process text in batches",
]

# =========================================================
# 2. Tokenize + build vocab
# =========================================================
def build_vocab(sentences, min_freq=1):
    from collections import Counter
    counter = Counter()
    for s in sentences:
        for w in s.strip().split():
            counter[w.lower()] += 1
    # Reserve 0,1 for special tokens
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab

def tokenize(sent, vocab):
    return [vocab.get(w.lower(), vocab["[UNK]"]) for w in sent.strip().split()]

vocab = build_vocab(sentences)
inv_vocab = {i: w for w, i in vocab.items()}

token_ids = [tokenize(s, vocab) for s in sentences]

def pad_sequences(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    batch = np.full((len(seqs), max_len), pad_id, dtype=np.int32)
    mask = np.zeros((len(seqs), max_len), dtype=np.float32)  # 1 = real, 0 = pad
    for i, s in enumerate(seqs):
        batch[i, :len(s)] = s
        mask[i, :len(s)] = 1.0
    return batch, mask

batch_tokens, mask = pad_sequences(token_ids, pad_id=vocab["[PAD]"])

# =========================================================
# 3. Embeddings + sinusoidal positional encoding
# =========================================================
np.random.seed(0)
d_model = 16          # embedding / model dimension
num_heads = 2
d_head = d_model // num_heads
vocab_size = len(vocab)

# Token embedding matrix
embed_matrix = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.1

def embed(tokens):
    # tokens: (batch, seq_len) -> (batch, seq_len, d_model)
    return embed_matrix[tokens]

def get_sinusoidal_positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

max_len = batch_tokens.shape[1]
pos_encoding = get_sinusoidal_positional_encoding(max_len, d_model)

# Input to encoder: token embedding + positional encoding
x0 = embed(batch_tokens) + pos_encoding[np.newaxis, :, :]   # (B, T, d_model)

# =========================================================
# 4. Mini Transformer encoder (1 layer)
# =========================================================
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def split_heads(x, num_heads):
    # (B, T, d_model) -> (B, H, T, d_head)
    B, T, D = x.shape
    d_head = D // num_heads
    x = x.reshape(B, T, num_heads, d_head)
    return x.transpose(0, 2, 1, 3)

def combine_heads(x):
    # (B, H, T, d_head) -> (B, T, d_model)
    B, H, T, d_head = x.shape
    x = x.transpose(0, 2, 1, 3)
    return x.reshape(B, T, H * d_head)

# --- Self-attention parameters (shared across heads via projections) ---
W_q = np.random.randn(d_model, d_model).astype(np.float32) / math.sqrt(d_model)
W_k = np.random.randn(d_model, d_model).astype(np.float32) / math.sqrt(d_model)
W_v = np.random.randn(d_model, d_model).astype(np.float32) / math.sqrt(d_model)
W_o = np.random.randn(d_model, d_model).astype(np.float32) / math.sqrt(d_model)

def scaled_dot_product_attention(q, k, v, mask=None):
    # q,k,v: (B, H, T, d_head)
    d_head = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(d_head)  # (B,H,T,T)
    if mask is not None:
        # mask is 1 at PAD positions -> add large negative
        scores = scores + (mask * -1e9)
    weights = softmax(scores, axis=-1)  # attention weights
    output = np.matmul(weights, v)      # (B,H,T,d_head)
    return output, weights

def self_attention_block(x, mask, num_heads=2, return_attention=False):
    # x: (B, T, d_model)
    q = np.matmul(x, W_q)
    k = np.matmul(x, W_k)
    v = np.matmul(x, W_v)

    qh = split_heads(q, num_heads)
    kh = split_heads(k, num_heads)
    vh = split_heads(v, num_heads)

    # mask: (B, T) -> pad_mask: (B,1,1,T), 1 where PAD
    pad_mask = (1.0 - mask)[:, np.newaxis, np.newaxis, :]

    context, attn_weights = scaled_dot_product_attention(qh, kh, vh, mask=pad_mask)

    context_combined = combine_heads(context)      # (B, T, d_model)
    out = np.matmul(context_combined, W_o)         # final projection
    if return_attention:
        return out, attn_weights
    return out

# --- Feed-forward network ---
d_ff = 64
W1 = np.random.randn(d_model, d_ff).astype(np.float32) / math.sqrt(d_model)
b1 = np.zeros((d_ff,), dtype=np.float32)
W2 = np.random.randn(d_ff, d_model).astype(np.float32) / math.sqrt(d_ff)
b2 = np.zeros((d_model,), dtype=np.float32)

def relu(x):
    return np.maximum(0, x)

def feed_forward(x):
    # (B, T, d_model) -> (B, T, d_model)
    h = np.matmul(x, W1) + b1
    h = relu(h)
    out = np.matmul(h, W2) + b2
    return out

def layer_norm(x, eps=1e-5):
    # Simple LN with gamma=1, beta=0
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def encoder_layer(x, mask, num_heads=2, return_attention=False):
    # 1) Self-attention + Add & Norm
    attn_out, attn_weights = self_attention_block(
        x, mask, num_heads=num_heads, return_attention=True
    )
    x = layer_norm(x + attn_out)

    # 2) Feed-forward + Add & Norm
    ff_out = feed_forward(x)
    x = layer_norm(x + ff_out)

    if return_attention:
        return x, attn_weights
    return x

# Run one encoder layer
x_encoded, attn_weights = encoder_layer(
    x0, mask, num_heads=num_heads, return_attention=True
)

# =========================================================
# 5. Show: input tokens, contextual embeddings, attention
# =========================================================
np.set_printoptions(precision=3, suppress=True)

print("=== Input tokens (after tokenization) ===")
for i, s in enumerate(sentences):
    length = int(mask[i].sum())
    toks = [inv_vocab[t] for t in batch_tokens[i, :length]]
    ids = batch_tokens[i, :length].tolist()
    print(f"Sentence {i}: {s}")
    print("  Tokens:", toks)
    print("  Token IDs:", ids)
    print()

print("=== Final contextual embeddings for sentence 0 ===")
sent_idx = 0
seq_len = int(mask[sent_idx].sum())
print("Shape:", x_encoded[sent_idx, :seq_len].shape)   # (T0, d_model)
print(x_encoded[sent_idx, :seq_len])

# -----------------------------
# Attention heatmap (sentence 0, head 0)
# -----------------------------
head_idx = 0
attn_mat = attn_weights[sent_idx, head_idx, :seq_len, :seq_len]
tokens = [inv_vocab[t] for t in batch_tokens[sent_idx, :seq_len]]

plt.figure()
plt.imshow(attn_mat)
plt.xticks(range(seq_len), tokens, rotation=45)
plt.yticks(range(seq_len), tokens)
plt.colorbar()
plt.title("Self-attention (sentence 0, head 0)")
plt.tight_layout()
plt.show()
