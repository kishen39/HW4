import torch
import math

# Make prints compact and readable
torch.set_printoptions(precision=3, sci_mode=False)

# ---------------------------------------------------
# 1. Scaled Dot-Product Attention function
#    Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
# ---------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, n_q, d_k)
    K: (batch, n_k, d_k)
    V: (batch, n_k, d_v)
    mask: (batch, n_q, n_k) with 0 or -inf (optional)
    """
    d_k = Q.size(-1)

    # (batch, n_q, n_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask  # usually mask has -inf where we want to ignore

    attn = torch.softmax(scores, dim=-1)      # attention weights
    output = torch.matmul(attn, V)           # weighted sum of V
    return output, attn, scores


# ---------------------------------------------------
# 2. Tiny, simple example with integer matrices
# ---------------------------------------------------
# batch_size = 1, n_q = 2, n_k = 2, d_k = d_v = 2

Q = torch.tensor([[
    [1., 0.],   # query 0
    [0., 1.]    # query 1
]])  # shape: (1, 2, 2)

K = torch.tensor([[
    [1., 0.],   # key 0
    [1., 1.]    # key 1
]])  # shape: (1, 2, 2)

V = torch.tensor([[
    [1., 0.],   # value 0
    [0., 1.]    # value 1
]])  # shape: (1, 2, 2)

# ---------------------------------------------------
# 3. Run attention
# ---------------------------------------------------
output_scaled, attn_scaled, scores_scaled = scaled_dot_product_attention(Q, K, V)

# For softmax stability / peaky-ness check: raw vs scaled
scores_raw = torch.matmul(Q, K.transpose(-2, -1))           # no 1/sqrt(d_k)
attn_raw = torch.softmax(scores_raw, dim=-1)                # softmax(no scaling)
attn_scaled_only = torch.softmax(scores_raw / math.sqrt(2), dim=-1)  # with scaling


# ---------------------------------------------------
# 4. Print everything (rounded, simple)
# ---------------------------------------------------
print("=== Q ===")
print(Q[0])

print("\n=== K ===")
print(K[0])

print("\n=== V ===")
print(V[0])

print("\n=== Raw scores (Q K^T, no scaling) ===")
print(scores_raw[0])

print("\n=== Scaled scores (Q K^T / sqrt(d_k)) ===")
print(scores_scaled[0])

print("\n=== Attention weights (softmax of scaled scores) ===")
print(attn_scaled[0])

print("\n=== Output vectors (Attention * V) ===")
print(output_scaled[0])

print("\n=== Softmax comparison: raw vs scaled ===")
print("Softmax over raw scores (no scaling):")
print(attn_raw[0])

print("\nSoftmax over scaled scores:")
print(attn_scaled_only[0])

print("\nMax prob per query (no scaling):", attn_raw[0].max(dim=-1).values)
print("Max prob per query (scaled):   ", attn_scaled_only[0].max(dim=-1).values)
