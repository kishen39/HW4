# HW4
---
**Kishen
700762472**
---

# 📊 **Q1 Character-Level RNN Language Model**

1️⃣ **output Data:**
```python
Temperature τ = 0.7:
----------------------------------------
Elara discovered an ancient map hidden in her grandfather's attic. The map showed the path to a legendary crystal cave deep in the Whispering Woods. With her trusted mechanical owl, Cogsworth, by her side, Elara embarked on an extraordinary adventure. They faced mysterious challenges and met magical creatures along the way. The trees seemed to whisper secrets as they journeyed deeper into the enchanted forest. After three days of travel, they finally reached the crystal cave, where glowing gems illuminated ancient writings on the walls. Elara realized the crystals held the wisdom of forgotten civilizations, waiting to be rediscovered by those brave enough to seek them.
```

# 📊 **Q2 Mini Transformer Encoder for Sentences**

1️⃣ **out put**
```python
Shape: (6, 16)
[[-0.184  2.256 -0.085 -0.239 -1.924  0.431 -2.051  0.876 -0.632  0.409
   0.397  0.487 -0.765  0.259  0.227  0.537]
 [ 0.707  1.807  0.023 -0.399 -2.009  0.208 -2.07   1.19  -0.402  0.639
   0.533  0.433 -1.081  0.265  0.061  0.097]
 [ 1.248  0.676  0.352 -0.79  -2.037  0.701 -1.99   1.099 -0.495  0.668
   0.79   0.626 -1.273  0.197  0.248 -0.02 ]
 [ 0.332  0.295  0.577 -1.422 -1.838  0.977 -1.777  1.146 -0.919  0.959
   0.646  0.871 -1.021  0.245  0.758  0.173]
 [-0.753  0.593  0.816 -1.304 -1.749  0.89  -1.67   1.259 -0.948  0.709
   0.647  1.22  -0.822  0.378  0.586  0.148]
 [-1.115  1.779  0.828 -1.521 -1.285  0.174 -1.684  1.078 -0.937  0.58
   0.653  0.618 -0.521  0.481  0.505  0.367]]
```

# 📊 **Q3 Implement Scaled Dot-Product Attention**

1️⃣ **out put**
```python
=== Output vectors (Attention * V) ===
tensor([[0.500, 0.500],
        [0.330, 0.670]])

=== Softmax comparison: raw vs scaled ===
Softmax over raw scores (no scaling):
tensor([[0.500, 0.500],
        [0.269, 0.731]])

Softmax over scaled scores:
tensor([[0.500, 0.500],
        [0.330, 0.670]])
```
