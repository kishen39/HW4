import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Embedding layer: convert character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN layer
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer: predict next character
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len)
        
        # Embed the input characters (EMBEDDING LAYER - converts discrete chars to continuous vectors)
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        
        # Pass through RNN
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        
        output, hidden = self.rnn(embedded, hidden)  # output: (batch_size, seq_len, hidden_size)
        
        # Get the last output for prediction
        last_output = output[:, -1, :]  # (batch_size, hidden_size)
        
        # Predict next character
        logits = self.fc(last_output)  # (batch_size, vocab_size)
        
        return logits, hidden
    
    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

def prepare_data(text, seq_length=25, train_ratio=0.8):
    """Prepare training data from text with train/validation split"""
    # Create character vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Convert text to indices
    data = [char_to_idx[ch] for ch in text]
    
    # Create training sequences
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    # Split into train and validation
    split_idx = int(len(sequences) * train_ratio)
    
    train_seq = sequences[:split_idx]
    train_targ = targets[:split_idx]
    val_seq = sequences[split_idx:]
    val_targ = targets[split_idx:]
    
    return (train_seq, train_targ, val_seq, val_targ, 
            char_to_idx, idx_to_char, vocab_size)

def train_model_with_validation(model, train_seq, train_targ, val_seq, val_targ, 
                               epochs=1000, lr=0.01):
    """Train the RNN model with validation tracking"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert to PyTorch tensors
    train_seq_tensor = torch.LongTensor(train_seq)
    train_targ_tensor = torch.LongTensor(train_targ)
    val_seq_tensor = torch.LongTensor(val_seq)
    val_targ_tensor = torch.LongTensor(val_targ)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass (TEACHER FORCING: using ground truth as input)
        logits, _ = model(train_seq_tensor)
        loss = criterion(logits, train_targ_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_seq_tensor)
            val_loss = criterion(val_logits, val_targ_tensor)
            val_losses.append(val_loss.item())
        
        if epoch % 200 == 0:
            print(f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return train_losses, val_losses

def generate_text(model, start_string, char_to_idx, idx_to_char, length=100, temperature=1.0):
    """Generate text using the trained model with temperature sampling"""
    model.eval()
    
    # Convert start string to indices
    chars = [ch for ch in start_string]
    input_seq = torch.LongTensor([char_to_idx[ch] for ch in chars]).unsqueeze(0)
    
    generated = chars.copy()
    
    with torch.no_grad():
        hidden = None
        
        # SAMPLING LOOP: autoregressively generate characters
        for _ in range(length):
            # Get prediction
            logits, hidden = model(input_seq, hidden)
            
            # Apply temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next character from probability distribution
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_char_idx]
            
            generated.append(next_char)
            
            # Update input sequence for next iteration
            input_seq = torch.LongTensor([[next_char_idx]])
    
    return ''.join(generated)

# Larger text dataset for better training
text = """
In a quiet village nestled between rolling hills, there lived a curious young inventor named Elara. 
She spent her days tinkering with gears and springs, creating marvelous contraptions that amazed the villagers. 
One morning, Elara discovered an ancient map hidden in her grandfather's attic. The map showed the path to 
a legendary crystal cave deep in the Whispering Woods. With her trusted mechanical owl, Cogsworth, by her side, 
Elara embarked on an extraordinary adventure. They faced mysterious challenges and met magical creatures along 
the way. The trees seemed to whisper secrets as they journeyed deeper into the enchanted forest. After three days 
of travel, they finally reached the crystal cave, where glowing gems illuminated ancient writings on the walls. 
Elara realized the crystals held the wisdom of forgotten civilizations, waiting to be rediscovered by those brave 
enough to seek them. She knew this was only the beginning of her incredible journey into the world of magic and machinery.
"""

# Experiment with different hyperparameters
def run_experiment(seq_length=25, hidden_size=128, epochs=1500):
    print(f"\n=== Experiment: seq_length={seq_length}, hidden_size={hidden_size} ===")
    
    # Prepare data
    train_seq, train_targ, val_seq, val_targ, char_to_idx, idx_to_char, vocab_size = prepare_data(
        text, seq_length=seq_length)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training sequences: {len(train_seq)}, Validation sequences: {len(val_seq)}")
    
    # Create and train model
    model = CharRNN(vocab_size, hidden_size)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    train_losses, val_losses = train_model_with_validation(
        model, train_seq, train_targ, val_seq, val_targ, epochs=epochs, lr=0.01)
    
    return model, train_losses, val_losses, char_to_idx, idx_to_char

# Run multiple experiments
plt.figure(figsize=(15, 10))

# Experiment 1: Baseline
model1, train1, val1, char_to_idx, idx_to_char = run_experiment(seq_length=25, hidden_size=128)

# Experiment 2: Longer sequence length
model2, train2, val2, _, _ = run_experiment(seq_length=40, hidden_size=128)

# Experiment 3: Larger hidden size
model3, train3, val3, _, _ = run_experiment(seq_length=25, hidden_size=256)

# 1. Training/Validation Loss Curves
plt.subplot(2, 2, 1)
plt.plot(train1, label='Train (seq=25, hidden=128)')
plt.plot(val1, label='Val (seq=25, hidden=128)')
plt.plot(train2, '--', label='Train (seq=40, hidden=128)')
plt.plot(val2, '--', label='Val (seq=40, hidden=128)')
plt.title('Training/Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')

plt.subplot(2, 2, 2)
plt.plot(train1, label='Train (seq=25, hidden=128)')
plt.plot(val1, label='Val (seq=25, hidden=128)')
plt.plot(train3, '--', label='Train (seq=25, hidden=256)')
plt.plot(val3, '--', label='Val (seq=25, hidden=256)')
plt.title('Effect of Hidden Size on Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')

# 2. Temperature-controlled generations
start_string = "Elara discovered"
print(f"\n{'='*60}")
print("TEMPERATURE-CONTROLLED GENERATIONS")
print(f"{'='*60}")

temperatures = [0.7, 1.0, 1.2]
for temp in temperatures:
    print(f"\nTemperature τ = {temp}:")
    print("-" * 40)
    generated = generate_text(model1, start_string, char_to_idx, idx_to_char, 
                            length=300, temperature=temp)
    print(generated)

# Reflection analysis
print(f"\n{'='*60}")
print("REFLECTION ON HYPERPARAMETER EFFECTS")
print(f"{'='*60}")

reflection = """
REFLECTION:

• SEQUENCE LENGTH: Longer sequences (40 vs 25) provide more context for predictions but require more memory and training time. 
  They capture longer-range dependencies but may overfit on smaller datasets. The embedding layer processes more characters per sequence.

• HIDDEN SIZE: Larger hidden dimensions (256 vs 128) increase model capacity and ability to learn complex patterns, but also 
  increase computational cost and risk of overfitting. This affects both the RNN hidden state and embedding dimensions.

• TEMPERATURE: Lower temperatures (τ=0.7) produce more conservative, repetitive text by amplifying high-probability characters. 
  Higher temperatures (τ=1.2) increase diversity and creativity but may generate nonsensical text. The sampling loop becomes 
  more exploratory with higher temperature.

• TRADEOFFS: Teacher forcing provides stable training but creates exposure bias. There's always a balance between model capacity, 
  training data size, and generalization performance. The embedding layer's quality directly impacts the model's semantic understanding.
"""

print(reflection)

plt.tight_layout()
plt.show()