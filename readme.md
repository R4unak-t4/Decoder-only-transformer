# Decoder-Only Transformer

This repository contains an implementation of a simplified **Decoder-Only Transformer** model using PyTorch and PyTorch Lightning. The model demonstrates the core components of a transformer architecture, including tokenization, positional encoding, attention mechanism, and training.

## Features

- **Tokenization and Embedding**: Converts tokens into dense embeddings.
- **Sinusoidal Positional Encoding**: Adds positional information to embeddings.
- **Scaled Dot-Product Attention**: Implements causal attention for decoder-only models.
- **Training with PyTorch Lightning**: Includes a simple training loop with the Adam optimizer.

## Mathematical Formulations

### Positional Encoding

The positional encoding is defined as:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Scaled Dot-Product Attention

Attention weights are computed as:

```
Attention(Q, K, V) = Softmax(QK^T / sqrt(d_model)) * V
```

Where:
- **Q**: Query matrix
- **K**: Key matrix  
- **V**: Value matrix
- **d_model**: Dimension of the model

### Adam Optimizer

The Adam optimizer updates parameters using the following equations:

```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²

m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

θ_t = θ_{t-1} - η * m̂_t / (sqrt(v̂_t) + ε)
```

Where:
- **η**: Learning rate
- **β₁, β₂**: Exponential decay rates
- **ε**: Small constant for numerical stability

## Code Implementation

### Tokenization

```python
token_to_id = {
    'what': 0,
    'is': 1,
    'transformer': 2,
    'magic': 3,
    '<EOS>': 4
}

id_to_token = dict(map(reversed, token_to_id.items()))
```

### Positional Encoding

```python
import torch
import torch.nn as nn

class PositionEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        super().__init__()
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        div_term = 1 / torch.tensor(10000.0) ** (embedding_index / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0), :]
```

### Attention Mechanism

```python
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, encoding_for_q, encoding_for_k, encoding_for_v, mask=None):
        q = self.W_q(encoding_for_q)
        k = self.W_k(encoding_for_k)
        v = self.W_v(encoding_for_v)
        sims = torch.matmul(q, k.transpose(0, 1)) / torch.sqrt(torch.tensor(k.size(1), dtype=torch.float))
        if mask is not None:
            sims = sims.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(sims, dim=1)
        return torch.matmul(attention_weights, v)
```

### Decoder-Only Transformer

```python
import pytorch_lightning as l
from torch.optim import Adam

class DecoderOnlyTransformer(l.LightningModule):
    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        super().__init__()
        self.we = nn.Embedding(num_tokens, d_model)
        self.pe = PositionEncoding(d_model, max_len)
        self.attention = Attention(d_model)
        self.fc = nn.Linear(d_model, num_tokens)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)
        mask = torch.tril(torch.ones((token_ids.size(0), token_ids.size(0)), device=self.device)) == 0
        attention_values = self.attention(position_encoded, position_encoded, position_encoded, mask=mask)
        residual_connection_values = attention_values + position_encoded
        logits = self.fc(residual_connection_values)
        return logits

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output = self.forward(input_tokens[0])
        loss = self.loss(output.view(-1, output.size(-1)), labels[0].view(-1))
        return loss
```

## Training

To train the model, use the following code:

```python
trainer = l.Trainer(max_epochs=30)
trainer.fit(model, train_dataloaders=dataloader)
```

## Making Predictions

Run the model to predict tokens:

```python
# Initialize model input
model_input = torch.tensor([token_to_id["what"], token_to_id["is"], token_to_id["transformer"], token_to_id["<EOS>"]])
predictions = model(model_input)
predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
predicted_ids = predicted_id

# Generate sequence
input_length = len(model_input)
max_length = 6

for i in range(input_length, max_length):
    if predicted_id == token_to_id["<EOS>"]:
        break
    model_input = torch.cat((model_input, predicted_id))
    predictions = model(model_input)
    predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
    predicted_ids = torch.cat((predicted_ids, predicted_id))

# Display results
print("Predicted Tokens:")
for id in predicted_ids:
    print("\t", id_to_token[id.item()])
```

## Usage

1. Define your vocabulary and create token mappings
2. Initialize the model with appropriate parameters
3. Prepare your training data
4. Train the model using PyTorch Lightning
5. Use the trained model for text generation

This implementation provides a foundational understanding of transformer architecture and can be extended for more complex applications.