---
title: "PyTorch Cheatsheet - Practical Guide"
date: 2025-11-25 16:00:00
categories: [machine learning, deep learning]
tags: [pytorch, machine learning, deep learning]    
image:
  path: /assets/imgs/headers/pytorch.png
---

## Introduction

PyTorch is an open-source deep learning library developed by Facebook AI Research (FAIR). It offers maximum flexibility for building and training neural networks thanks to its automatic gradient computation system (autograd) and intuitive Pythonic syntax.

**Why PyTorch?**
- **Dynamic**: Define-by-run computational graph
- **Pythonic**: Integrates naturally with the Python ecosystem
- **Performant**: Native GPU support and advanced optimizations
- **Flexible**: Ideal for research and production
- **Rich Ecosystem**: TorchVision, TorchText, TorchAudio, etc.

## Installation

PyTorch can be installed with or without GPU support. CUDA support significantly accelerates model training by using NVIDIA GPUs.

```bash
# CPU only
pip install torch torchvision

# GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Tensors

Tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but with GPU support and automatic gradient computation. A tensor can be a scalar (0D), a vector (1D), a matrix (2D), or a multidimensional array (nD).

### Creating Tensors

PyTorch offers several methods to create tensors according to your needs:

```python
import torch

# From a list
x = torch.tensor([[1, 2], [3, 4]])

# Empty tensor
x = torch.empty(3, 4)

# Filled with zeros
x = torch.zeros(2, 3)

# Filled with ones
x = torch.ones(2, 3)

# Random values (uniform distribution [0, 1))
x = torch.rand(2, 3)

# Random values (normal distribution)
x = torch.randn(2, 3)

# Sequence of values
x = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
x = torch.linspace(0, 10, 5)  # [0, 2.5, 5, 7.5, 10]

# Identity matrix
x = torch.eye(3)

# Tensor with a specific type
x = torch.tensor([1, 2, 3], dtype=torch.float32)
```

### Tensor Properties

Each tensor has important attributes that define its structure and data type:

```python
x = torch.randn(2, 3, 4)

print(x.shape)        # torch.Size([2, 3, 4])
print(x.size())       # torch.Size([2, 3, 4])
print(x.dtype)        # torch.float32
print(x.device)       # cpu or cuda
print(x.ndim)         # 3 (number of dimensions)
print(x.numel())      # 24 (total number of elements)
```

## Tensor Operations

### Arithmetic Operations

PyTorch supports all classic arithmetic operations. These operations can be performed element-wise:

```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Addition
z = x + y
z = torch.add(x, y)

# Subtraction
z = x - y
z = torch.sub(x, y)

# Multiplication (element-wise)
z = x * y
z = torch.mul(x, y)

# Division
z = x / y
z = torch.div(x, y)

# Power
z = x ** 2
z = torch.pow(x, 2)
```

### Matrix Operations

Matrix operations are essential for neural networks. PyTorch offers several optimized methods for matrix multiplication:

```python
x = torch.randn(2, 3)
y = torch.randn(3, 4)

# Matrix multiplication
z = torch.mm(x, y)           # (2, 4)
z = x @ y                    # (2, 4)

# Batch multiplication
x = torch.randn(10, 3, 4)
y = torch.randn(10, 4, 5)
z = torch.bmm(x, y)          # (10, 3, 5)

# Dot product
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = torch.dot(x, y)          # 32

# Transpose
x = torch.randn(2, 3)
y = x.T                      # (3, 2)
y = x.transpose(0, 1)        # (3, 2)
```

### Reshape and View

Reshaping tensors is crucial for adapting data to different network layers. `view()` requires a contiguous tensor in memory, while `reshape()` can copy data if necessary:

```python
x = torch.randn(2, 3, 4)

# Reshape
y = x.view(6, 4)           # (6, 4)
y = x.view(-1, 4)          # (-1 computed automatically)
y = x.reshape(2, 12)       # (2, 12)

# Flatten
y = x.flatten()            # (24,)
y = x.view(-1)             # (24,)

# Squeeze/Unsqueeze
x = torch.randn(1, 3, 1, 4)
y = x.squeeze()            # (3, 4) - removes dimensions of size 1
y = x.squeeze(0)           # (3, 1, 4) - removes dimension 0
y = x.unsqueeze(0)         # (1, 1, 3, 1, 4) - adds a dimension
```

### Indexing and Slicing

Indexing works like NumPy, allowing access and modification of specific parts of a tensor:

```python
x = torch.randn(4, 5)

# Basic indexing
y = x[0]              # First row
y = x[:, 0]           # First column
y = x[1:3, 2:4]       # Submatrix

# Boolean indexing
mask = x > 0
y = x[mask]           # All positive elements

# Fancy indexing
indices = torch.tensor([0, 2])
y = x[indices]        # Rows 0 and 2
```

### Concatenation and Split

These operations allow combining or separating tensors along a specific dimension:

```python
x = torch.randn(2, 3)
y = torch.randn(2, 3)

# Concatenation
z = torch.cat([x, y], dim=0)     # (4, 3)
z = torch.cat([x, y], dim=1)     # (2, 6)

# Stack
z = torch.stack([x, y], dim=0)   # (2, 2, 3)

# Split
z = torch.randn(6, 3)
chunks = torch.split(z, 2, dim=0)  # 3 tensors of size (2, 3)
chunks = torch.chunk(z, 3, dim=0)  # 3 tensors of size (2, 3)
```

## Autograd

Autograd is PyTorch's automatic differentiation engine. It records all operations performed on tensors with `requires_grad=True` and builds a dynamic computational graph to automatically compute gradients via backpropagation.

### Automatic Gradient

Computing gradients is essential for neural network optimization:

```python
# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Computation
z = x**2 + y**3
z.backward()  # Compute gradients

print(x.grad)  # dz/dx = 2x = 4
print(y.grad)  # dz/dy = 3y² = 27

# Reset gradients
x.grad.zero_()
y.grad.zero_()
```

### no_grad Context

```python
x = torch.randn(3, requires_grad=True)

# Temporarily disable gradient computation
with torch.no_grad():
    y = x * 2
    # y does not track gradients

# Alternative
y = (x * 2).detach()  # Detach y from the computational graph
```

## Neural Networks

PyTorch uses `nn.Module` as the base class for all neural networks. Each model inherits from this class and must implement the `forward()` method which defines the forward pass.

### Simple Model

Here's an example of a fully-connected network (MLP) for classification:

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Usage
model = SimpleNet(784, 128, 10)
x = torch.randn(32, 784)  # Batch of 32 images
output = model(x)         # (32, 10)
```

### Common Layers

PyTorch provides a wide range of pre-implemented layers for building different types of networks:

```python
# Linear layer (fully connected)
fc = nn.Linear(in_features=100, out_features=50)

# 2D Convolution
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

# Pooling
pool = nn.MaxPool2d(kernel_size=2, stride=2)
pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Batch Normalization
bn = nn.BatchNorm2d(num_features=64)

# Dropout
dropout = nn.Dropout(p=0.5)

# Activation
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
```

### CNN Example

Convolutional Neural Networks (CNNs) are particularly effective for processing images. They use convolutions to extract hierarchical spatial features:

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (64, 8, 8)
        x = x.view(-1, 64 * 8 * 8)            # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN(num_classes=10)
```

### ResNet Block

Residual connections (skip connections) enable training very deep networks by solving the vanishing gradient problem. The idea is to add the input to the output of a block of layers:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

### Sequential

`nn.Sequential` allows you to quickly create models by chaining layers sequentially, without having to define a custom class:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)
```

### ModuleList and ModuleDict

These containers allow you to dynamically manage lists or dictionaries of modules while correctly registering their parameters:

```python
# ModuleList - list of modules
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(5)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

# ModuleDict - dictionary of modules
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({
            'conv': nn.Conv2d(1, 20, 5),
            'pool': nn.MaxPool2d(2),
            'fc': nn.Linear(320, 10)
        })
    
    def forward(self, x):
        x = self.layers['pool'](F.relu(self.layers['conv'](x)))
        x = x.view(x.size(0), -1)
        x = self.layers['fc'](x)
        return x
```

## RNN and LSTM

Recurrent Neural Networks (RNNs) and their variants (LSTM, GRU) are designed to process sequential data such as text, audio, or time series. They maintain a hidden state that captures information from previous time steps.

### Simple RNN

The basic RNN is the simplest form of recurrent network, but suffers from the vanishing gradient problem for long sequences:

```python
# Basic RNN
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)

# Input: (batch_size, seq_len, input_size)
x = torch.randn(32, 100, 10)
h0 = torch.zeros(2, 32, 20)  # (num_layers, batch_size, hidden_size)

output, hn = rnn(x, h0)
# output: (32, 100, 20) - output at each time step
# hn: (2, 32, 20) - final hidden state
```

### LSTM

Long Short-Term Memory (LSTM) solves the gradient problem by using gates to control information flow. It maintains both a hidden state and a cell state:

```python
# LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, 
               batch_first=True, dropout=0.2)

x = torch.randn(32, 100, 10)
h0 = torch.zeros(2, 32, 20)
c0 = torch.zeros(2, 32, 20)

output, (hn, cn) = lstm(x, (h0, c0))
# output: (32, 100, 20)
# hn: (2, 32, 20) - final hidden state
# cn: (2, 32, 20) - final cell state
```

### GRU

Gated Recurrent Unit (GRU) is a simplified variant of LSTM with fewer parameters. It is often faster to train while offering similar performance:

```python
# GRU
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, 
             batch_first=True, bidirectional=True)

x = torch.randn(32, 100, 10)
h0 = torch.zeros(2*2, 32, 20)  # *2 for bidirectional

output, hn = gru(x, h0)
# output: (32, 100, 40) - 40 because bidirectional (20*2)
```

### Complete LSTM Model

Here's a complete example of an LSTM classifier for sentiment analysis or text classification. It uses embeddings to represent words and a bidirectional LSTM to capture context in both directions:

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 dropout=0.5, bidirectional=True):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout,
                           bidirectional=bidirectional)
        
        # If bidirectional, hidden_dim * 2
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        # text: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(text))
        # embedded: (batch_size, seq_len, embedding_dim)
        
        output, (hidden, cell) = self.lstm(embedded)
        # output: (batch_size, seq_len, hidden_dim*2)
        
        # Concatenate forward and backward hidden states of the last layer
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # hidden: (batch_size, hidden_dim*2)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output

# Usage
model = LSTMClassifier(vocab_size=10000, embedding_dim=100, 
                       hidden_dim=256, output_dim=2, n_layers=2)
```

### Attention Mechanism

The attention mechanism allows the model to focus on the most relevant parts of the input sequence. It computes attention weights for each position and creates a weighted representation:

```python
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        attention_weights = torch.softmax(
            self.attention(lstm_output).squeeze(-1), dim=1
        )
        # attention_weights: (batch_size, seq_len)
        
        # Apply attention
        attention_weights = attention_weights.unsqueeze(1)
        # attention_weights: (batch_size, 1, seq_len)
        
        weighted = torch.bmm(attention_weights, lstm_output)
        # weighted: (batch_size, 1, hidden_dim)
        
        return weighted.squeeze(1), attention_weights.squeeze(1)

class LSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Apply attention
        attended, attention_weights = self.attention(lstm_output)
        
        output = self.fc(attended)
        return output, attention_weights
```

## Transformers

Transformers have revolutionized natural language processing (NLP) and computer vision. Unlike RNNs, they process the entire sequence in parallel thanks to the self-attention mechanism, making them much faster and more efficient.

### Transformer Encoder Layer

The Transformer encoder uses multi-head self-attention to capture relationships between all elements of a sequence:

```python
import math

# Transformer Encoder Layer
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,        # Model dimension
    nhead=8,           # Number of attention heads
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)

encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# Input
src = torch.randn(32, 100, 512)  # (batch, seq_len, d_model)
output = encoder(src)
# output: (32, 100, 512)
```

### Complete Transformer

Here's a complete implementation of a Transformer with encoder and decoder. Positional Encoding adds information about the position of elements in the sequence:

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, max_seq_length, num_classes):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, src, tgt):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

### Multi-Head Attention

Multi-head attention allows the model to learn different representations of information in parallel. Each "head" focuses on different aspects of the relationships between tokens:

```python
# Direct use of Multi-Head Attention
multihead_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

query = torch.randn(32, 100, 512)
key = torch.randn(32, 100, 512)
value = torch.randn(32, 100, 512)

attn_output, attn_weights = multihead_attn(query, key, value)
# attn_output: (32, 100, 512)
# attn_weights: (32, 100, 100)
```

## Training

Training a neural network follows an iterative process: forward pass (prediction), loss computation, backward pass (gradient computation), and weight update. PyTorch facilitates this process with its intuitive API.

### Complete Training Loop

Here's the standard pattern for training a model. This loop iterates over epochs and batches, performing backpropagation at each step:

```python
import torch.optim as optim

# Preparation
model = SimpleNet(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()  # Reset gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
```

## DataLoader

PyTorch's `DataLoader` facilitates loading data in batches, shuffling, and parallel loading. To use it, you must first create a `Dataset` that defines how to access your data.

### Custom Dataset

Create your own Dataset by inheriting from `torch.utils.data.Dataset` and implementing `__len__()` and `__getitem__()`:

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# Usage
dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

### Built-in Datasets

TorchVision provides popular pre-configured datasets (MNIST, CIFAR-10, ImageNet, etc.) that can be downloaded automatically:

```python
from torchvision import datasets, transforms

# Transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# MNIST
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

# CIFAR-10
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# ImageFolder
train_dataset = datasets.ImageFolder(
    root='./data/train',
    transform=transform
)
```

## Data Augmentation

Data augmentation is an essential technique for improving model generalization by creating artificial variations of training data. This reduces overfitting and improves performance.

### Image Transformations

TorchVision offers numerous transformations to augment images. Use them only on training data, not on test data:

```python
from torchvision import transforms

# Basic transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Advanced transformations
transform_advanced = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Test transformations (without augmentation)
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### AutoAugment and RandAugment

These advanced augmentation techniques were developed through neural architecture search to automatically find the best augmentation strategies:

```python
# AutoAugment
transform = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# RandAugment
transform = transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### Mixup and CutMix

Mixup linearly combines two images and their labels, creating interpolated training examples. This encourages the model to have linear behavior between classes and improves generalization:

```python
import numpy as np

# Mixup
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Usage in training loop
for data, targets in train_loader:
    data, targets_a, targets_b, lam = mixup_data(data, targets, alpha=1.0)
    outputs = model(data)
    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
    loss.backward()
    optimizer.step()
```

## Loss Functions

The loss function measures the difference between the model's predictions and the true values. Choosing the right loss function depends on your task (classification, regression, etc.).

```python
# Binary classification
criterion = nn.BCELoss()                      # Binary Cross Entropy
criterion = nn.BCEWithLogitsLoss()            # BCE with integrated sigmoid

# Multi-class classification
criterion = nn.CrossEntropyLoss()             # Softmax + NLL Loss
criterion = nn.NLLLoss()                      # Negative Log Likelihood

# Regression
criterion = nn.MSELoss()                      # Mean Squared Error
criterion = nn.L1Loss()                       # Mean Absolute Error
criterion = nn.SmoothL1Loss()                 # Huber Loss

# Others
criterion = nn.KLDivLoss()                    # Kullback-Leibler Divergence
criterion = nn.CosineEmbeddingLoss()          # Cosine Similarity Loss
```

## Optimizers

Optimizers update the model's parameters using the computed gradients. Each optimizer uses a different strategy to adjust weights and accelerate convergence.

**Quick comparison:**
- **SGD**: Simple but robust, often with momentum
- **Adam**: Adaptive, very popular, good default choice
- **AdamW**: Adam with improved weight decay
- **RMSprop**: Good for RNNs

```python
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (Adam with weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Usage
for epoch in range(num_epochs):
    train(...)
    validate(...)
    scheduler.step()  # Update learning rate
```

## GPU

Using a GPU can accelerate training by 10x to 100x depending on the model. PyTorch makes it easy to transfer data and models to the GPU with the `.to(device)` method.

### GPU Usage

To use the GPU, you must move both the model and the data to the CUDA device:

```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Move model to GPU
model = model.to(device)

# Move data to GPU
x = x.to(device)
y = y.to(device)

# In training loop
for data, targets in train_loader:
    data = data.to(device)
    targets = targets.to(device)
    
    outputs = model(data)
    loss = criterion(outputs, targets)
    # ...

# Multi-GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)
```

### GPU Operations

```python
# Create directly on GPU
x = torch.randn(3, 4, device='cuda')

# GPU information
print(torch.cuda.device_count())              # Number of GPUs
print(torch.cuda.current_device())            # Current GPU ID
print(torch.cuda.get_device_name(0))          # GPU name
print(torch.cuda.memory_allocated())          # Allocated memory
print(torch.cuda.memory_reserved())           # Reserved memory

# Clear GPU cache
torch.cuda.empty_cache()
```

## Transfer Learning

Transfer learning involves reusing a pre-trained model on a large database (like ImageNet) and adapting it to your specific task. This is particularly useful when you have limited data.

**Advantages:**
- Faster convergence
- Better results with limited data
- Reuse of pre-learned features

### Load a Pre-trained Model

Two main strategies: freeze all layers except the last one (feature extraction) or fine-tune the entire network:

```python
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10 custom classes

# Only the last layer will be trained
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

### Selective Fine-tuning

```python
# Unfreeze the last layers
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Different learning rates for different layers
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

### Popular Pre-trained Models

```python
# Vision
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
vgg19 = models.vgg19(pretrained=True)
densenet121 = models.densenet121(pretrained=True)
inception_v3 = models.inception_v3(pretrained=True)
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
efficientnet_b0 = models.efficientnet_b0(pretrained=True)
vit_b_16 = models.vit_b_16(pretrained=True)  # Vision Transformer

# Segmentation
fcn_resnet50 = models.segmentation.fcn_resnet50(pretrained=True)
deeplabv3_resnet50 = models.segmentation.deeplabv3_resnet50(pretrained=True)

# Object detection
faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
mask_rcnn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
```

## TorchVision

TorchVision is PyTorch's official library for computer vision. It provides datasets, pre-trained models, and utilities for manipulating images.

### Image Operations

Utility functions for visualizing and manipulating images:

```python
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF

# Functional operations
img = TF.resize(img, size=(224, 224))
img = TF.rotate(img, angle=30)
img = TF.hflip(img)
img = TF.adjust_brightness(img, brightness_factor=1.5)
img = TF.adjust_contrast(img, contrast_factor=1.5)

# Create an image grid
images = torch.randn(64, 3, 224, 224)
grid = make_grid(images, nrow=8, padding=2)
save_image(grid, 'grid.png')

# Save a batch of images
save_image(images, 'batch.png', nrow=8)
```

### Detection and Segmentation

```python
# Object detection
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = torch.randn(1, 3, 800, 800)
with torch.no_grad():
    predictions = model(image)
    # predictions[0]['boxes'] - box coordinates
    # predictions[0]['labels'] - object labels
    # predictions[0]['scores'] - confidence scores

# Semantic segmentation
model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

image = torch.randn(1, 3, 520, 520)
with torch.no_grad():
    output = model(image)['out']
    # output: (1, 21, 520, 520) - 21 PASCAL VOC classes
    predictions = torch.argmax(output, dim=1)
```

## TorchText

TorchText facilitates preprocessing and loading text data for NLP. It handles tokenization, vocabulary creation, and embeddings.

### Vocabulary and Tokenization

Tokenization divides text into units (words, subwords). The vocabulary maps these tokens to numerical indices:

```python
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

# Tokenizer
tokenizer = get_tokenizer('basic_english')

# Build vocabulary
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(train_data),
    specials=['<unk>', '<pad>', '<bos>', '<eos>'],
    min_freq=2
)
vocab.set_default_index(vocab['<unk>'])

# Convert text -> indices
text = "Hello world"
tokens = tokenizer(text)
indices = [vocab[token] for token in tokens]

# Padding
from torch.nn.utils.rnn import pad_sequence

sequences = [torch.tensor(vocab(tokenizer(text))) for text in texts]
padded = pad_sequence(sequences, batch_first=True, padding_value=vocab['<pad>'])
```

### Pre-trained Embeddings

```python
from torchtext.vocab import GloVe, FastText

# GloVe embeddings
glove = GloVe(name='6B', dim=100)

# Get word embedding
word_embedding = glove['hello']

# Create embedding matrix for your vocabulary
embedding_matrix = torch.zeros(len(vocab), 100)
for i, word in enumerate(vocab.get_itos()):
    if word in glove.stoi:
        embedding_matrix[i] = glove[word]

# Use in a model
embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
```

## TorchAudio

TorchAudio provides tools for loading, transforming, and augmenting audio data. It supports various audio formats and offers common transformations like spectrograms and MFCC.

### Audio Loading and Processing

Audio transformations are essential for preparing data for deep learning models:

```python
import torchaudio
import torchaudio.transforms as T

# Load audio file
waveform, sample_rate = torchaudio.load('audio.wav')

# Resampling
resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
waveform_resampled = resampler(waveform)

# Spectrogram
spectrogram = T.Spectrogram(
    n_fft=1024,
    win_length=None,
    hop_length=512
)
spec = spectrogram(waveform)

# Mel Spectrogram
mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=128
)
mel_spec = mel_spectrogram(waveform)

# MFCC
mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=40,
    melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23}
)
mfccs = mfcc_transform(waveform)

# Audio augmentation
time_stretch = T.TimeStretch()
waveform_stretched = time_stretch(spec, rate=1.2)

pitch_shift = T.PitchShift(sample_rate, n_steps=4)
waveform_shifted = pitch_shift(waveform)
```

## Distributed Training

Distributed training allows using multiple GPUs or multiple machines to accelerate training. PyTorch offers two main approaches: DataParallel (simple but limited) and DistributedDataParallel (recommended for performance).

### DataParallel (Simple)

The simplest method to use multiple GPUs, but with performance limitations:

```python
# Multi-GPU on a single machine
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

# The rest of the code remains identical
```

### DistributedDataParallel (Recommended)

DDP is more efficient than DataParallel because it uses one process per GPU and synchronizes gradients in an optimized way:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for data, targets in train_loader:
            data, targets = data.to(rank), targets.to(rank)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    cleanup()

# Launch training
import torch.multiprocessing as mp

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

## Quantization

Quantization reduces the precision of weights and activations (from float32 to int8) to decrease model size and accelerate inference, with minimal accuracy loss.

**Advantages:**
- Model size reduced by ~75%
- Inference 2-4x faster
- Less memory used

### Post-Training Quantization

The simplest method: quantize an already trained model without retraining:

```python
# Dynamic quantization (easy, fast)
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},  # Layers to quantize
    dtype=torch.qint8
)

# Static quantization (better accuracy)
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)

# Calibration with representative data
with torch.no_grad():
    for data, _ in calibration_loader:
        model_prepared(data)

model_quantized = torch.quantization.convert(model_prepared)

# Save
torch.save(model_quantized.state_dict(), 'model_quantized.pth')

# Size comparison
print(f"Original size: {os.path.getsize('model.pth') / 1e6:.2f} MB")
print(f"Quantized size: {os.path.getsize('model_quantized.pth') / 1e6:.2f} MB")
```

### Quantization-Aware Training

```python
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model)

# Train normally
for epoch in range(num_epochs):
    for data, targets in train_loader:
        optimizer.zero_grad()
        outputs = model_prepared(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Convert to quantized model
model_prepared.eval()
model_quantized = torch.quantization.convert(model_prepared)
```

## ONNX Export

ONNX (Open Neural Network Exchange) is a standard format for representing deep learning models. Exporting to ONNX allows using your PyTorch model in other frameworks or optimizing inference.

**Use cases:**
- Production deployment with ONNX Runtime
- Interoperability between frameworks (PyTorch → TensorFlow)
- Hardware-specific inference optimizations

### Export to ONNX

ONNX export traces the model with an example input:

```python
import torch.onnx

# Prepare model
model.eval()

# Example input
dummy_input = torch.randn(1, 3, 224, 224)

# Export
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Verify ONNX model
import onnx
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))
```

### Inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Create session
session = ort.InferenceSession('model.onnx')

# Prepare input
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

x = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Inference
result = session.run([output_name], {input_name: x})
print(result[0].shape)
```

## JIT and TorchScript

TorchScript allows creating serializable and optimizable models independent of Python. This is essential for production deployment, especially in non-Python environments (C++, mobile).

**Advantages:**
- Independent of Python
- JIT (Just-In-Time) optimizations
- Mobile deployment (iOS, Android)
- Improved inference performance

### TorchScript by Tracing

Tracing records operations performed during a forward pass. Simple but doesn't support dynamic control structures:

```python
# Trace the model
model.eval()
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save('model_traced.pt')

# Load
loaded_model = torch.jit.load('model_traced.pt')
output = loaded_model(example_input)
```

### TorchScript by Scripting

```python
# Script the model (supports control structures)
scripted_model = torch.jit.script(model)

# Save
scripted_model.save('model_scripted.pt')

# Load and use in C++
# torch::jit::script::Module module = torch::jit::load("model_scripted.pt");
```

### JIT Optimization

```python
import time

# Optimization for inference
with torch.jit.optimized_execution(True):
    traced_model = torch.jit.trace(model, example_input)
    traced_model = torch.jit.freeze(traced_model)

# Warm-up
for _ in range(10):
    _ = traced_model(example_input)

# Measure performance
start = time.time()
for _ in range(1000):
    with torch.no_grad():
        _ = traced_model(example_input)
print(f"Time: {(time.time() - start) * 1000:.2f} ms")
```

## Save and Load

### Save/Load Model

```python
# Save only weights
torch.save(model.state_dict(), 'model_weights.pth')

# Load weights
model = SimpleNet(784, 128, 10)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Save complete model
torch.save(model, 'model_complete.pth')

# Load complete model
model = torch.load('model_complete.pth')
model.eval()

# Save complete checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## Tips and Best Practices

### Performance

```python
# 1. Use DataLoader with num_workers
train_loader = DataLoader(dataset, batch_size=32, num_workers=4)

# 2. Use pin_memory for GPU
train_loader = DataLoader(dataset, batch_size=32, pin_memory=True)

# 3. Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data, targets in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(data)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 4. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Debugging

```python
# Check for NaN/Inf
torch.isnan(x).any()
torch.isinf(x).any()

# Fix seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Gradient checking
from torch.autograd import gradcheck

input = torch.randn(1, 3, requires_grad=True, dtype=torch.double)
test = gradcheck(model, input, eps=1e-6, atol=1e-4)
print(f'Gradient check: {test}')

# Display model architecture
print(model)
from torchsummary import summary
summary(model, input_size=(3, 224, 224))
```

### Hooks for Debugging

```python
# Hook on gradients
def gradient_hook(grad):
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient mean: {grad.mean()}, std: {grad.std()}")
    return grad

x = torch.randn(3, 3, requires_grad=True)
handle = x.register_hook(gradient_hook)

y = x.sum()
y.backward()

handle.remove()  # Remove hook

# Hook on layers
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.layer1.register_forward_hook(get_activation('layer1'))
model.layer2.register_forward_hook(get_activation('layer2'))

# After forward pass, activations contains outputs
output = model(x)
print(activations['layer1'].shape)
```

### Profiling

```python
from torch.profiler import profile, record_function, ProfilerActivity

# Simple profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        output = model(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Advanced profiler with TensorBoard
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/model'),
    record_shapes=True,
    with_stack=True
) as prof:
    for step, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        prof.step()
        if step >= 10:
            break
```

### Memory Profiling

```python
# Monitor memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Memory profiler
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    output = model(input)

print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=10
))

# Detect memory leaks
torch.cuda.reset_peak_memory_stats()
output = model(input)
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

## Advanced Techniques

These techniques help improve training stability, model generalization, and handle hardware constraints.

### Gradient Accumulation

Useful when your GPU doesn't have enough memory for a large batch. Accumulates gradients over multiple small batches before updating weights:

```python
# To simulate larger batch sizes
accumulation_steps = 4

optimizer.zero_grad()
for i, (data, targets) in enumerate(train_loader):
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps  # Normalize loss
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Gradient Clipping

Prevents gradient explosion by limiting their magnitude. Essential for training RNNs and Transformers:

```python
# Clipping by norm
max_grad_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# Clipping by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### Early Stopping

Stops training when validation performance no longer improves, avoiding overfitting:

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), 'best_model.pth')

# Usage
early_stopping = EarlyStopping(patience=10)
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
```

### Label Smoothing

Reduces model overconfidence by smoothing labels (0 and 1 become for example 0.1 and 0.9). Improves generalization:

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_pred = F.log_softmax(pred, dim=-1)
        
        # Create smoothed distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_pred, dim=-1))

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

### Focal Loss

Designed for class imbalance problems. It gives less weight to easy examples and focuses on hard examples:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

### Custom Learning Rate Warmup

Gradually increases learning rate at the beginning of training to stabilize early iterations, particularly useful for Transformers:

```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

# Usage
warmup = WarmupScheduler(optimizer, warmup_steps=1000, base_lr=0.001)

for epoch in range(num_epochs):
    for data, targets in train_loader:
        # ... training code ...
        warmup.step()
```

### Stochastic Weight Averaging (SWA)

Averages model weights over multiple epochs at the end of training. Improves generalization with minimal cost:

```python
from torch.optim.swa_utils import AveragedModel, SWALR

# Create averaged model
swa_model = AveragedModel(model)

# Scheduler for SWA
swa_scheduler = SWALR(optimizer, swa_lr=0.05)

# Normal training then SWA
swa_start = 75  # Start SWA at epoch 75

for epoch in range(num_epochs):
    train_epoch(model, train_loader, optimizer)
    
    if epoch > swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

# Update batch normalization
torch.optim.swa_utils.update_bn(train_loader, swa_model)

# Use swa_model for inference
```

### Model Ensemble

Combines multiple models to improve performance. The final prediction is typically the average of individual predictions:

```python
class ModelEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        predictions = [model(x) for model in self.models]
        # Average predictions
        return torch.mean(torch.stack(predictions), dim=0)

# Usage
model1 = ResNet50()
model2 = EfficientNet()
model3 = VGG16()

ensemble = ModelEnsemble([model1, model2, model3])
output = ensemble(x)
```

### Test-Time Augmentation (TTA)

Applies multiple random transformations to the test image and averages predictions. Improves robustness and accuracy at the cost of longer inference time:

```python
def test_time_augmentation(model, image, transforms, num_augmentations=5):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_augmentations):
            augmented = transforms(image)
            pred = model(augmented)
            predictions.append(pred)
        
        # Average predictions
        final_pred = torch.mean(torch.stack(predictions), dim=0)
    
    return final_pred

# Usage
tta_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1)
])

prediction = test_time_augmentation(model, image, tta_transforms)
```


