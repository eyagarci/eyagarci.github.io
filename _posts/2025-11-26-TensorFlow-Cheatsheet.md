---
title: "TensorFlow Cheatsheet - Practical Guide"
date: 2025-11-25 16:00:00
categories: [machine learning, deep learning]
tags: [TensorFlow, machine learning, deep learning]    
image:
  path: /assets/imgs/headers/tensorflow.png
---

## Introduction

TensorFlow is an open-source library developed by Google for machine learning and deep learning. It provides a high-level API (Keras) to build and train neural networks, as well as tools for production and model deployment.

## Installation

```bash
# CPU version
pip install tensorflow

# GPU version (requires CUDA)
pip install tensorflow[and-cuda]

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Basic Imports

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
```

## Tensors

Tensors are the fundamental data structures of TensorFlow. Similar to NumPy arrays, they represent multidimensional arrays that can be used on CPU or GPU. TensorFlow uses tensors for all computational operations.

### Creating Tensors

```python
# From Python lists
tf.constant([1, 2, 3, 4])

# From numpy arrays
tf.constant(np.array([1, 2, 3, 4]))

# Zeros and ones
tf.zeros((3, 4))
tf.ones((2, 3))

# Random tensors
tf.random.normal((3, 3), mean=0.0, stddev=1.0)
tf.random.uniform((2, 2), minval=0, maxval=10)

# Range
tf.range(start=0, limit=10, delta=2)

# Identity matrix
tf.eye(3)
```

### Tensor Properties

```python
x = tf.constant([[1, 2], [3, 4]])

x.shape          # Shape of tensor
x.dtype          # Data type
x.ndim           # Number of dimensions
tf.size(x)       # Total number of elements
```

### Type Conversion

```python
# Cast to different dtype
x = tf.constant([1.5, 2.7, 3.9])
tf.cast(x, dtype=tf.int32)

# Convert to numpy
x.numpy()
```

## Tensor Operations

TensorFlow provides a wide range of mathematical operations to manipulate tensors. These operations are optimized for parallel computing and can be executed on GPU for better performance.

### Basic Math

```python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# Element-wise operations
tf.add(a, b)           # a + b
tf.subtract(a, b)      # a - b
tf.multiply(a, b)      # a * b
tf.divide(a, b)        # a / b
tf.pow(a, 2)           # a ** 2

# Matrix operations
x = tf.constant([[1, 2], [3, 4]])
y = tf.constant([[5, 6], [7, 8]])

tf.matmul(x, y)        # Matrix multiplication
tf.transpose(x)        # Transpose
```

### Reduction Operations

```python
x = tf.constant([[1, 2, 3], [4, 5, 6]])

tf.reduce_sum(x)              # Sum all elements
tf.reduce_sum(x, axis=0)      # Sum along axis 0
tf.reduce_sum(x, axis=1)      # Sum along axis 1

tf.reduce_mean(x)             # Mean
tf.reduce_max(x)              # Maximum
tf.reduce_min(x)              # Minimum
tf.argmax(x, axis=1)          # Index of max value
tf.argmin(x, axis=1)          # Index of min value
```

### Reshaping

```python
x = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])

tf.reshape(x, (4, 2))         # Reshape to (4, 2)
tf.reshape(x, (-1,))          # Flatten
tf.expand_dims(x, axis=0)     # Add dimension
tf.squeeze(x)                 # Remove dimensions of size 1
```

### Indexing and Slicing

```python
x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

x[0]                          # First row
x[:, 0]                       # First column
x[0:2, 1:3]                   # Slice
tf.gather(x, [0, 2])          # Gather specific rows
tf.boolean_mask(x, [True, False, True])  # Boolean indexing
```

### Concatenation and Stacking

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

tf.concat([a, b], axis=0)     # Concatenate along axis 0
tf.concat([a, b], axis=1)     # Concatenate along axis 1
tf.stack([a, b], axis=0)      # Stack tensors
```

## Neural Network Layers

Keras offers a vast collection of pre-built layers to assemble neural networks. Each layer performs a specific transformation on input data, allowing complex architectures to be created in a modular way.

### Dense (Fully Connected)

Dense layers are fully connected layers where each neuron is connected to all neurons in the previous layer.

```python
layer = layers.Dense(units=64, activation='relu', use_bias=True)
```

### Convolutional Layers

Convolutional layers are essential for image processing. They apply filters to detect spatial features like edges, textures, and patterns.

```python
# 2D Convolution
layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')

# 1D Convolution
layers.Conv1D(filters=64, kernel_size=3, activation='relu')

# Transposed Convolution
layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')
```

### Pooling Layers

```python
layers.MaxPooling2D(pool_size=2, strides=2)
layers.AveragePooling2D(pool_size=2)
layers.GlobalMaxPooling2D()
layers.GlobalAveragePooling2D()
```

### Recurrent Layers

Recurrent layers (RNN, LSTM, GRU) are designed to process sequential data like text, time series, or audio. They maintain an internal memory to capture temporal dependencies.

```python
layers.LSTM(units=128, return_sequences=True, return_state=True)
layers.GRU(units=64, return_sequences=False)
layers.SimpleRNN(units=32)
layers.Bidirectional(layers.LSTM(64))
```

### Normalization

```python
layers.BatchNormalization()
layers.LayerNormalization()
```

### Dropout and Regularization

```python
layers.Dropout(rate=0.5)
layers.SpatialDropout2D(rate=0.2)

# L1/L2 regularization in layers
layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.01))
```

### Other Useful Layers

```python
layers.Flatten()
layers.Reshape((28, 28, 1))
layers.Embedding(input_dim=10000, output_dim=128)
layers.Attention()
layers.MultiHeadAttention(num_heads=8, key_dim=64)
```

## Building Models

Keras offers three approaches to building models: the Sequential API (for simple linear models), the Functional API (for more complex architectures with multiple inputs/outputs), and Subclassing (for full control and maximum customization).

### Sequential API

The Sequential API is the simplest, ideal for stacking layers linearly.

```python
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Functional API

The Functional API allows creating models with complex topologies: non-linear graphs, residual connections, multiple inputs/outputs.

```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)
```

### Subclassing API

Subclassing offers maximum flexibility by allowing custom behaviors to be defined in the `call()` method.

```python
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

model = MyModel()
```

## Model Compilation

Before training, the model must be compiled by specifying the optimizer (algorithm for updating weights), the loss function (measures error), and metrics (to track performance).

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# With custom learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)
```

## Common Optimizers

Optimizers adjust network weights to minimize the loss function. Each optimizer uses a different strategy to update parameters.

```python
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
```

## Loss Functions

```python
# Classification
keras.losses.BinaryCrossentropy()
keras.losses.CategoricalCrossentropy()
keras.losses.SparseCategoricalCrossentropy()

# Regression
keras.losses.MeanSquaredError()
keras.losses.MeanAbsoluteError()
keras.losses.Huber()
```

## Metrics

```python
keras.metrics.Accuracy()
keras.metrics.Precision()
keras.metrics.Recall()
keras.metrics.AUC()
keras.metrics.MeanSquaredError()
keras.metrics.MeanAbsoluteError()
```

## Training

Training involves presenting data to the model iteratively so it learns to make accurate predictions. TensorFlow automatically handles forward propagation, gradient computation, and weight updates.

### Basic Training

```python
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val),
    verbose=1
)
```

### With Callbacks

Callbacks allow monitoring and controlling training: saving the best models, stopping training early, adjusting the learning rate, or logging.

```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7
    ),
    keras.callbacks.TensorBoard(
        log_dir='./logs'
    )
]

history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks
)
```

### Using tf.data

The `tf.data` API creates efficient input pipelines with parallel loading, caching, and data prefetching to optimize GPU usage.

```python
# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.AUTOTUNE)

# Train with dataset
model.fit(dataset, epochs=10)
```

## Evaluation and Prediction

```python
# Evaluate
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

# Predict
predictions = model.predict(x_test)

# Single prediction
single_prediction = model.predict(x_test[0:1])
```

## Saving and Loading Models

```python
# Save entire model
model.save('my_model.h5')
model.save('my_model.keras')
model.save('my_model/')  # SavedModel format

# Load model
loaded_model = keras.models.load_model('my_model.h5')

# Save/load weights only
model.save_weights('model_weights.h5')
model.load_weights('model_weights.h5')

# Save/load architecture only
json_config = model.to_json()
loaded_model = keras.models.model_from_json(json_config)
```

## Custom Training Loop

For full control over training, you can create a custom loop using `GradientTape` to manually compute gradients and apply updates.

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(y, predictions)

# Training loop
for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)
    
    print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}')
```

## Image Preprocessing

Image preprocessing is crucial for improving model performance. Data augmentation (rotation, zoom, flip) helps create a more robust model and avoid overfitting.

```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load single image
img = image.load_img('path/to/image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rescale=1./255
)

train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

## Transfer Learning

Transfer learning reuses models pre-trained on large databases (like ImageNet) to solve new problems with less data and training time.

```python
# Load pre-trained model
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Build new model
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)
```

## Popular Pre-trained Models

```python
# Image classification
keras.applications.VGG16(weights='imagenet')
keras.applications.ResNet50(weights='imagenet')
keras.applications.MobileNetV2(weights='imagenet')
keras.applications.EfficientNetB0(weights='imagenet')
keras.applications.InceptionV3(weights='imagenet')

# With custom top
base_model = keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
```

## GradientTape for Custom Gradients

`GradientTape` records operations to automatically compute derivatives. It's essential for implementing custom optimization algorithms or complex architectures.

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

# Compute gradient
dy_dx = tape.gradient(y, x)
print(dy_dx)  # 6.0

# Multiple variables
with tf.GradientTape() as tape:
    y = x1 ** 2 + x2 ** 2

gradients = tape.gradient(y, [x1, x2])
```

## GPU Configuration

Properly configuring the GPU is essential to maximize performance. TensorFlow can automatically manage GPU memory and distribute computations across multiple GPUs.

```python
# List physical devices
gpus = tf.config.list_physical_devices('GPU')

# Set memory growth
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set visible devices
tf.config.set_visible_devices(gpus[0], 'GPU')

# Mixed precision training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

## Distributed Training

Distributed training allows using multiple GPUs or machines to accelerate training of large models. `MirroredStrategy` synchronizes models across all available GPUs.

```python
# Multi-GPU training with MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

model.fit(train_dataset, epochs=10)
```

## Useful Utilities

```python
# One-hot encoding
tf.one_hot(indices=[0, 1, 2], depth=3)

# Normalization
layer = layers.Normalization()
layer.adapt(data)  # Compute mean and variance

# Text vectorization
vectorize_layer = layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=100
)
vectorize_layer.adapt(text_data)

# Learning rate schedules
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.96
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

## Debugging

```python
# Enable eager execution (default in TF 2.x)
tf.config.run_functions_eagerly(True)

# Print tensor values
tf.print(tensor)

# Check for NaN/Inf
tf.debugging.assert_all_finite(tensor, 'Tensor contains NaN or Inf')

# Model summary
model.summary()

# Visualize model
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
```

## Best Practices

1. **Use tf.data for input pipelines**: Better performance with prefetching and parallel processing
2. **Enable mixed precision**: Faster training on modern GPUs with minimal accuracy loss
3. **Use @tf.function decorator**: Convert Python functions to TensorFlow graphs for better performance
4. **Normalize inputs**: Scale features to similar ranges (0-1 or standardize)
5. **Use callbacks**: Monitor training, save best models, reduce learning rate
6. **Start with pre-trained models**: Transfer learning for faster convergence
7. **Set random seeds**: For reproducibility
   ```python
   tf.random.set_seed(42)
   np.random.seed(42)
   ```

## Common Activation Functions

```python
layers.Dense(64, activation='relu')
layers.Dense(64, activation='sigmoid')
layers.Dense(64, activation='tanh')
layers.Dense(64, activation='softmax')
layers.Dense(64, activation='elu')
layers.Dense(64, activation='selu')
layers.Dense(64, activation='swish')
```

