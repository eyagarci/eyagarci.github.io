---
title: "Deep Learning Guide"
date:   2025-11-22 19:00:00
categories: [Machine Learning]
tags: [Deep-Learning, Neural-Networks, Machine-Learning, AI]    
image:
  path: /assets/imgs/headers/deeplearning.jpg
---


## Introduction 

### What is a Neural Network?

A neural network is a system that learns from examples, inspired by the human brain. It is composed of **layers of neurons** that progressively transform data.

**Basic structure:**
- **Input layer**: receives data (images, text, numbers)
- **Hidden layers**: extract increasingly complex features
- **Output layer**: produces the final prediction

### How a Network Learns

Learning happens in 4 repeated steps:

1. **Forward Propagation**: data passes through the network
2. **Error calculation**: we measure how wrong the prediction is
3. **Backward Propagation**: we calculate how to adjust each neuron
4. **Update**: we modify the weights to reduce error

This cycle repeats thousands of times until the model becomes accurate.

### The Two Major Problems

**Underfitting**
- The model is **too simple**
- Doesn't capture patterns in the data
- Poor performance everywhere (train AND validation)
- **Solution**: make the model more complex

**Overfitting**
- The model **memorizes** instead of understanding
- Excellent performance on training data
- Poor performance on new data
- **Solution**: regularization, more data

---

## Sequence Models - Architecture and Usage

### What is a Sequence Model?

**Beginner definition:**
A Sequence Model is a special type of neural network that understands that **the order of data is important**. Unlike classic networks that look at each data point in isolation, Sequence Models "remember" what they've seen before.

**Why is order important?**

Let's take simple examples:

**Example 1 - Text:**
- "The dog eats the apple" ✅ (makes sense)
- "Apple the eats dog the" ❌ (same words, but no sense)

→ Word order completely changes the meaning!

**Example 2 - Video:**
- Image 1: person standing
- Image 2: person bent
- Image 3: person on ground
- → Sequence = someone falling

If we shuffle the order: person on ground → standing → bent (weird!)

**Example 3 - Weather:**
- Day 1: 20°C
- Day 2: 22°C
- Day 3: 24°C
- → Trend: it's warming up

To predict Day 4, you need to know this trend (previous days).

---

### Types of data for Sequence Models

**Sequence Models** handle data that has an **order** or **temporal dependency**:

**1. Text:**
- Words follow in a specific order
- "I love you" ≠ "you love I"
- Example: translation, text generation, chatbots

**2. Time series:**
- Data that evolves over time
- Stock prices, temperature, sales numbers
- The past influences the future

**3. Audio:**
- Sounds that follow each other
- Speech recognition (Siri, Alexa)
- A word depends on previous sounds

**4. Video:**
- Sequence of images
- Action detection (someone running, jumping)
- A single image is not enough

---

### Fundamental difference

**Classic networks (no memory):**
```
Input 1 → Network → Output 1
Input 2 → Network → Output 2
Input 3 → Network → Output 3
```
→ Each prediction is **independent**
→ The network doesn't remember Input 1 when processing Input 2

**Sequence Models (with memory):**
```
Input 1 → Network → Output 1
            ↓ (memory)
Input 2 → Network → Output 2
            ↓ (updated memory)
Input 3 → Network → Output 3
```
→ Each prediction uses **history**
→ The network "remembers" what it saw before

### Types of Sequence Models

#### 1. RNN (Recurrent Neural Networks)

**Simple definition:**
RNN is the most basic Sequence Model. It reads data one by one (like reading a book word by word) and maintains a "memory" of what it saw before.

**How it works - Concrete example:**

Imagine you're reading the sentence: "The cat eats the mouse"

**Step 1:** Reads "The"
- Memory: ["The"]
- Understands: a definite article

**Step 2:** Reads "cat" + remembers "The"
- Memory: ["The", "cat"]
- Understands: we're talking about a specific cat

**Step 3:** Reads "eats" + remembers "The cat"
- Memory: ["The", "cat", "eats"]
- Understands: the cat is performing an action

**Step 4:** Reads "the" + remembers the context
- Memory: ["The", "cat", "eats", "the"]
- Understands: a second thing will be mentioned

**Step 5:** Reads "mouse" + all the context
- Complete memory: ["The", "cat", "eats", "the", "mouse"]
- Understands: the cat eats the mouse (complete meaning!)

**The secret of RNN:**
At each step, it **combines**:
1. The new word it just read
2. Its memory of everything it read before

**Detailed analogy:** 
It's like reading a detective novel:
- Page 1: you discover the crime
- Page 50: you discover a suspect, but you remember the crime
- Page 100: a new clue, but you remember everything you've read
- Final page: you understand the whole story thanks to your memory

**Strengths:**
- Simple to understand
- Can process sequences of any length
- Shares knowledge across all steps

**Weaknesses:**
- **Forgets** on long sequences (loses memory)
- Difficulty connecting distant information in text
- Slow to train (must process sequentially)

**When to use it:**
- Short sequences (less than 50 elements)
- Simple tasks
- Predicting the next word/number in a short sequence

---

#### 2. LSTM (Long Short-Term Memory)

**Simple definition:**
LSTM is a much smarter version of RNN. Instead of simple memory, it has a sophisticated system of "gates" that decide what to keep, what to forget, and what to use.

**The problem with simple RNN:**
Imagine reading a 300-page book. By page 300, you've forgotten the details from page 1!

→ RNN has the same problem: after 50-100 elements, it forgets the beginning.

**How LSTM solves this problem:**

LSTM has 3 "smart gates" (like guards that control information):

**1. Forget Gate:**
```
Question: "Is this information still useful?"

Example: "Mary loves cats. She also adores dogs. Mary..."
→ KEEP "Mary" (important for the rest)
→ FORGET "loves cats" (less important now)
```

**2. Input Gate:**
```
Question: "Is this new information important to remember?"

Example: "The president resigned yesterday."
→ VERY important! We keep it in memory

Example: "It was nice weather."
→ Less important, we can forget it
```

**3. Output Gate:**
```
Question: "What information do I need RIGHT NOW?"

Example: "Mary and Peter arrived. Mary said..."
→ At this moment, we need to remember that Mary is a person
→ We extract this information from memory
```

**Detailed analogy with a library:**

Imagine your brain as a library with a librarian (the LSTM):

**Forget Gate:**
- Librarian: "This book about dinosaurs hasn't been used in 5 years"
- Action: Remove it from the main shelves

**Input Gate:**
- Librarian: "This new book about AI is in high demand"
- Action: Put it prominently on the main shelf

**Output Gate:**
- You: "I need information about Python"
- Librarian: Searches in memory and pulls out the right book

**Concrete example - Translation:**

```
English: "The cat that was sitting on the mat is black"
French: "Le chat qui était assis sur le tapis est noir"

Step 1: Reads "The cat"
→ KEEP in memory: "cat" (main subject)

Step 2-6: Reads "that was sitting on the mat"
→ UNDERSTANDS: description of the cat
→ KEEPS: the information that it's still the same cat

Step 7: Reads "is black"
→ USES memory: remembers that "black" describes "the cat" from the beginning
→ Correctly translates: "Le chat... est noir" (gender agreement)
```

**Why it's better than RNN:**
- RNN: forgets "The cat" after 5-10 words
- LSTM: remembers "The cat" even after 50-100 words

**Strengths:**
- **Long-term memory**: remembers over hundreds of elements
- Solves RNN's forgetting problem
- Automatically learns what is important

**Weaknesses:**
- More complex than RNN
- Slower to train
- Requires more data

**When to use it:**
- Long texts (articles, books)
- Time series with long-term trends
- Sentiment analysis over multiple sentences
- Financial forecasting
- Speech recognition

---

#### 3. GRU (Gated Recurrent Unit)

**Simple definition:**
GRU is a simplified and faster version of LSTM. Instead of 3 gates, it has only 2, but it works almost as well.

**The idea behind GRU:**
The creators of GRU said: "LSTM works well, but it's complicated. Can we make it simpler?"

**Result:** Merge some of LSTM's gates

**The 2 gates of GRU:**

**1. Update Gate:**
```
Question: "How much old information to keep VS new to add?"

Example: "Peter likes soccer. Now, he prefers basketball."

→ Old info: "Peter likes soccer" (less important)
→ New info: "Peter prefers basketball" (more important)
→ The gate decides: keep 20% old, 80% new
```

**2. Reset Gate:**
```
Question: "Should I completely ignore the past for this new info?"

Example: "Mary talks about her cat. Subject change: the weather is nice."

→ The gate decides: RESET! New subject, we start from zero
```

**Comparison LSTM vs GRU:**

```
LSTM (3 gates):
├─ Forget gate: what to erase?
├─ Input gate: what to add?
└─ Output gate: what to use?
   → More powerful but slower

GRU (2 gates):
├─ Update gate: how much old/new?
└─ Reset gate: start from zero?
   → Simpler and faster
```

**Analogy - Updating a CV:**

**LSTM (3 decisions):**
1. What to remove from the old CV? (Forget gate)
2. What to add new? (Input gate)
3. What to show the recruiter? (Output gate)

**GRU (2 decisions):**
1. How much to keep old VS new? (Update gate)
2. Redo the CV completely? (Reset gate)

**When to choose GRU over LSTM:**

✅ **Use GRU if:**
- You have a not huge dataset (< 100k examples)
- You want to train faster
- You have memory constraints
- You're a beginner (simpler to understand)

✅ **Use LSTM if:**
- You have a lot of data (> 1M)
- The task is very complex
- You have time and resources
- You want to extract maximum performance

**Strengths:**
- Faster than LSTM
- Fewer parameters = less risk of overfitting
- Performance often similar to LSTM

**Weaknesses:**
- Slightly less powerful than LSTM on very complex tasks

**When to use it:**
- Alternative to LSTM when resources are limited
- Medium-sized dataset
- Need for training speed

---

#### 4. Transformers

**Simple definition:**
Transformer is a revolution in Sequence Models. Instead of reading word by word (RNN, LSTM), it looks at **THE WHOLE sentence at once** and decides which words are important to each other.

**The problem with RNN/LSTM:**

Imagine reading this sentence word by word:
```
"The cat that I saw yesterday in my neighbor's garden eats a mouse"
```

**With RNN/LSTM:**
- Reads "The"
- Reads "cat" (remembers "The")
- Reads "that"
- Reads "I"...
- ...(many words)...
- Reads "eats"

→ Problem: when we get to "eats", we've almost forgotten "The cat"!
→ It's slow: we must read in order, word by word

**With Transformer:**
- Looks at ALL words at once
- Immediately identifies: "The cat" is the subject, "eats" is the action
- Instantly connects "The cat" and "eats" even if they're far apart

---

**The Attention Mechanism Explained Simply**

**Analogy - Classroom:**

Imagine a class with 10 students. The teacher asks a question:

**Without attention (RNN):**
- The teacher asks each student one by one
- Each student answers without listening to others
- Slow and sequential

**With attention (Transformer):**
- The teacher asks the question to EVERYONE at once
- Each student can look at and listen to ALL the others
- Student 5 can say "I agree with student 2"
- Fast and parallel

---

**Concrete Example - Translation**

French sentence: "Le chat noir mange la souris grise"
English translation: "The black cat eats the grey mouse"

**Transformer pays attention to:**

**To translate "black":**
```
Looks at: "noir"
Pays attention to: "chat" (to know that "noir" describes the cat)
Translation: places "black" after "cat" → "black cat"
```

**To translate "grey":**
```
Looks at: "grise"
Pays attention to: "souris" (to know that "grise" describes the mouse)
Translation: places "grey" after "mouse" → "grey mouse"
```

**Attention matrix (who looks at who):**
```
           Le  chat  noir  eats  the  mouse grey
Le        [++   +     -     -     -    -     -  ]
chat      [ +  ++     +     +     -    -     -  ]
noir      [ -   ++   ++     -     -    -     -  ]
eats      [ +   +     -    ++     -    +     -  ]
the       [ -   -     -     +    ++    +     -  ]
mouse     [ -   -     -     +     +   ++     +  ]
grey      [ -   -     -     -     -   ++    ++  ]
```

++ = pays a lot of attention
+  = pays a little attention
-  = ignores

**Reading:**
- "noir" pays a lot of attention to "chat" (it describes it)
- "grise" pays a lot of attention to "souris" (it describes it)
- "mange" pays attention to "chat" (subject) and "souris" (object)

---

**Why it's revolutionary:**

**1. Parallelization:**
```
RNN/LSTM: Word1 → Word2 → Word3 → Word4 (sequential, slow)
Transformer: All words processed AT THE SAME TIME (parallel, fast)
```

**2. No distance limit:**
```
Sentence: "The man that I saw yesterday who wore a red hat eats"

RNN/LSTM: difficult to connect "The man" and "eats" (too far)
Transformer: easily connects "The man" and "eats" (direct attention)
```

**3. Multi-head Attention (multiple heads):**

The Transformer doesn't pay attention just ONCE, it does it SEVERAL times in parallel:

```
Head 1: looks at grammar (subject, verb, object)
Head 2: looks at adjectives and their nouns
Head 3: looks at temporal relations
Head 4: looks at general context
...
```

It's like having 8-12 experts analyzing the sentence differently, then combining their opinions.

---

**Famous applications of Transformers:**

**GPT (Generative Pre-trained Transformer):**
- ChatGPT, GPT-4
- Generates text like a human

**BERT (Bidirectional Encoder Representations from Transformers):**
- Google Search
- Understands search context

**T5, BART:**
- Translation
- Text summarization

**DALL-E, Stable Diffusion:**
- Image generation from text
- Use adapted Transformers

**Strengths:**
- **Parallelization**: ultra-fast training (all words at once)
- Easily captures relationships between distant words
- Architecture used in GPT, BERT, ChatGPT
- Best performance on almost all tasks

**Weaknesses:**
- Consumes a lot of memory for very long sequences
- Requires enormous data and computing power
- More complex to understand and implement

**When to use it:**
- Machine translation (Google Translate, DeepL)
- Chatbots and assistants (ChatGPT, Claude)
- Long document analysis
- Text generation
- Automatic Question-Answering

---

### Quick Comparison of Sequence Models

| Model | Memory | Speed | Complexity | Best for |
|-------|--------|-------|------------|----------|
| **RNN** | Short | Slow | Simple | Short sequences |
| **LSTM** | Long | Slow | Medium | Long sequences |
| **GRU** | Long | Medium | Medium | Speed/performance compromise |
| **Transformer** | Very long | Fast | High | Modern NLP, large datasets |

---

## Activation Functions

### What is an Activation Function?

An activation function decides whether a neuron should "activate" or not. It transforms the signal that the neuron receives.

**Why it's important:**
Without activation, the network would just be a series of multiplications (linear). Activations add **non-linearity**, allowing learning of complex patterns.

---

### Main Activation Functions

#### ReLU (Rectified Linear Unit)

**Simple definition:**
ReLU is like a filter that only lets through positive values. It's the most used activation function in deep learning.

**Principle:** 
- If the signal is positive → keep it as is
- If the signal is negative → set it to zero

**Advantages:**
- Extremely fast to compute
- Works well in 90% of cases
- Simple and effective

**Disadvantages:**
- "Dead neurons": some neurons can get stuck at zero permanently

**When to use:** 
- **Default choice** for hidden layers
- Suitable for almost all types of networks

**Example of use:**
- Input signal: [2, -1, 0.5, -3]
- After ReLU: [2, 0, 0.5, 0]

---

#### Leaky ReLU

**Simple definition:**
Improved version of ReLU that lets through a very small value even for negative numbers, preventing neurons from "dying".

**Principle:**
- If positive signal → keep
- If negative signal → keep a **small value** (instead of complete zero)

**Advantages:**
- Avoids dead neurons
- Slightly better than ReLU in some cases

**When to use:**
- If you notice many stuck neurons with ReLU
- Safe alternative to ReLU

---

#### Sigmoid

**Simple definition:**
Sigmoid transforms any number into a value between 0 and 1, which is perfect for representing a probability (e.g., 0.8 = 80% chance).

**Principle:**
Transforms any value into a number **between 0 and 1**

**Advantages:**
- Interpretable as a **probability**
- Output between 0 and 1 (perfect for "yes/no")

**Disadvantages:**
- In deep networks, the gradient becomes very weak
- Slows down learning

**When to use:**
- **Output layer** for binary classification (true/false, spam/not-spam)
- **DO NOT use** in hidden layers

**Example:**
- Very positive signal (5) → close to 1 (very confident "YES")
- Signal close to 0 → around 0.5 (uncertain)
- Very negative signal (-5) → close to 0 (very confident "NO")

---

#### Tanh

**Simple definition:**
Like Sigmoid but with values between -1 and 1 instead of 0 and 1. It's better for internal network layers because it's centered on zero.

**Principle:**
Transforms any value into a number **between -1 and 1**

**Advantages:**
- Centered on zero (better than sigmoid for internal layers)
- Output between -1 and 1

**Disadvantages:**
- Same weak gradient problem as sigmoid

**When to use:**
- Some RNN and LSTM (in the gates)
- Alternative to sigmoid, but less common now

---

#### Softmax

**Simple definition:**
Softmax takes multiple scores and transforms them into probabilities that total 100%. Useful when you need to choose ONE option among several (cat OR dog OR bird).

**Principle:**
Transforms a vector of numbers into **probabilities that sum to 1**

**How it works:**
- Takes multiple values
- Transforms them into probabilities
- The largest value gets the largest probability

**When to use:**
- **Output layer** for multi-class classification
- When you want to choose among multiple options

**Concrete example:**
```
Input: [2.0, 1.0, 0.1]  (raw scores)
Softmax: [0.66, 0.24, 0.10]  (probabilities)

Interpretation:
- 66% chance it's class 1 (cat)
- 24% chance it's class 2 (dog)
- 10% chance it's class 3 (bird)
```

---

#### GELU (for Transformers)

**Simple definition:**
GELU is a modern activation specially designed for Transformers (like ChatGPT). It's a more sophisticated version of ReLU.

**Principle:**
Smoother and more complex version of ReLU

**When to use:**
- **Modern Transformers** (BERT, GPT)
- State-of-the-art NLP
- Better performance than ReLU on these architectures

---

#### Swish

**Simple definition:**
Swish is an activation automatically discovered by Google's artificial intelligence. It works particularly well for image recognition.

**Principle:**
Smooth curve that works well in practice

**When to use:**
- Modern alternative to ReLU
- Computer vision
- Automatically found by Google

---

### Summary Table of Activation Functions

| Function | Output Range | Main Usage | Speed | Notes |
|----------|--------------|------------|-------|-------|
| **ReLU** | 0 to ∞ | Hidden layers | ⚡⚡⚡ | Default choice |
| **Leaky ReLU** | -∞ to ∞ | Hidden layers | ⚡⚡⚡ | Alternative to ReLU |
| **Sigmoid** | 0 to 1 | Binary output | ⚡⚡ | Only in output |
| **Tanh** | -1 to 1 | RNN/LSTM | ⚡⚡ | Less common |
| **Softmax** | 0 to 1 (sum=1) | Multi-class output | ⚡ | Classification |
| **GELU** | -∞ to ∞ | Transformers | ⚡ | Modern NLP |
| **Swish** | -∞ to ∞ | Vision | ⚡⚡ | Recent alternative |

**Quick selection guide:**
- **Hidden layers?** → ReLU
- **Binary classification in output?** → Sigmoid
- **Multi-class classification in output?** → Softmax
- **Transformers?** → GELU
- **Problems with ReLU?** → Leaky ReLU

---

## Hyperparameters 

### What is a Hyperparameter?

**Hyperparameters** are settings that **you choose** before training. The model doesn't learn them automatically - it's up to you to define them.

**Important difference:**
- **Parameters**: the model learns them (weights of connections between neurons)
- **Hyperparameters**: you choose them (number of layers, learning rate...)

---

### Network Architecture

#### Number of Layers

**Impact:**
- **Few layers** (1-2): simple model, learns basic patterns
- **Many layers** (10-100): complex model, learns abstractions

**How to choose:**
- Simple problem (XOR, addition) → 1-2 layers
- Images (classification) → 20-100 layers
- Complex NLP (translation) → 12-96 layers

**Practical rule:**
Start small (2-3 layers), increase if performance is insufficient.

---

#### Number of Neurons per Layer

**Impact:**
- **Few neurons** (32-64): fast but limited
- **Many neurons** (512-2048): powerful but slow

**Common patterns:**

**Funnel architecture** (recommended):
```
512 → 256 → 128 → 64
```
Progressively reduces, extracts increasingly abstract features

**Constant architecture**:
```
256 → 256 → 256
```
Maintains the same capacity at each level

**How to choose:**
- Small dataset → fewer neurons (avoids overfitting)
- Large dataset → more neurons (can learn more)
- Start with 128-256, adjust based on results

---

### Training Hyperparameters

#### Learning Rate - THE MOST IMPORTANT

**What is it:**
Controls the **step size** that the model takes to improve.

**Analogy:**
Descending a mountain in fog:
- **High learning rate**: big jumps → risk of jumping over the minimum
- **Low learning rate**: small steps → arrives slowly but surely
- **Optimal learning rate**: balance between speed and precision

**Typical values:**
- **0.1**: very high, risk of divergence
- **0.01**: high, for SGD
- **0.001**: **standard for Adam** (recommended starting point)
- **0.0001**: low, for fine-tuning
- **0.00001**: very low, very slow convergence

**Symptoms and solutions:**

**Too high:**
- Loss oscillates violently
- Loss becomes NaN (Not a Number)
- The model never improves
- **Solution**: divide by 10

**Too low:**
- Loss decreases very slowly
- Training takes hours/days
- Stuck in a local minimum
- **Solution**: multiply by 2-5

---

#### Batch Size

**What is it:**
Number of examples processed **simultaneously** before updating weights.

**Impact:**

**Small batch (8-32):**
- ✅ Better generalization
- ✅ Less memory required
- ❌ Slower training
- ❌ Noisy gradient (less stable)

**Large batch (128-512):**
- ✅ Faster training
- ✅ More stable gradient
- ✅ Better GPU utilization
- ❌ Requires a lot of memory
- ❌ Risk of poorer generalization

**How to choose:**
- **8GB GPU**: batch 32-64
- **16GB GPU**: batch 64-128
- **32GB+ GPU**: batch 128-512

**Advice**: use the largest batch your GPU can support.

---

#### Number of Epochs

**What is it:**
One **epoch** = one complete pass over all training data.

**How to choose:**
- **Small dataset** (<10k examples): 100-500 epochs
- **Medium dataset** (10k-100k): 50-200 epochs
- **Large dataset** (>1M): 10-50 epochs

**Important**: use **early stopping** which automatically stops when the model stagnates (see Regularization section).

---

### Optimizers - How the Model Learns

#### SGD (Stochastic Gradient Descent)

**Simple definition:**
SGD is the most basic learning algorithm. It adjusts the network's weights little by little by following the direction that reduces error. It's like descending a hill step by step.

**Principle:**
The basic algorithm - descends in the direction that reduces error.

**Advantages:**
- Simple to understand
- Well-studied theoretically

**Disadvantages:**
- Slow to converge
- Very sensitive to learning rate

**When to use it:**
- Rarely in practice (mainly for teaching)

---

#### SGD + Momentum

**Simple definition:**
Momentum adds inertia to SGD, like a rolling ball that picks up speed. This allows faster learning and avoids getting stuck.

**Principle:**
Accumulates directions of previous steps (like a rolling ball picking up speed).

**Advantages:**
- Converges faster than simple SGD
- Fewer oscillations
- Can escape certain local minima

**When to use it:**
- Computer vision (some famous models use it)
- When you have time to properly tune the learning rate
- Searching for the best possible generalization

---

#### Adam (Adaptive Moment Estimation) - THE DEFAULT CHOICE

**Simple definition:**
Adam is the most popular and easiest to use optimizer. It automatically adjusts the learning speed for each network parameter, making it very effective without too much configuration.

**Principle:**
Automatically adapts the learning rate for each parameter. Combines several smart techniques.

**Advantages:**
- **Works well "out of the box"** with little tuning
- Not very sensitive to initial learning rate choice
- Converges quickly
- Used in 90% of cases

**Disadvantages:**
- Can sometimes generalize less well than SGD+Momentum

**When to use it:**
- **Default choice** to start
- NLP, Transformers
- When you want quick results
- Most situations

**Recommended configuration:**
- Learning rate: 0.001 (standard value)

---

#### AdamW (Improved Adam)

**Simple definition:**
AdamW is Adam with better regularization management. It has become the standard for training large language models like GPT and BERT.

**Principle:**
Improved version of Adam that handles regularization better.

**When to use it:**
- Modern Transformers (BERT, GPT)
- State-of-the-art NLP
- When Adam + L2 regularization doesn't work well

---

#### RMSprop

**Simple definition:**
RMSprop is an optimizer that adapts well to data that changes over time. It's particularly effective for recurrent networks (RNN, LSTM).

**Principle:**
Adapts learning rate, good for certain architectures.

**When to use it:**
- **RNN and LSTM** (very effective)
- Non-stationary data

---

### Optimizer Comparison Table

| Optimizer | Adaptive | Speed | Ease of use | Best for |
|-----------|----------|-------|-------------|----------|
| **SGD** | ❌ | Slow | Difficult | Baseline |
| **SGD + Momentum** | ❌ | Medium | Difficult | Vision (ResNet) |
| **Adam** | ✅ | Fast | **Easy** | **Everything (default)** |
| **AdamW** | ✅ | Fast | Easy | Transformers |
| **RMSprop** | ✅ | Fast | Easy | RNN/LSTM |

**Selection guide:**
- **Don't know what to choose?** → Adam (lr=0.001)
- **RNN/LSTM?** → RMSprop
- **Transformers?** → AdamW
- **Vision + time to tune?** → SGD + Momentum
- **Everything else?** → Adam

---

## Loss Functions - Measuring Error

### What is a Loss Function?

The **loss function** measures **how wrong the model is**. It's what the model tries to **minimize** during training.

**Analogy:** A teacher grading a test - the loss is the number of points lost.

---

### Loss Functions for Regression

#### Mean Squared Error (MSE)

**Simple definition:**
MSE measures error by calculating the square of the difference between prediction and reality. The larger the error, the more severe the penalty (exponential effect).

**Description:** It's like a very strict teacher who punishes 4 times harder for an error that's 2 times larger.

**Principle:**
Calculates the **square** of the difference between prediction and truth, then takes the average.

**Characteristics:**
- Penalizes **very hard** large errors
- Error of 10 → penalty of 100
- Error of 20 → penalty of 400 (2× error = 4× penalty!)

**Advantages:**
- Easy to optimize
- Heavily penalizes large errors

**Disadvantages:**
- Very sensitive to outliers
- A single large error can dominate training

**When to use:**
- Standard regression (price, temperature)
- When large errors are really serious

---

#### Mean Absolute Error (MAE)

**Simple definition:**
MAE measures error by simply taking the absolute difference between prediction and reality. It's a fairer and easier to understand measure.

**Description:** It's like a fair teacher who punishes proportionally: error 2 times larger = punishment 2 times larger.

**Principle:**
Calculates the **absolute value** of the difference, then takes the average.

**Characteristics:**
- Penalty **proportional** to error
- Error of 10 → penalty of 10
- Error of 20 → penalty of 20 (2× error = 2× penalty)

**Advantages:**
- Robust to outliers
- Easy to interpret (same unit as predicted variable)

**Disadvantages:**
- Treats all errors the same way

**When to use:**
- Presence of outliers in data
- You want to treat small and large errors fairly
- Interpretability is important

---

#### Huber Loss - The Compromise

**Simple definition:**
Huber Loss combines the best of MSE and MAE. It's strict on small errors but more tolerant on large errors (outliers).

**Description:** It's like a teacher who is strict on small mistakes but understands that you can make big errors by accident.

**Principle:**
- **Small errors**: MSE behavior (penalizes hard)
- **Large errors**: MAE behavior (penalizes moderately)

**Advantages:**
- Combines advantages of MSE and MAE
- Robust to outliers while still penalizing small errors

**When to use:**
- Outliers present but you still want to penalize them a bit
- Compromise between MSE and MAE

---

### Loss Functions for Classification

#### Binary Cross-Entropy (BCE)

**Simple definition:**
BCE measures how wrong the model is when it must choose between two options (YES or NO, true or false). The more confident AND wrong the model is, the more enormous the penalty.

**Description:** Imagine a doctor who is 99% sure you're sick, but you're healthy - the error is catastrophic!

**Principle:**
Measures the gap between predicted probability and true class (for 2 classes).

**Characteristics:**
- Penalizes **exponentially** confident but wrong predictions

**Intuitive example:**
```
True label: YES (sick)
Prediction 90% YES → small penalty ✅
Prediction 10% YES → big penalty ❌
Prediction 1% YES → huge penalty ⚠️
```

**When to use:**
- Binary classification (spam/not-spam, sick/healthy)
- Network output = sigmoid

---

#### Categorical Cross-Entropy (CCE)

**Simple definition:**
CCE is the version of BCE for choosing among multiple options (cat, dog, bird...). It penalizes the model if it's confident on the wrong answer.

**Description:** It's like a multiple-choice quiz where you lose more points if you're very confident on the wrong answer.

**Principle:**
Version of BCE for **more than 2 classes**.

**Concrete example:**
```
Classes: cat, dog, bird
True label: cat

Prediction: [0.7 cat, 0.2 dog, 0.1 bird]
→ Small loss (good! 70% confident on cat) ✅

Prediction: [0.1 cat, 0.8 dog, 0.1 bird]
→ Big loss (bad! confident on dog when it's a cat) ❌
```

**When to use:**
- Multi-class classification (digits 0-9, types of animals)
- Network output = softmax

---

#### Focal Loss - For Imbalanced Classes

**Simple definition:**
Focal Loss is a smart version of Cross-Entropy that makes the model focus on difficult examples rather than wasting time on easy examples.

**Description:** It's like a teacher who gives more attention to struggling students rather than those who already understand everything.

**Principle:**
Improved version of Cross-Entropy that **focuses on difficult examples**.

**How it works:**
- Easy examples (well predicted) → reduced penalty
- Difficult examples (poorly predicted) → normal penalty

**Effect:**
The model spends more time learning difficult cases.

**When to use:**
- **Very imbalanced classes** (e.g., 1% fraud, 99% normal)
- Object detection
- Anomaly detection

---

### Comparison Table of Loss Functions

| Loss | Type | Outlier Robustness | Usage |
|------|------|-------------------|--------|
| **MSE** | Regression | ❌ Low | Standard regression |
| **MAE** | Regression | ✅ High | Regression with outliers |
| **Huber** | Regression | ✅ Medium | Compromise |
| **Binary Cross-Entropy** | Classification | N/A | 2 classes |
| **Categorical Cross-Entropy** | Classification | N/A | Multi-class |
| **Focal Loss** | Classification | N/A | Imbalanced classes |

---

### How to Choose Your Loss Function?

**Simple decision tree:**

```
Regression (predict a number)?
├─ Yes
│  ├─ Outliers present? 
│  │  ├─ Yes → MAE or Huber
│  │  └─ No → MSE
│
└─ No (Classification)
   ├─ 2 classes? → Binary Cross-Entropy
   ├─ Balanced multi-class? → Categorical Cross-Entropy
   └─ Imbalanced classes? → Focal Loss
```

---

## Evaluation Metrics

### Difference: Loss vs Metrics

**Loss Function:**
- What the model **optimizes** during training
- Not always easy for a human to understand

**Metrics:**
- What we use to **evaluate** performance
- **Easy to interpret** and understand
- Aligned with business objectives

**Example:** We train with Cross-Entropy (loss), but we evaluate with Accuracy (metric).

---

### Metrics for Classification

#### The Confusion Matrix

To understand metrics, you must first understand 4 concepts:

```
                Model Prediction
                YES         NO
Reality  YES    TP          FN
         NO     FP          TN
```

**Simple definitions:**
- **TP (True Positive)**: model says YES, it's really YES ✅
- **TN (True Negative)**: model says NO, it's really NO ✅
- **FP (False Positive)**: model says YES, but it's NO ❌ (false alarm)
- **FN (False Negative)**: model says NO, but it's YES ❌ (missed case)

**Example (Medical test):**
- **TP**: test says sick, patient really sick ✅
- **TN**: test says healthy, patient really healthy ✅
- **FP**: test says sick, patient healthy ❌ (false alarm, unnecessary panic)
- **FN**: test says healthy, patient sick ❌ (DANGEROUS! Missed case)

---

#### Accuracy (Overall Precision)

**Simple definition:**
Accuracy is the simplest metric: it's the percentage of correct predictions. If you have 85% accuracy, it means the model is right 85 times out of 100.

**Description:** It's like your grade on an exam: 85/100 = 85% success.

**Detailed explanation:**
Percentage of correct predictions.

**Calculation:**
Accuracy = (Correct predictions) / (Total predictions)

**Example:**
```
Out of 100 tests:
- 85 correct predictions
- 15 errors
Accuracy = 85/100 = 85%
```

**Advantages:**
- Very simple to understand
- Easy to explain

**IMPORTANT TRAP:**
Doesn't work well with imbalanced classes!

**Example of the trap:**
```
Dataset: 95% not-spam, 5% spam
Naive model that ALWAYS says "not-spam"
→ Accuracy = 95% (looks great!)
But: detects NO spam (useless!)
```

**When to use:**
- Balanced classes (50/50 or close)
- Both types of errors have the same importance

**When to avoid:**
- Imbalanced classes
- One type of error is more serious than the other

---

#### Precision

**Simple definition:**
Precision measures: "When the model says YES, how often is it really right?" It's important when false alarms are costly.

**Description:** Imagine a smoke detector that's too sensitive and goes off all the time - it has bad precision (many false alarms).

**Question asked:**
"When the model says YES, how often is it right?"

**Calculation:**
Precision = True positives / All predicted positives

**Concrete example (Spam detector):**
```
Model marks 100 emails as spam
- 70 are really spam (TP)
- 30 are important (FP - ERROR!)

Precision = 70/100 = 70%
```

**Interpretation:**
Out of 10 emails marked spam, 7 are really spam, but 3 are important.

**High importance when:**
The cost of a **false alarm** is high

**Examples:**
- **Spam filter**: marking an important email as spam → angry customer
- **Recommendations**: recommending a bad product → frustrated user
- **Marketing targeting**: targeting someone who won't buy → unnecessary cost

**Business interpretation:**
High precision = **conservative** model: "I only say YES if I'm very sure"

---

#### Recall (Sensitivity)

**Simple definition:**
Recall measures: "Of the really positive cases, how many does the model detect?" It's important when missing a case is dangerous.

**Description:** Imagine a smoke detector that never goes off - it has bad recall (misses real dangers).

**Question asked:**
"Of the really positive cases, how many does the model find?"

**Calculation:**
Recall = True positives / All real positives

**Concrete example (Cancer detection):**
```
100 patients really have cancer
- 90 are detected by the model (TP)
- 10 are NOT detected (FN - SERIOUS!)

Recall = 90/100 = 90%
```

**Interpretation:**
The model detects 90% of cancers, but misses 10% (dangerous!).

**High importance when:**
The cost of **missing a case** is high

**Examples:**
- **Cancer detection**: missing a cancer → potentially fatal 🚨
- **Fraud detection**: missing fraud → financial loss
- **Security**: missing a danger → catastrophe
- **COVID**: missing a case → disease spread

**Business interpretation:**
High recall = **liberal** model: "I prefer to say YES too often than miss cases"

---

#### The Precision vs Recall Dilemma

**Problem:**
Improving one often degrades the other!

**Conservative model (high precision):**
- Says YES only if very sure
- Few false alarms (high precision ✅)
- But misses many cases (low recall ❌)

**Liberal model (high recall):**
- Says YES at the slightest doubt
- Detects almost all cases (high recall ✅)
- But many false alarms (low precision ❌)

**Example:**
```
Very strict spam detector:
→ Precision = 95% (almost everything marked spam is really spam)
→ Recall = 50% (but misses half of the spam)

Very loose spam detector:
→ Precision = 40% (many false alarms)
→ Recall = 98% (catches almost all spam)
```

---

#### F1-Score - The Balance

**Simple definition:**
F1-Score is a compromise between Precision and Recall. It's an overall score that penalizes the model if either one is bad.

**Description:** It's like a harmonic mean - to have a good F1, you MUST be good at Precision AND Recall, not just one of them.

**Principle:**
Combines Precision and Recall into **a single number**.

**Important characteristic:**
If Precision OR Recall is low, F1 will be low too.

**Example:**
```
Case 1:
Precision = 100%, Recall = 10%
F1 = 18% (low, because recall too low)

Case 2:
Precision = 50%, Recall = 50%
F1 = 50% (balance)

Case 3:
Precision = 90%, Recall = 80%
F1 = 85% (good balance)
```

**When to use:**
- You want a **compromise** between Precision and Recall
- Imbalanced classes
- You want a single metric to compare models

---

#### AUC-ROC - Overall Performance

**Simple definition:**
AUC-ROC is a score between 0 and 1 that measures the model's overall ability to distinguish positive from negative, regardless of the decision threshold chosen.

**Description:** It's like evaluating a student's general ability, not just their grade on a single exam. A good model will have an AUC close to 1.

**Principle:**
Measures model performance **for all possible thresholds**.

**What is a threshold?**
```
The model gives a probability: 0.7 (70% confident)
Threshold = 0.5: 0.7 > 0.5 → prediction YES
Threshold = 0.8: 0.7 < 0.8 → prediction NO
```

**ROC Curve:**
Tests all possible thresholds and plots a curve.

**AUC (Area Under Curve):**
Area under the ROC curve - value between 0 and 1.

**Interpretation:**
- **AUC = 1.0**: perfect model 🏆
- **AUC = 0.9-1.0**: excellent
- **AUC = 0.8-0.9**: good
- **AUC = 0.7-0.8**: acceptable
- **AUC = 0.5-0.7**: weak
- **AUC = 0.5**: as good as coin flip 🎲
- **AUC < 0.5**: worse than random (reverse predictions!)

**Advantages:**
- Evaluates the model globally (all thresholds)
- No need to choose a threshold in advance
- Facilitates comparison between models

**When to use:**
- Compare multiple models
- Imbalanced classes
- You don't yet know which threshold to use

---

### Metrics for Regression

#### MAE (Mean Absolute Error)

**Principle:**
Average error, in the unit of what you're predicting.

**Example (Price in dollars):**
```
Predictions: [$100, $200, $150]
True values: [$110, $180, $155]
Errors: [$10, $20, $5]
MAE = (10 + 20 + 5) / 3 = $11.67
```

**Simple interpretation:**
"On average, I'm off by $11.67"

**Advantages:**
- Very easy to interpret
- Same unit as what you're predicting
- Robust to outliers

---

#### RMSE (Root Mean Squared Error)

**Principle:**
Similar to MAE but penalizes large errors more heavily.

**Difference with MAE:**
- MAE: error of 10 → penalty of 10
- RMSE: error of 10 → penalty closer to 10.5
- Large error of 50 → RMSE penalizes much more

**Advantages:**
- Same unit as what you're predicting
- Penalizes large errors more

**Comparison MAE vs RMSE:**
- If RMSE >> MAE: you have large errors (outliers)
- If RMSE ≈ MAE: your errors are homogeneous

---

#### R² (R-squared / Coefficient of Determination)

**Principle:**
Measures **what percentage of variability** is explained by the model.

**Interpretation:**
- **R² = 1**: perfect model (100% of variance explained) 🏆
- **R² = 0.9**: excellent (90% explained)
- **R² = 0.7**: good
- **R² = 0.5**: medium
- **R² = 0**: model is as good as predicting the mean
- **R² < 0**: model is WORSE than predicting the mean 😱

**Intuitive example:**
```
Predicting house prices
R² = 0.85

Interpretation:
85% of price variations are explained by characteristics
  (size, location, number of bedrooms...)
15% remain unexplained
  (luck, negotiation, market conditions...)
```

**Advantages:**
- Easy to interpret (percentage)
- Allows comparison between different datasets

---

### Summary Table - Classification

| Metric | Question | Use when |
|--------|----------|----------|
| **Accuracy** | How many correct predictions? | Balanced classes |
| **Precision** | When I say YES, am I right? | False alarms costly |
| **Recall** | Of real YES, how many do I find? | Missed cases costly |
| **F1-Score** | What Precision/Recall balance? | Compromise needed |
| **AUC-ROC** | Overall performance all thresholds? | Compare models |

---

### Summary Table - Regression

| Metric | Unit | Advantage | Use when |
|--------|------|-----------|----------|
| **MAE** | Same as y | Easy to interpret | Outliers present |
| **RMSE** | Same as y | Penalizes large errors | Large errors serious |
| **R²** | Unitless (%) | Intuitive | Compare datasets |

---

## Regularization - Fighting Overfitting

### Understanding Overfitting

**Analogy:**
A student who memorizes past exams by heart, but fails on new questions.

**Signs of overfitting:**
- Train accuracy = 98%, Validation accuracy = 75% (**large gap**)
- Train loss continues to decrease, validation loss **goes back up**
- Excellent on training data, poor on new data

---

### Regularization Techniques

#### 1. Early Stopping - The Simplest

**Simple definition:**
Early Stopping automatically stops training when the model starts to overfit. It's the easiest and most effective regularization technique.

**Description:** It's like a coach telling you "Stop, you're at your best, if you continue you'll overtrain and hurt yourself".

**Principle:**
Stops training when validation performance starts to degrade.

**How it works:**
```
Epochs 1-20: validation improves → continue
Epoch 25: best validation → save
Epochs 26-35: validation stagnates/degrades
Epoch 36: patience exhausted → STOP and restore epoch 25
```

**Recommended configuration:**
- Patience: 10-15 epochs
- Monitor: validation loss
- Restore best weights: YES

**Advantages:**
- Free (no additional computation)
- Very effective
- Saves time

---

#### 2. Dropout - Powerful Technique

**Simple definition:**
Dropout randomly deactivates neurons during training to force the network not to depend too much on specific neurons. This avoids overfitting.

**Description:** It's like training a soccer team where in each game, some players are randomly absent - all players must become versatile.

**Principle:**
During training, **randomly deactivates** neurons.

**Why it's effective:**

**Prevents co-dependency:**
- Neurons can't count on their neighbors
- Each neuron must learn useful things independently

**Forces redundancy:**
- Important information must be encoded by multiple neurons
- Makes the network more robust

**Analogy:**
Training a team where each game, different players play. Everyone must be versatile.

**Configuration:**
- **Dropout = 0.2**: 20% of neurons deactivated (light)
- **Dropout = 0.5**: 50% deactivated (standard for large layers)

**Where to place:**
- ✅ After large layers (512+ neurons)
- ❌ Not on output layer
- ❌ Not on first layer

---

#### 3. L2 Regularization (Weight Decay)

**Simple definition:**
L2 Regularization penalizes weights that are too large in the network, forcing the model to find simpler and more general solutions.

**Description:** It's like a diet for the network - we prevent it from becoming too "heavy" and complex by limiting weight size.

**Principle:**
Penalizes weights that are too high, pushes the model toward simpler solutions.

**Effect:**
- Weights are pushed toward zero (but not exactly zero)
- The model becomes less "confident" and more cautious

**Configuration:**
- Lambda = 0.01: moderate regularization (default)
- Lambda = 0.001: light
- Lambda = 0.1: strong

**When to use:**
- Moderate to strong overfitting
- Many features
- Model with many parameters

---

#### 4. Data Augmentation

**Simple definition:**
Data Augmentation artificially creates new training data by slightly modifying existing data (rotation, zoom, etc.). More data = less overfitting.

**Description:** It's like having 100 photos of a cat and creating 1000 photos by rotating them, zooming, changing brightness - the model sees more variations.

**Principle:**
Artificially create new training data by transforming existing data.

**For images:**
- Rotation (-20° to +20°)
- Zoom
- Horizontal flip
- Brightness change
- Contrast change

**Example:**
```
1 cat image
→ Rotation: 5 new images
→ Zoom: 3 new images
→ Flip: 1 new image
Total: 10 images from one!
```

**For text:**
- Replace words with synonyms
- Swap two words
- Insert random words

**When to use:**
- Small dataset (<10k images)
- Image classification (very effective)
- You want a robust model

**Caution:**
Transformations must be **realistic**!
- ✅ Rotation of a face: OK
- ❌ Vertical flip of a face: weird

---

#### 5. Batch Normalization

**Simple definition:**
Batch Normalization normalizes data between each layer of the network to stabilize and accelerate learning. It's like resetting the counters at each step.

**Description:** Imagine an assembly line where at each station, we check and adjust parts so they're always in the right dimensions.

**Principle:**
Normalizes activations between layers during training.

**Multiple benefits:**
- **Accelerates** training (2-3× faster)
- **Regularizes** (adds beneficial noise)
- Allows using higher learning rates
- Makes the network more stable

**When to use:**
- Deep networks (>10 layers)
- Computer vision (very common)
- Unstable training
- You want faster convergence

---

#### 6. ReduceLROnPlateau

**Simple definition:**
ReduceLROnPlateau automatically reduces the learning rate (learning speed) when the model stops progressing, allowing fine-tuning at the end of training.

**Description:** It's like slowing down when you approach your destination - at first you drive fast, then you slow down to park precisely.

**Principle:**
Automatically reduces learning rate when improvement stagnates.

**Metaphor:**
At start: car (fast, exploration)
Near goal: on foot (slow, precise)

**Configuration:**
- Reduction: divide by 2
- Patience: 5-10 epochs
- Minimum learning rate: 0.0000001

**Advantages:**
- Automatic
- Avoids plateaus
- Fine-tuning at end of training

---

### Regularization Strategy by Levels

**Level 1 - Light regularization:**
1. Early stopping (patience 15-20)
2. Data augmentation (if images)
3. Light dropout (0.2-0.3)

**Level 2 - Moderate regularization:**
1. Stronger dropout (0.4-0.5)
2. L2 regularization (lambda=0.01)
3. ReduceLROnPlateau

**Level 3 - Strong regularization:**
1. Increase lambda (0.05-0.1)
2. Very strong dropout (0.6)
3. Collect more data

**Principle:** Add progressively, not all at once!

---

## Training Methodology

### Phase 1: Data Preparation

**1. Exploration**
- Visualize data
- Understand distribution
- Identify anomalies

**2. Cleaning**
- Handle missing values
- Remove extreme outliers
- Correct errors

**3. Splitting**
- **Train**: 70-80% (to learn)
- **Validation**: 10-15% (to adjust)
- **Test**: 10-15% (for final evaluation)

**4. Preprocessing**
- Normalization (put between 0 and 1)
- Standardization (mean=0, std=1)
- Encoding categorical variables

---

### Phase 2: Baseline Model

**Objective:** Establish a simple reference

**Actions:**
1. Create a **simple** model: 1-2 layers, 64-128 neurons
2. No regularization (except early stopping)
3. Adam (lr=0.001), batch 32
4. Train 50-100 epochs

**Evaluation:**
- Performance: is the problem feasible?
- Train/val gap: overfitting already present?

---

### Phase 3: Increase Complexity

**When:** Low performance on train AND validation (underfitting)

**Progressive actions:**
1. Double number of neurons (64 → 128)
2. Add 1-2 layers
3. Increase number of epochs
4. Add Batch Normalization

**Check after each change:**
- Train accuracy increases? → good sign
- Val accuracy follows? → excellent
- Val accuracy stagnates/drops? → move to phase 4

---

### Phase 4: Regularization

**When:** Large gap between train and validation (overfitting)

**Apply by levels:**
1. Increase early stopping patience
2. Add data augmentation
3. Add dropout (0.3-0.5)
4. Add L2 regularization
5. ReduceLROnPlateau

---

### Phase 5: Fine-Tuning

**When:** Model performs well, you want the last %

**Actions:**
- Test different learning rates
- Test different batch sizes
- Adjust dropouts
- Try different optimizers

---

## Best Practices

### Recommended Workflow

**1. Always start simple**
- Minimal viable model
- No regularization at first
- Establish baseline

**2. Iterate methodically**
- One change at a time
- Measure impact
- Keep what works

**3. Visualize constantly**
- Loss curves (train vs val)
- Metric curves
- Confusion matrix

**4. Save smartly**
- Best model (minimal val loss)
- Regular checkpoints

---

### Checklist Before Training

- [ ] Data explored and understood
- [ ] Data cleaned
- [ ] Train/val/test split done
- [ ] Preprocessing defined
- [ ] Architecture chosen
- [ ] Hyperparameters defined
- [ ] Early stopping configured
- [ ] Metrics chosen

---

### Checklist During Training

- [ ] Monitor train vs val loss
- [ ] Verify loss is decreasing
- [ ] Observe stability
- [ ] Save regularly

---

### Checklist After Training

- [ ] Evaluate on test set
- [ ] Analyze errors
- [ ] Calculate all metrics
- [ ] Document results
- [ ] Save final model

---

### Common Mistakes to Avoid

**1. Not separating train/val/test**
→ Impossible to detect overfitting

**2. Testing too often on validation**
→ Overfitting on validation!

**3. Forgetting to normalize**
→ Very slow convergence

**4. No early stopping**
→ Waste of time and overfitting

**5. Changing too many things at once**
→ Impossible to know what works

**6. Poorly chosen learning rate**
→ #1 cause of failure!

**7. Not visualizing**
→ Misses obvious problems

**8. Not analyzing errors**
→ No targeted improvement

---

### Quick Diagnosis

**Problem**: Loss becomes NaN
**Cause**: Learning rate too high
**Solution**: Divide learning rate by 10

---

**Problem**: No improvement after 20 epochs
**Cause**: Learning rate too low
**Solution**: Multiply by 2-5

---

**Problem**: Train accuracy = 100%, Val accuracy = 70%
**Cause**: Severe overfitting
**Solution**: Dropout + L2 + Data Augmentation

---

**Problem**: Train accuracy = 60%, Val accuracy = 58%
**Cause**: Underfitting
**Solution**: More complex model

---

**Problem**: All predictions = same class
**Cause**: Imbalanced classes or LR too high
**Solution**: Class weights + Reduce LR

---

## Quick Start Guide

### Recommended Default Configuration

**Architecture:**
- 2-3 hidden layers
- 128-256 neurons per layer
- ReLU activation
- Softmax or Sigmoid output

**Training:**
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32 (adjust based on memory)
- Epochs: 100-200

**Regularization:**
- Early stopping (patience 15)
- Dropout 0.3-0.5

**This configuration works for 80% of cases!**

---

### When to Change What

**Low performance everywhere?**
→ Increase complexity (+ layers, + neurons)

**Large train/val gap?**
→ More regularization (dropout, L2)

**Unstable training?**
→ Reduce learning rate, add Batch Norm

**Training too slow?**
→ Increase learning rate, increase batch size

---

## Simple Glossary

**Epoch**: One complete pass over all training data

**Batch**: Group of examples processed together

**Forward Pass**: Data passes through the network

**Backward Pass**: Network calculates how to improve

**Gradient**: Direction to adjust weights

**Learning Rate**: Step size to adjust weights

**Overfitting**: Excessive memorization

**Underfitting**: Model too simple

**Regularization**: Techniques against overfitting


