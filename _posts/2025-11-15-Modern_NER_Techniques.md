---
title: "Named Entity Recognition: Modern Techniques"
date: 2025-11-15 14:00:00
categories: [machine-learning, natural-language-processing]
tags: [ner, nlp, deep-learning, python, transformers, ai]    
image:
  path: /assets/imgs/headers/ner.jpg
---

## Introduction

Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP) that involves identifying and classifying specific entities in text, such as names of people, organizations, locations, dates, or other relevant categories.

## Definition and Importance

NER enables automatic extraction of structured information from unstructured text. This technique is essential for:
- Information extraction
- Semantic search
- Question-answering systems
- Contextual sentiment analysis
- Knowledge graph creation

## Modern NER Techniques

### 1. Transformer-Based Approaches

Transformers represent the most powerful architecture family for NER today. Unlike traditional models that read text sequentially, Transformers analyze all words simultaneously through their attention mechanism, enabling richer context understanding.

#### BERT (Bidirectional Encoder Representations from Transformers)
BERT revolutionized NER through its ability to capture bidirectional context:
- **Architecture**: Transformer encoder with multi-head attention
- **Fine-tuning**: Adaptation on annotated NER datasets

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load a pre-trained BERT model for NER
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# NER pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Usage example
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
entities = nlp(text)
print(entities)
```

#### RoBERTa and Optimized Variants

Several BERT variants have been developed to improve either performance or efficiency. These models offer different tradeoffs between accuracy, speed, and size.

- **RoBERTa**: Improved BERT version with optimized training
- **DistilBERT**: Lightweight version (40% smaller) with 97% of the performance
- **ALBERT**: Architecture factorizing parameters to reduce model size

### 2. Generative Models for NER

Rather than classifying each word, generative models treat NER as a text generation problem. This approach provides great flexibility: new entity categories can be defined simply by modifying instructions, without retraining.

#### GPT and Autoregressive Models
Modern generative models can perform NER via:
- **Prompt engineering**: Formulating the task as text generation
- **Few-shot learning**: Learning with few examples
- **Zero-shot NER**: Without specific training examples

```python
# Example with GPT for NER
prompt = """Extract all named entities from the following text and classify them:
Text: "Microsoft announced that Satya Nadella will speak at the conference in Seattle next Monday."
Entities:
- Person: 
- Organization: 
- Location: 
- Date: 
"""
```

#### T5 (Text-to-Text Transfer Transformer)

T5 adopts a unified approach where all NLP tasks are formulated as "text-in, text-out". For NER, the model receives text and directly generates the list of entities.

- Formulation of NER as a text-to-text generation task
- Flexibility to define new entity categories
- Competitive performance with less training data

### 3. Hybrid and Multi-Model Approaches

Combining different architectures often yields better performance than a single model. These approaches leverage the complementary strengths of each method to maximize accuracy.

#### Ensemble Methods
Combining multiple models to improve robustness:
- Majority voting between BERT, RoBERTa, and ELECTRA
- Model stacking with meta-learner
- 2-5% improvement on F1 metrics

#### BiLSTM-CRF with Contextual Embeddings

This architecture combines three powerful components: rich BERT embeddings, LSTMs to model sequences, and CRFs to ensure prediction consistency. Each element plays a specific role in the processing chain.

Hybrid architecture combining:
- **Contextual embeddings**: ELMo, BERT, or XLNet
- **BiLSTM**: Captures sequential dependencies
- **CRF**: Tag consistency constraints


```python
import torch
import torch.nn as nn
from transformers import BertModel

class BertBiLSTMCRF(nn.Module):
    def __init__(self, bert_model, num_labels, hidden_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.lstm = nn.LSTM(768, hidden_dim // 2, 
                           bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)
        emissions = self.hidden2tag(lstm_output)
        
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())
```

### 4. Span-Based NER Techniques

Rather than labeling each word individually, span-based methods directly identify complete text segments. This approach is more natural as entities are perceived as whole units rather than word sequences.

#### SpanBERT and Span-Oriented Models
Instead of classifying each token, these models:
- Directly identify entity spans (segments)
- Avoid tokenization issues
- Better handling of multi-word entities

#### LUKE (Language Understanding with Knowledge-based Embeddings)

LUKE enriches textual representations with structured knowledge from bases like Wikipedia. This allows the model to better understand entities by leveraging external information.

- Integrates entity knowledge into embeddings
- Transformer architecture with entity-aware attention
- State-of-the-art on multiple benchmarks

### 5. Transfer Learning and Few-Shot Learning

These techniques enable rapid adaptation of models to new domains with very few annotated examples. Particularly useful when annotation is expensive or for highly specialized domains.

#### Meta-Learning for NER
- **Prototypical Networks**: Learning representations for few-shot classification
- **MAML** (Model-Agnostic Meta-Learning): Rapid adaptation to new domains
- Useful for domains with limited annotated data

#### Domain Adaptation

Domain adaptation solves the following problem: a model trained on news articles performs poorly on medical texts. These techniques enable knowledge transfer from a source domain to a target domain.

Techniques for adapting models to new domains:
- **Adversarial training**: Domain-invariant learning
- **Self-training**: Using high-confidence predictions as labels
- **Multi-task learning**: Joint training on multiple tasks

### 6. Multilingual and Cross-Lingual Approaches

These models learn shared representations across many languages, enabling knowledge transfer from data-rich languages to lower-resource languages. A model trained in English can often work directly in French or Spanish.

#### mBERT and XLM-RoBERTa
- Pre-trained models on 100+ languages
- Zero-shot transfer between languages
- Performance comparable to monolingual models

```python
from transformers import XLMRobertaForTokenClassification

# Multilingual model for NER
model = XLMRobertaForTokenClassification.from_pretrained(
    "xlm-roberta-large-finetuned-conll03-english"
)

# Can be used on different languages without fine-tuning
```

### 7. LLMs and In-Context Learning

Large language models like GPT-4 and Claude represent a radically different approach: they can perform NER without specific training, simply by following natural language instructions. This unprecedented flexibility comes at a cost: higher latency and significant API fees.

#### GPT-4, Claude, and Other Modern LLMs
Large language models offer:
- **Zero-shot NER**: Without training examples
- **Few-shot prompting**: With a few in-context examples
- **Instruction following**: Understanding complex instructions
- **Custom entities**: Defining new categories on-the-fly

```python
# Example with OpenAI API
import openai

prompt = """Extract named entities from the following text. Identify PERSON, ORGANIZATION, LOCATION, and DATE entities.

Text: "Elon Musk announced that Tesla will open a new factory in Austin, Texas by December 2024."

Return the entities in JSON format."""

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

## Performance Enhancement Techniques

Even with the best models, the quality and quantity of training data remain crucial. These techniques maximize performance with limited resources and reduce annotation costs.

### 1. Data Augmentation

Data augmentation automatically creates variations of existing examples to enrich the dataset without additional manual annotation. A dataset of 5000 examples can thus be extended to 15000-20000 examples.

- **Synonym replacement**: Replacement with synonyms
- **Back-translation**: Round-trip translation
- **Entity swapping**: Exchanging similar entities
- **Contextual word embeddings augmentation**

### 2. Active Learning

Rather than randomly annotating data, active learning intelligently identifies the most useful examples for the model. This strategic approach can halve the number of examples needed to reach a given performance level.

- Intelligent selection of examples to annotate
- 50-70% reduction in annotation costs
- Strategies: uncertainty sampling, query-by-committee

### 3. Weak Supervision

Weak supervision enables automatic generation of training data using rules, dictionaries, or existing knowledge bases. Although less precise than human annotations, this data allows training functional models quickly.

- **Snorkel**: Using labeling functions
- **Distant supervision**: Using knowledge bases (Wikipedia, Wikidata)
- Automatic training data generation

### 4. Handling Nested Entities

Some texts contain nested entities, like "Bank of America" (ORGANIZATION) which contains "America" (LOCATION). Classic BIO approaches cannot handle these cases, requiring specialized techniques.

Techniques for multi-level entities:
- **Layered CRF**: Multiple CRF layers
- **Hypergraph-based approaches**: Hypergraph modeling
- **Anchor-Region Networks**

## Evaluation and Metrics

### Standard Metrics
- **Precision**: Proportion of correctly identified entities
- **Recall**: Proportion of actual entities found
- **F1-Score**: Harmonic mean of precision and recall
- **Exact match** vs **Partial match**

### Reference Datasets
- **CoNLL-2003**: English, news articles
- **OntoNotes 5.0**: Multi-domain, multi-type
- **WNUT**: Social media texts
- **MIT Restaurant & Movie**: Specific domains

### Performance Comparison on CoNLL-2003

| Model | F1-Score | Parameters | Speed (tokens/sec) | Year |
|--------|----------|------------|---------------------|-------|
| BERT-base | 92.4% | 110M | 500 | 2018 |
| BERT-large | 92.8% | 340M | 200 | 2018 |
| RoBERTa-large | 93.1% | 355M | 180 | 2019 |
| ALBERT-xxlarge | 93.3% | 235M | 150 | 2019 |
| XLNet-large | 93.5% | 340M | 160 | 2019 |
| SpanBERT-large | 93.7% | 340M | 190 | 2020 |
| LUKE-large | 94.3% | 483M | 140 | 2020 |
| ELECTRA-large | 93.9% | 335M | 250 | 2020 |
| DeBERTa-v3-large | 94.1% | 434M | 170 | 2021 |
| BiLSTM-CRF + BERT | 93.6% | 120M | 450 | 2019 |
| GPT-4 (few-shot) | 91-94%* | 1.7T+ | 50 | 2023 |
| GPT-4o (few-shot) | 92-95%* | - | 80 | 2024 |

*Variable performance depending on prompting and number of examples

### Accuracy vs Resources Trade-offs

| Approach | Accuracy | Latency | Compute Cost | Fine-tuning Ease | Ideal Use Case |
|----------|-----------|---------|--------------|---------------------|----------------|
| spaCy (rule-based) | Average | Very low | Very low | Easy | Rapid prototyping |
| BERT-base | High | Medium | Medium | Easy | Standard production |
| BERT-large | Very high | High | High | Medium | High precision |
| DistilBERT | Good | Low | Low | Easy | Edge/Mobile |
| LLMs (API) | Excellent | Variable | Very high | None | Maximum flexibility |
| BiLSTM-CRF | Average | Low | Low | Difficult | Legacy systems |
| Ensemble | Excellent | Very high | Very high | Complex | Competitions |

## Current Challenges and Future Directions

### Challenges
1. **Emerging entities**: New entities not seen during training
2. **Contextual ambiguity**: "Apple" (fruit vs company)
3. **Multi-word and discontinuous entities**
4. **Limited resources**: Low-resource languages and domains
5. **Data bias**: Unequal entity representation

### Future Trends
1. **Continual Learning**: Continuous adaptation without forgetting
2. **Multimodal NER**: Integration of text + images/videos
3. **Explainability**: Interpretation of model decisions
4. **Efficiency**: Lighter models for edge deployment
5. **Universal NER**: Models generalizable to all domains

## Practical Use Cases

NER finds concrete applications in many sectors. Each domain presents specific challenges and often requires adaptation of generic models.


### 1. Legal Document Analysis

Legal documents contain specialized terminology and formal structures. NER helps automatically extract key information to facilitate analysis and search.

- Extraction of party names, dates, jurisdictions
- Identification of legal references
- Automatic contract structuring

### 2. Social Media Monitoring

Social media texts are informal, with abbreviations, spelling errors, and neologisms. NER must be robust to these variations to correctly identify mentions of brands, people, and places.

- Detection of brand and personality mentions
- Contextual sentiment analysis
- Geographic trend detection

### 3. Medical Sector

The medical domain requires maximum precision and strict regulatory compliance (HIPAA, GDPR). NER helps structure patient records and anonymize sensitive data.

- Patient information extraction (PHI)
- Identification of medications, diseases, symptoms
- HIPAA compliance with anonymization

### 4. Financial Analysis

Financial analysis requires rapid extraction of key information from reports, news articles, and regulatory documents. NER automates this extraction to accelerate decision-making.

- Extraction of company names, financial indicators
- Market event detection
- Financial report structuring

## Practical Fine-Tuning Guide

This guide provides a complete journey to adapt a pre-trained model to your specific domain. From data preparation to deployment, each step is detailed with concrete code examples.

### 1. Data Preparation

Training data quality is crucial. A well-prepared dataset can compensate for a simpler model, while poor-quality data will limit even the best models.

#### Data Format
NER data must be in BIO (Begin, Inside, Outside) or IOB2 format:

```python
# Exemple de format CoNLL
"""Pierre B-PER
Dupont I-PER
travaille O
chez O
Google B-ORG
à O
Paris B-LOC
. O
"""

# Format JSON pour Hugging Face
data = {
    "tokens": ["Pierre", "Dupont", "travaille", "chez", "Google", "à", "Paris", "."],
    "ner_tags": [1, 2, 0, 0, 3, 0, 5, 0]  # B-PER, I-PER, O, O, B-ORG, O, B-LOC, O
}
```

#### Dataset Sizes
- **Minimum viable**: 1000-2000 annotated sentences
- **Recommended**: 5000-10000 sentences
- **Optimal**: 20000+ sentences
- **Few-shot with LLMs**: 10-50 examples

### 2. Training Configuration

#### Recommended Hyperparameters for BERT

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./ner-model",
    
    # Optimal hyperparameters
    learning_rate=2e-5,  # 2e-5 to 5e-5 for BERT
    per_device_train_batch_size=16,  # Adjust according to GPU
    per_device_eval_batch_size=32,
    num_train_epochs=3,  # 3-5 epochs generally sufficient
    weight_decay=0.01,
    
    # Learning rate strategy
    lr_scheduler_type="linear",  # or "cosine"
    warmup_ratio=0.1,  # 10% warmup
    
    # Evaluation and saving
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    
    # Stability
    seed=42,
    fp16=True,  # If GPU compatible
    
    # Logging
    logging_dir="./logs",
    logging_steps=100,
    report_to="tensorboard"
)
```

#### Batch Sizes by GPU

| GPU | VRAM | BERT-base batch | BERT-large batch | RoBERTa-large batch |
|-----|------|-----------------|------------------|--------------------|
| RTX 3060 | 12GB | 16-24 | 4-8 | 4-8 |
| RTX 3090 | 24GB | 32-48 | 12-16 | 12-16 |
| A100 | 40GB | 64-96 | 24-32 | 24-32 |
| A100 | 80GB | 128+ | 48-64 | 48-64 |

### 3. Complete Training Pipeline

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer
)
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# 1. Load and prepare data
dataset = load_dataset("conll2003")  # Or your own data
label_list = dataset["train"].features["ner_tags"].feature.names

# 2. Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label_list),
    id2label={i: label for i, label in enumerate(label_list)},
    label2id={label: i for i, label in enumerate(label_list)}
)

# 3. Tokenization function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens ignored
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # For subtokens, use same label or -100
                label_ids.append(label[word_idx])  # or -100
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 4. Apply tokenization
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# 5. Metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored labels
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# 6. Trainer
data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. Training
trainer.train()

# 8. Final evaluation
results = trainer.evaluate()
print(results)

# 9. Save model
model.save_pretrained("./final-ner-model")
tokenizer.save_pretrained("./final-ner-model")
```

### 4. Optimization Techniques

#### Gradient Accumulation
To simulate larger batches with limited GPU:
```python
training_args = TrainingArguments(
    gradient_accumulation_steps=4,  # Effective batch = 16 * 4 = 64
    per_device_train_batch_size=16,
)
```

#### Mixed Precision Training
```python
training_args = TrainingArguments(
    fp16=True,  # Reduces memory usage by ~50%
)
```

#### Learning Rate Scheduling
```python
# Option 1: Linear decay with warmup (recommended)
lr_scheduler_type="linear"
warmup_ratio=0.1

# Option 2: Cosine annealing
lr_scheduler_type="cosine"
warmup_steps=500
```

### 5. Validation and Debugging

#### Overfitting Detection
```python
# Monitor these metrics
- Training loss decreases but validation loss increases
- F1 train >> F1 validation (gap > 5%)

# Solutions:
- Increase weight_decay (0.01 -> 0.1)
- Add dropout (hidden_dropout_prob=0.2)
- Reduce epochs
- Data augmentation
```

#### Error Analysis
```python
from seqeval.metrics import classification_report

# Detailed report by class
print(classification_report(true_labels, predictions))

# Identify problematic entities
errors = []
for i, (true, pred) in enumerate(zip(true_labels, predictions)):
    if true != pred:
        errors.append({
            "sentence_id": i,
            "true": true,
            "predicted": pred
        })
```

### 6. Best Practices

✅ **Do**
- Validate on a dataset representative of the target domain
- Use early stopping based on F1-score
- Save multiple checkpoints
- Test on out-of-distribution data
- Document hyperparameters

❌ **Don't**
- Fine-tune on < 500 examples (except specialized few-shot)
- Use learning rate too high (> 5e-5 for BERT)
- Ignore class imbalance
- Over-optimize on validation set
- Forget to set seeds for reproducibility

## Production Deployment

Moving from a functional prototype to a robust production system requires optimizing latency, cost, and reliability. This chapter covers essential techniques for successful deployment.

### 1. Inference Optimization

Transformer models are powerful but resource-intensive. Several techniques can drastically reduce size and inference time with minimal accuracy loss.

#### Quantization
Size reduction and model acceleration:

```python
from transformers import AutoModelForTokenClassification
import torch

# Dynamic quantization
model = AutoModelForTokenClassification.from_pretrained("./ner-model")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Gain: 4x smaller, 2-3x faster
# F1 loss: < 0.5%
```

#### ONNX Runtime
```python
from optimum.onnxruntime import ORTModelForTokenClassification

# ONNX conversion
model = ORTModelForTokenClassification.from_pretrained(
    "./ner-model",
    export=True
)

# Gain: 1.5-2x faster
# Compatible CPU and GPU
```

#### Distillation
```python
# Use DistilBERT for 40% size reduction
# with 97% of the performance
from transformers import DistilBertForTokenClassification

student_model = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-cased",
    num_labels=len(label_list)
)

# Train with distillation from BERT-large
```

### 2. Deployment Architecture

#### REST API with FastAPI
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()

# Load model at startup
ner_pipeline = pipeline(
    "ner",
    model="./ner-model",
    aggregation_strategy="simple",
    device=0  # GPU if available
)

class TextInput(BaseModel):
    text: str
    
class Entity(BaseModel):
    entity_group: str
    score: float
    word: str
    start: int
    end: int

@app.post("/extract-entities", response_model=list[Entity])
async def extract_entities(input_data: TextInput):
    try:
        entities = ner_pipeline(input_data.text)
        return entities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

#### Batch Processing
```python
# For processing large volumes
def batch_ner_processing(texts, batch_size=32):
    all_entities = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        entities = ner_pipeline(batch)
        all_entities.extend(entities)
    
    return all_entities

# Parallel processing
from concurrent.futures import ThreadPoolExecutor

def parallel_ner(texts, n_workers=4):
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(ner_pipeline, texts))
    return results
```

### 3. Metrics

#### Latence and Throughput

| Configuration | Latence (ms/requête) | Throughput (req/sec) | Coût |
|---------------|---------------------|---------------------|------|
| BERT-base CPU | 150-300 | 3-7 | Low |
| BERT-base GPU (T4) | 20-40 | 25-50 | Medium |
| DistilBERT CPU | 50-100 | 10-20 | Low |
| DistilBERT GPU | 10-20 | 50-100 | Medium |
| BERT-large GPU (A100) | 30-60 | 16-33 | High |
| LLM API (GPT-4) | 500-2000 | 0.5-2 | Very high |

#### Monitoring
```python
import time
import logging
from prometheus_client import Counter, Histogram

# Métriques Prometheus
request_count = Counter('ner_requests_total', 'Total NER requests')
request_duration = Histogram('ner_request_duration_seconds', 'Request duration')
error_count = Counter('ner_errors_total', 'Total errors')

@app.post("/extract-entities")
async def extract_entities(input_data: TextInput):
    start_time = time.time()
    request_count.inc()
    
    try:
        entities = ner_pipeline(input_data.text)
        duration = time.time() - start_time
        request_duration.observe(duration)
        
        logging.info(f"Request processed in {duration:.2f}s")
        return entities
        
    except Exception as e:
        error_count.inc()
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 4. Containerization

#### Optimized Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY ./ner-model ./ner-model
COPY ./app ./app

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose for Dev
```yaml
version: '3.8'

services:
  ner-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/ner-model
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/ner-model
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### 5. Production Testing and Validation

#### A/B Testing
```python
import random

def route_to_model(user_id: str):
    # 90% BERT-base, 10% new model
    if hash(user_id) % 100 < 10:
        return "ner-model-v2"
    return "ner-model-v1"

@app.post("/extract-entities")
async def extract_entities(input_data: TextInput, user_id: str):
    model_version = route_to_model(user_id)
    pipeline = get_pipeline(model_version)
    return pipeline(input_data.text)
```

#### Shadow Mode
```python
# Test new model without impacting production
@app.post("/extract-entities")
async def extract_entities(input_data: TextInput):
    # Production
    entities_v1 = ner_pipeline_v1(input_data.text)
    
    # Shadow (async, non-blocking)
    asyncio.create_task(shadow_predict(input_data.text))
    
    return entities_v1

async def shadow_predict(text):
    entities_v2 = ner_pipeline_v2(text)
    log_shadow_results(entities_v2)  # For offline analysis
```

## Ethics, Bias, and Privacy

Deploying NER systems raises important ethical questions. Models can perpetuate existing biases, violate privacy, or discriminate against certain groups. A responsible approach requires vigilance and proactive measures.

### 1. Bias in NER Models

NER models reflect biases present in their training data. These biases can affect different groups unequally, leading to variable performance depending on context.

#### Identified Bias Types

**Geographic Bias**
- Better performance on Western entities
- Example: "Paris" detected at 98% vs "Ouagadougou" at 75%

**Gender Bias**
```python
# Example of observed bias
text1 = "Dr. Martin examined the patient."
text2 = "Dr. Martin examined the patient."  # Feminine form

# Some models have more difficulty with feminine forms
```

**Temporal Bias**
- Recent entities (post-training) less well recognized
- "ChatGPT" in 2021 vs 2024

#### Bias Measurement
```python
from collections import defaultdict

def analyze_bias(model, test_cases):
    results = defaultdict(list)
    
    for category, texts in test_cases.items():
        for text in texts:
            entities = model(text)
            f1 = calculate_f1(entities, text.ground_truth)
            results[category].append(f1)
    
    # Compare performance by category
    for category, scores in results.items():
        print(f"{category}: F1 = {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
    
    # Alert if gap > 10%
    if max(scores) - min(scores) > 0.10:
        print("⚠️ Significant bias detected")
```

### 2. Bias Mitigation

#### Balanced Data Augmentation
```python
# Ensure diverse representation
augmentation_strategy = {
    "western_names": 1000,
    "asian_names": 1000,
    "african_names": 1000,
    "middle_eastern_names": 1000,
}

# Synthetic generation
from faker import Faker

fake_en = Faker('en_US')
fake_ar = Faker('ar_SA')
fake_zh = Faker('zh_CN')

diverse_names = [
    fake_en.name() for _ in range(1000)
] + [
    fake_ar.name() for _ in range(1000)
] + [
    fake_zh.name() for _ in range(1000)
]
```

#### Fairness Constraints
```python
# Training with fairness constraints
from fairlearn.reductions import DemographicParity

# Ensure similar performance on subgroups
constraint = DemographicParity()
```

### 3. Privacy and Compliance

#### GDPR

**Obligations**
- Data minimization: Extract only what's necessary
- Right to be forgotten: Delete personal data on request
- Transparency: Inform about NER usage

**Implementation**
```python
# Automatic anonymization
def anonymize_entities(text, entity_types=["PERSON", "EMAIL", "PHONE"]):
    entities = ner_pipeline(text)
    anonymized = text
    
    for ent in sorted(entities, key=lambda x: x['start'], reverse=True):
        if ent['entity_group'] in entity_types:
            # Replace with placeholder
            placeholder = f"[{ent['entity_group']}_{hash(ent['word']) % 1000}]"
            anonymized = (
                anonymized[:ent['start']] + 
                placeholder + 
                anonymized[ent['end']:]
            )
    
    return anonymized

# Example
text = "Jean Dupont (jean.dupont@email.com) lives in Paris."
print(anonymize_entities(text))
# Output: [PERSON_452] ([EMAIL_789]) lives in Paris.
```

#### HIPAA (Healthcare - USA)
```python
# PHI Detection (Protected Health Information)
PHI_ENTITIES = [
    "PERSON",      # Patient names
    "DATE",        # Dates of birth
    "PHONE",       # Phone numbers
    "EMAIL",       # Emails
    "SSN",         # Social security numbers
    "MEDICAL_ID",  # Medical identifiers
    "ADDRESS"      # Addresses
]

def is_hipaa_compliant(text):
    entities = ner_pipeline(text)
    phi_found = [e for e in entities if e['entity_group'] in PHI_ENTITIES]
    
    if phi_found:
        return False, phi_found
    return True, []
```

### 4. Transparency and Explainability

#### Confidence Scores
```python
# Always return confidence scores
@app.post("/extract-entities")
async def extract_entities(input_data: TextInput):
    entities = ner_pipeline(input_data.text)
    
    # Filter by confidence threshold
    confidence_threshold = 0.85
    high_confidence = [
        e for e in entities 
        if e['score'] >= confidence_threshold
    ]
    
    return {
        "entities": high_confidence,
        "low_confidence_count": len(entities) - len(high_confidence),
        "model_version": "bert-base-v1.2"
    }
```

#### Attention Visualization
```python
from transformers import BertTokenizer, BertForTokenClassification
import torch

def visualize_attention(text, entity):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    
    # Extract attention for entity
    attentions = outputs.attentions[-1]  # Last layer
    
    # Display which tokens influenced the prediction
    # (complete implementation requires viz library)
```

## Conclusion

Named Entity Recognition has evolved considerably with the advent of transformers and LLMs. Modern techniques offer:
- **High accuracy**: F1-scores > 94% on standard benchmarks (CoNLL-2003)
- **Flexibility**: Rapid adaptation to new domains with fine-tuning
- **Multilinguality**: Support for 100+ languages via XLM models
- **Ease of use**: Accessible APIs and frameworks
- **Production deployment**: Solutions optimized for latency and cost
- **Ethical considerations**: Tools for bias and privacy

The future of NER is moving towards more general, explainable, and efficient models, capable of continuously adapting to new contexts while requiring less annotated data. The main challenges remain bias mitigation, privacy protection, and performance/cost optimization for large-scale deployment.

**Key points for successful NER project in 2025:**
1. Choose the right model according to constraints (accuracy vs latency vs cost)
2. Fine-tune with at least 5000 quality annotated examples
3. Implement performance monitoring and drift detection
4. Regularly audit for bias and GDPR/HIPAA compliance
5. Optimize for production (quantization, ONNX, caching)


