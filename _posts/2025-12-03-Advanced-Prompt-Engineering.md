---
title: "Advanced Prompt Engineering: Techniques for Professional LLM Applications"
date: 2025-12-03 10:00:00
categories: [LLM]
tags: [LLM, Prompt Engineering, Chain-of-Thought, ReAct, Few-Shot, Tree-of-Thoughts]
image:
  path: /assets/imgs/headers/prompt_engineering_advanced.png
---

## Introduction

Prompt engineering has evolved from simple "trial and error" to a sophisticated discipline with proven techniques and frameworks. In 2024-2025, advanced prompting strategies have demonstrated **30-50% accuracy improvements** over naive approaches, while reducing hallucinations by up to **60%** and cutting inference costs by **40%** through better model utilization.

This article covers state-of-the-art prompt engineering techniques used in production systems at scale, from Chain-of-Thought to Tree-of-Thoughts and beyond, with real-world benchmarks and implementation patterns.

### What You'll Learn

- **Foundational Techniques**: CoT, ReAct, Self-Consistency with production-ready implementations
- **Advanced Methods**: Tree-of-Thoughts, Program-Aided Language Models, Meta-Prompting
- **Optimization Frameworks**: APE, DSPy, automatic prompt optimization at scale
- **Structured Generation**: JSON mode, function calling, constrained decoding
- **Evaluation & Monitoring**: Metrics, A/B testing, continuous improvement
- **Production Patterns**: Cost optimization, latency reduction, security best practices
- **2024-2025 Innovations**: Constitutional AI prompting, multimodal techniques, prompt caching

> **Target Audience**: ML Engineers, LLM practitioners, and technical leads building production LLM applications requiring high accuracy and reliability.

### Impact Metrics

| Technique | Accuracy Gain | Cost Reduction | Use Case |
|-----------|---------------|----------------|----------|
| Chain-of-Thought | +15-40% | -10% | Math, reasoning |
| Self-Consistency | +10-20% | -30%* | High-stakes decisions |
| Few-Shot (optimized) | +20-35% | -25% | Domain-specific tasks |
| ReAct | +25-45% | Variable | Tool-using agents |
| Structured Output | +30-50% | -40% | Data extraction |

*Cost reduction from reducing retries and error handling

## 1. Prompt Engineering Fundamentals Revisited

### Anatomy of an Effective Prompt

```
┌─────────────────────────────────────────────────────────────────────┐
│                   EFFECTIVE PROMPT STRUCTURE                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 1. SYSTEM CONTEXT / ROLE                                            │
│    "You are an expert [domain] with [expertise]"                    │
│    └─► Sets model's perspective and knowledge base                  │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. TASK DESCRIPTION                                                 │
│    "Your task is to [specific action]"                              │
│    └─► Clear, unambiguous instruction                               │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. CONTEXT & INPUT DATA                                             │
│    Background information, variables, data to process               │
│    └─► Everything model needs to complete task                      │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. OUTPUT FORMAT SPECIFICATION                                      │
│    JSON schema, template, length constraints                        │
│    └─► Ensures parseable, consistent outputs                        │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. CONSTRAINTS & GUIDELINES                                         │
│    Do's and Don'ts, edge cases, error handling                      │
│    └─► Prevents common failure modes                                │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. FEW-SHOT EXAMPLES (Optional)                                     │
│    Input-output pairs demonstrating desired behavior                │
│    └─► 20-30% accuracy boost for complex tasks                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# Basic structure
prompt = """
[System Context / Role]
[Task Description]
[Input Data]
[Output Format]
[Constraints / Guidelines]
[Examples (Few-shot)]
"""

# Example
prompt = """
You are an expert technical writer who creates clear, concise documentation.

Task: Summarize the following API documentation in 3 bullet points.

Input:
{api_documentation}

Output Format:
- [Bullet point 1]
- [Bullet point 2]
- [Bullet point 3]

Guidelines:
- Focus on key functionality
- Use simple, non-technical language
- Include practical use cases
"""
```

### Prompt Design Principles

✅ **Be specific**: Vague prompts → vague outputs
✅ **Provide context**: Role, task, constraints
✅ **Show examples**: Few-shot > zero-shot
✅ **Structure output**: Specify format (JSON, list, etc.)
✅ **Iterate**: Test and refine based on results

## 2. Chain-of-Thought (CoT) Prompting

**Paper**: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)

### Concept

Ask the model to show its reasoning step-by-step before answering. CoT transforms implicit reasoning into explicit steps, dramatically improving performance on multi-step problems.

```
┌─────────────────────────────────────────────────────────────────────┐
│              CHAIN-OF-THOUGHT REASONING FLOW                        │
└─────────────────────────────────────────────────────────────────────┘

  Without CoT:                    With CoT:
  ─────────────                   ──────────
  
  Question                        Question
      │                               │
      ▼                               ▼
  ┌─────────┐                    ┌─────────────┐
  │   LLM   │───► Answer         │     LLM     │
  │(Direct) │    (often wrong)   │ (Reasoning) │
  └─────────┘                    └─────────────┘
                                      │
                          ┌───────────┼───────────┐
                          ▼           ▼           ▼
                       Step 1      Step 2      Step 3
                          │           │           │
                          └───────────┴───────────┘
                                      │
                                      ▼
                                   Answer
                                (higher accuracy)
```

### Performance Benchmarks

| Dataset | Direct Prompting | CoT | Improvement |
|---------|------------------|-----|-------------|
| GSM8K (Math) | 17.7% | 40.7% | **+130%** |
| SVAMP (Math) | 69.9% | 79.0% | **+13%** |
| AQuA (Math) | 33.7% | 41.8% | **+24%** |
| StrategyQA | 54.3% | 62.1% | **+14%** |
| CommonsenseQA | 55.2% | 58.9% | **+7%** |

*Benchmarks on GPT-3.5 (175B parameters)

### Without CoT

```python
prompt = "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"

response = llm.generate(prompt)
# Output: "11 tennis balls" (often wrong!)
```

### With CoT

```python
prompt = """
Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

Let's think step by step:
"""

response = llm.generate(prompt)
# Output:
# "1. Roger starts with 5 tennis balls
#  2. He buys 2 cans, each with 3 balls
#  3. 2 cans × 3 balls = 6 new balls
#  4. 5 original + 6 new = 11 total
#  Answer: 11 tennis balls"
```

### Few-Shot CoT

```python
prompt = """
Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. How many eggs does she have left?

A: Let's think step by step.
Janet gets 16 eggs per day.
She eats 3 eggs for breakfast: 16 - 3 = 13
She uses 4 for muffins: 13 - 4 = 9
Answer: 9 eggs

Q: A store had 20 oranges. They sold 15 oranges and received a new shipment of 45 oranges. How many oranges do they have now?

A: Let's think step by step.
The store starts with 20 oranges.
They sold 15: 20 - 15 = 5
They received 45 new oranges: 5 + 45 = 50
Answer: 50 oranges

Q: {your_question}

A: Let's think step by step.
"""
```

### Zero-Shot CoT (Magic Phrase)

```python
# Simply add "Let's think step by step" to any prompt!
prompt = f"{question}\n\nLet's think step by step:"

# Works surprisingly well without examples
```

### When to Use CoT

✅ Math word problems
✅ Multi-step reasoning
✅ Logic puzzles
✅ Complex decision-making
❌ Simple factual questions (overkill)
❌ Creative writing (restricts creativity)

## 3. ReAct (Reasoning + Acting)

**Paper**: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)

### Concept

Interleave **reasoning traces** with **actions** (tool use, searches, etc.). ReAct enables LLMs to dynamically interact with external tools, APIs, and databases while maintaining coherent reasoning.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ReAct AGENT ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────┘

                        User Query
                            │
                            ▼
                    ┌───────────────┐
                    │  LLM Engine   │
                    │   (ReAct)     │
                    └───────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
         Thought       Action       Observation
              │             │             │
              │             ▼             │
              │      ┌─────────────┐     │
              │      │Tool Executor│     │
              │      └─────────────┘     │
              │             │             │
              │    ┌────────┼────────┐   │
              │    ▼        ▼        ▼   │
              │  Search  Calculate  API  │
              │    │        │        │   │
              │    └────────┴────────┘   │
              │             │             │
              └─────────────┴─────────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
            Continue Loop        Final Answer

  Typical Loop: 3-7 iterations
  Success Rate: 85-92% on HotPotQA
  Latency: 2-8 seconds (depends on tool calls)
```

### ReAct Pattern

```
Thought: [reasoning about what to do]
Action: [call a tool/function]
Observation: [result from action]
... (repeat until answer found)
Answer: [final answer]
```

### Production Use Cases

**1. Customer Support Agent**
```python
# Real-world example: E-commerce support
tools = {
    "check_order_status": lambda order_id: db.query(f"SELECT * FROM orders WHERE id={order_id}"),
    "search_kb": lambda query: knowledge_base.search(query),
    "update_shipping": lambda order_id, address: shipping_api.update(order_id, address),
    "refund_order": lambda order_id: payment_api.refund(order_id)
}

# Query: "I want to change the shipping address for order #12345"
# Thought: Need to check order status first
# Action: check_order_status(12345)
# Observation: Order is "Processing", can be modified
# Thought: Update the shipping address
# Action: update_shipping(12345, new_address)
# Observation: Success
# Answer: "I've updated your shipping address. Your order will be delivered to [new address]."
```

**2. Data Analysis Assistant**
```python
tools = {
    "sql_query": lambda q: database.execute(q),
    "plot_chart": lambda data, type: matplotlib.create(data, type),
    "calculate_stats": lambda data: numpy.stats(data)
}

# Query: "Show me top 5 customers by revenue this quarter"
# Thought: Need to query database for revenue data
# Action: sql_query("SELECT customer_id, SUM(revenue) FROM sales WHERE quarter='Q1' GROUP BY customer_id ORDER BY revenue DESC LIMIT 5")
# Observation: [customer data]
# Thought: Calculate total and create visualization
# Action: plot_chart(data, "bar")
# Answer: [Chart + summary]
```

### Implementation

```python
tools = {
    "search": lambda query: google_search(query),
    "calculate": lambda expr: eval(expr),
    "wikipedia": lambda topic: wikipedia.summary(topic)
}

prompt = """
Answer the following question using the tools available.

Question: What is the square root of the year the Eiffel Tower was completed?

You have access to these tools:
- search(query): Search Google
- calculate(expression): Perform calculations
- wikipedia(topic): Get Wikipedia summary

Use this format:
Thought: [your reasoning]
Action: [tool_name](argument)
Observation: [I will provide the result]
... (repeat as needed)
Answer: [final answer]

Begin!

Question: What is the square root of the year the Eiffel Tower was completed?

Thought:"""

# Model generates:
"""
Thought: I need to find when the Eiffel Tower was completed.
Action: wikipedia("Eiffel Tower")
Observation: The Eiffel Tower was completed in 1889...

Thought: Now I need to calculate the square root of 1889.
Action: calculate("1889 ** 0.5")
Observation: 43.464...

Thought: I have the answer now.
Answer: The square root of 1889 (completion year) is approximately 43.46.
"""
```

### ReAct with LangChain

```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("What is the population of Paris times 2?")

# Output shows ReAct loop:
# Thought: I need to find the population of Paris
# Action: Search[population of Paris]
# Observation: 2.1 million
# Thought: Now I need to multiply by 2
# Action: Calculator[2.1 * 2]
# Observation: 4.2
# Answer: 4.2 million
```

## 4. Self-Consistency

**Paper**: "Self-Consistency Improves Chain of Thought Reasoning" (Wang et al., 2022)

### Concept

Generate **multiple reasoning paths** and take the **majority vote**.

### Implementation

```python
def self_consistency(question, num_samples=5):
    prompt = f"{question}\n\nLet's think step by step:"
    
    answers = []
    for _ in range(num_samples):
        response = llm.generate(prompt, temperature=0.7)  # Non-zero temp for diversity
        answer = extract_final_answer(response)
        answers.append(answer)
    
    # Majority vote
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common

# Example
question = "If a train travels 120 km in 2 hours, how far will it travel in 5 hours?"

# Sample 1: 300 km (correct reasoning)
# Sample 2: 300 km (correct reasoning)
# Sample 3: 240 km (mistake in calculation)
# Sample 4: 300 km (correct reasoning)
# Sample 5: 300 km (correct reasoning)

# Result: 300 km (majority vote wins!)
```

### Benefits

- **10-20% accuracy improvement** over single-path CoT
- More robust to reasoning errors and model inconsistencies
- No additional training required
- Works across all model sizes

### Performance vs Cost Trade-off

```
┌─────────────────────────────────────────────────────────────────────┐
│         SELF-CONSISTENCY: ACCURACY vs COST ANALYSIS                 │
└─────────────────────────────────────────────────────────────────────┘

  Accuracy                         Cost per Request
     ▲                                  ▲
 90% │        ●────────────●         $0.025│              ●
     │      ●                             │            ●
 85% │    ●                          $0.020│          ●
     │  ●                                 │        ●
 80% │●                               $0.015│      ●
     │                                    │    ●
     └─────────────────────►             $0.010│  ●
      1  3  5  7  10                          │●
      Num Samples                              └─────────────────────►
                                                1  3  5  7  10
                                                Num Samples
  Optimal: 5 samples                      Optimal: 3-5 samples
  (diminishing returns after 7)           (balance cost/accuracy)
```

### Production Implementation with Caching

```python
import asyncio
from functools import lru_cache

class SelfConsistencyOptimized:
    def __init__(self, num_samples=5, cache_size=1000):
        self.num_samples = num_samples
        self.cache = {}
    
    @lru_cache(maxsize=1000)
    def get_cached_answer(self, question):
        """Cache common questions to avoid regenerating"""
        return self._generate_with_consistency(question)
    
    async def _generate_sample_async(self, question):
        """Generate single sample asynchronously"""
        prompt = f"{question}\n\nLet's think step by step:"
        response = await llm.generate_async(prompt, temperature=0.7)
        return self.extract_answer(response)
    
    async def generate_with_consistency(self, question):
        """Generate multiple samples in parallel"""
        # Check cache first
        cache_key = hash(question)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate samples in parallel (3x faster than sequential)
        tasks = [self._generate_sample_async(question) for _ in range(self.num_samples)]
        answers = await asyncio.gather(*tasks)
        
        # Majority vote with confidence scoring
        from collections import Counter
        answer_counts = Counter(answers)
        most_common_answer, count = answer_counts.most_common(1)[0]
        confidence = count / self.num_samples
        
        result = {
            "answer": most_common_answer,
            "confidence": confidence,
            "all_answers": answers
        }
        
        # Cache if high confidence
        if confidence >= 0.6:
            self.cache[cache_key] = result
        
        return result

# Usage
sc = SelfConsistencyOptimized(num_samples=5)
result = await sc.generate_with_consistency("If a train travels 120 km in 2 hours, how far will it travel in 5 hours?")

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.1%}")
# Output: Answer: 300 km, Confidence: 80%
```

### Cost Optimization Strategies

```python
def adaptive_sampling(question, min_samples=3, max_samples=10):
    """
    Start with min_samples, add more if disagreement is high
    """
    answers = []
    
    for i in range(max_samples):
        answer = generate_sample(question)
        answers.append(answer)
        
        if i >= min_samples - 1:
            # Check if we have clear majority
            counts = Counter(answers)
            most_common_count = counts.most_common(1)[0][1]
            agreement = most_common_count / len(answers)
            
            if agreement >= 0.7:  # 70% agreement threshold
                break  # Early stopping
    
    return Counter(answers).most_common(1)[0][0]

# Saves 30-40% on cost by stopping early when confidence is high
```

### Benchmarks on Common Tasks

| Task | Single-Path CoT | Self-Consistency (k=5) | Improvement | Cost Multiplier |
|------|----------------|----------------------|-------------|----------------|
| GSM8K | 40.7% | 54.5% | **+34%** | 5x |
| Math23K | 42.3% | 51.2% | **+21%** | 5x |
| AQuA | 41.8% | 49.3% | **+18%** | 5x |
| StrategyQA | 62.1% | 69.4% | **+12%** | 5x |

**ROI Analysis**: For high-value decisions (financial, medical, legal), the accuracy gain justifies 5x cost increase.

## 5. Tree-of-Thoughts (ToT)

**Paper**: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)

### Concept

Explore **multiple reasoning branches** like a search tree, backtrack when needed. ToT enables LLMs to perform **deliberate search** over problem-solving strategies, similar to human problem-solving.

```
┌─────────────────────────────────────────────────────────────────────┐
│              TREE-OF-THOUGHTS SEARCH STRATEGY                       │
└─────────────────────────────────────────────────────────────────────┘

                     [Initial Problem]
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         [Approach A]  [Approach B]  [Approach C]
         Score: 7      Score: 4      Score: 8
              │             ✗             │
         ┌────┴────┐              ┌───────┴───────┐
         ▼         ▼              ▼               ▼
     [Step A1] [Step A2]     [Step C1]       [Step C2]
     Score: 6   Score: 8     Score: 9        Score: 5
         ✗          │             │               ✗
              ┌─────┴─────┐       │
              ▼           ▼       ▼
          [A2.1]      [A2.2]  [C1.1]
          Score: 7    Score: 9 Score: 10 ← BEST PATH
              ✗           │        │
                          ▼        ▼
                    [Solution] [Solution*]

  Search Strategy: Breadth-First Search (BFS) or Best-First Search
  Evaluation: After each step, score promising-ness (1-10)
  Pruning: Discard branches with score < threshold (e.g., < 5)
  Backtracking: Return to higher-scoring branch if dead-end
```

### ToT vs Other Methods

| Method | Search Space | Backtracking | Best For |
|--------|--------------|--------------|----------|
| **Direct** | Single path | ❌ | Simple questions |
| **CoT** | Single path | ❌ | Multi-step reasoning |
| **Self-Consistency** | Multiple independent paths | ❌ | Error reduction |
| **ToT** | Tree exploration | ✅ | Complex planning, puzzles |

### ToT Pattern

```
              [Initial State]
               /     |     \
         [Step1A] [Step1B] [Step1C]
          /   \      |       /   \
     [2A1] [2A2]  [2B1]  [2C1] [2C2]
       ✓     ✗      ✗      ✓     ✗
```

### Production-Ready Implementation

```python
import heapq
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ThoughtNode:
    state: str
    depth: int
    score: float
    path: List[str]
    parent: 'ThoughtNode' = None
    
    def __lt__(self, other):
        return self.score > other.score  # Higher score = better

class TreeOfThoughts:
    def __init__(self, max_depth=4, beam_width=3, score_threshold=5.0):
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.score_threshold = score_threshold
        self.evaluation_cache = {}
    
    def evaluate_state(self, problem: str, state: str) -> float:
        """Score how promising this state is (1-10)"""
        cache_key = hash(f"{problem}:{state}")
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        prompt = f"""Evaluate the following reasoning step for solving this problem.
Rate from 1 (dead-end) to 10 (excellent progress).

Problem: {problem}

Reasoning step: {state}

Provide only a number between 1-10:"""
        
        response = llm.generate(prompt, temperature=0, max_tokens=5)
        try:
            score = float(response.strip())
            score = max(1.0, min(10.0, score))  # Clamp to [1, 10]
        except ValueError:
            score = 5.0  # Default neutral score
        
        self.evaluation_cache[cache_key] = score
        return score
    
    def generate_next_steps(self, problem: str, current_state: str) -> List[str]:
        """Generate possible next reasoning steps"""
        prompt = f"""Problem: {problem}

Current reasoning: {current_state}

Generate {self.beam_width} different next steps to solve this problem.
Number each step (1., 2., 3., ...):"""
        
        response = llm.generate(prompt, temperature=0.7, max_tokens=300)
        
        # Parse numbered steps
        steps = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit() and '.' in line:
                step = line.split('.', 1)[1].strip()
                if step:
                    steps.append(step)
        
        return steps[:self.beam_width]
    
    def solve(self, problem: str) -> Tuple[str, List[str]]:
        """
        Solve problem using Tree-of-Thoughts search
        
        Returns:
            (solution, reasoning_path)
        """
        # Initialize with root node
        root = ThoughtNode(
            state=problem,
            depth=0,
            score=5.0,  # Neutral starting score
            path=[problem]
        )
        
        # Priority queue for best-first search
        frontier = [root]
        heapq.heapify(frontier)
        
        best_solution = None
        best_score = -1
        nodes_explored = 0
        
        while frontier and nodes_explored < 100:  # Limit exploration
            # Pop best node
            current = heapq.heappop(frontier)
            nodes_explored += 1
            
            # Check if max depth reached
            if current.depth >= self.max_depth:
                if current.score > best_score:
                    best_score = current.score
                    best_solution = current
                continue
            
            # Generate and evaluate next steps
            next_steps = self.generate_next_steps(problem, current.state)
            
            for step in next_steps:
                score = self.evaluate_state(problem, step)
                
                # Prune low-scoring branches
                if score < self.score_threshold:
                    continue
                
                # Create child node
                child = ThoughtNode(
                    state=step,
                    depth=current.depth + 1,
                    score=score,
                    path=current.path + [step],
                    parent=current
                )
                
                heapq.heappush(frontier, child)
        
        if best_solution is None:
            return "No solution found", []
        
        return best_solution.state, best_solution.path

# Usage
tot = TreeOfThoughts(max_depth=4, beam_width=3)

problem = "Use numbers 4, 9, 10, 13 with +,-,*,/ to get 24"
solution, path = tot.solve(problem)

print("Solution:", solution)
print("\nReasoning path:")
for i, step in enumerate(path, 1):
    print(f"{i}. {step}")
```

### Benchmarks: ToT Performance

| Task | CoT | Self-Consistency | ToT | Improvement |
|------|-----|------------------|-----|-------------|
| Game of 24 | 7.3% | 9.0% | **74%** | **+900%** |
| Creative Writing | 12% | 15% | **56%** | **+367%** |
| Mini Crosswords | 43% | 48% | **78%** | **+62%** |

*Benchmarks using GPT-4

### Implementation

```python
def tree_of_thoughts(problem, depth=3, breadth=3):
    """
    Args:
        problem: Initial problem statement
        depth: How many steps to look ahead
        breadth: How many alternatives per step
    """
    
    def evaluate_state(state):
        """Score how promising this state is (1-10)"""
        prompt = f"""
Rate the following reasoning step on a scale of 1-10:
Problem: {problem}
Current step: {state}
Rating (1-10):"""
        rating = llm.generate(prompt)
        return float(rating)
    
    def generate_next_steps(state):
        """Generate possible next reasoning steps"""
        prompt = f"""
Problem: {problem}
Current reasoning: {state}
Generate {breadth} different next steps:"""
        steps = llm.generate(prompt).split('\n')
        return steps[:breadth]
    
    # Breadth-first search with pruning
    frontier = [(problem, 0, [])]  # (state, depth, path)
    best_solution = None
    best_score = -1
    
    while frontier:
        state, current_depth, path = frontier.pop(0)
        
        if current_depth >= depth:
            score = evaluate_state(state)
            if score > best_score:
                best_score = score
                best_solution = path + [state]
            continue
        
        # Generate and evaluate next steps
        next_steps = generate_next_steps(state)
        scored_steps = [(step, evaluate_state(step)) for step in next_steps]
        scored_steps.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top candidates
        for step, score in scored_steps[:breadth]:
            if score >= 5:  # Threshold
                frontier.append((step, current_depth + 1, path + [state]))
    
    return best_solution

# Example: Game of 24
problem = "Use numbers 4, 9, 10, 13 with +,-,*,/ to get 24"
solution = tree_of_thoughts(problem)
```

### Use Cases

- Complex puzzles (Game of 24, Sudoku)
- Strategic planning
- Creative writing (plot development)
- Code generation with backtracking

## 6. Few-Shot Prompting Best Practices

### Example Selection Strategies

#### a. Similar Examples (k-NN)

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Example pool
examples = [
    {"input": "2 + 2", "output": "4"},
    {"input": "10 * 5", "output": "50"},
    # ... 100 more examples
]

# Select most similar examples
def select_examples(query, k=3):
    query_embedding = model.encode(query)
    example_embeddings = model.encode([ex["input"] for ex in examples])
    
    similarities = util.cos_sim(query_embedding, example_embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:k]
    
    return [examples[i] for i in top_indices]

# Use in prompt
query = "15 / 3"
selected = select_examples(query, k=3)

prompt = "\n\n".join([f"Q: {ex['input']}\nA: {ex['output']}" for ex in selected])
prompt += f"\n\nQ: {query}\nA:"
```

#### b. Diverse Examples

```python
# Select examples covering different categories
examples_by_category = {
    "addition": [...],
    "subtraction": [...],
    "multiplication": [...],
    "division": [...]
}

selected = []
for category in examples_by_category:
    selected.append(random.choice(examples_by_category[category]))

# Ensures diverse coverage
```

### Optimal Number of Examples

| Task Complexity | Recommended Examples |
|----------------|---------------------|
| Simple classification | 3-5 |
| Complex reasoning | 5-10 |
| Code generation | 2-3 (detailed) |
| Creative writing | 1-2 (avoid constraining) |

## 7. Prompt Optimization Frameworks

### a. APE (Automatic Prompt Engineer)

Automatically generates and evaluates prompts.

```python
def ape_optimize(task_description, training_examples):
    # Step 1: Generate candidate prompts
    meta_prompt = f"""
Generate 10 different prompts for this task:
{task_description}

Examples:
{training_examples}

Prompts:"""
    
    candidate_prompts = llm.generate(meta_prompt).split('\n')
    
    # Step 2: Evaluate each prompt
    best_prompt = None
    best_score = 0
    
    for prompt in candidate_prompts:
        score = evaluate_prompt(prompt, training_examples)
        if score > best_score:
            best_score = score
            best_prompt = prompt
    
    return best_prompt

def evaluate_prompt(prompt, examples):
    correct = 0
    for example in examples:
        full_prompt = prompt + example["input"]
        output = llm.generate(full_prompt)
        if output.strip() == example["output"].strip():
            correct += 1
    return correct / len(examples)
```

### b. DSPy (Declarative Self-improving Python)

Framework for programmatic prompt optimization.

```python
import dspy

# Define task signature
class QA(dspy.Signature):
    """Answer questions based on context"""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

# Define program
class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(QA)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Compile (optimize prompts automatically!)
rag = RAG()
compiled_rag = dspy.Teleprompt().compile(rag, trainset=train_examples)

# Use
answer = compiled_rag("What is the capital of France?")
```

## 8. Structured Output Generation

### JSON Mode (OpenAI)

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs JSON."
        },
        {
            "role": "user",
            "content": "Extract person's name, age, and occupation: John is 30 and works as a doctor."
        }
    ]
)

print(response.choices[0].message.content)
# Output: {"name": "John", "age": 30, "occupation": "doctor"}
```

### Function Calling

```python
functions = [
    {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    functions=functions,
    function_call="auto"
)

# Model decides to call function
function_call = response.choices[0].message.function_call
print(function_call.name)  # "get_weather"
print(function_call.arguments)  # '{"location": "Paris", "unit": "celsius"}'
```

### Pydantic for Validation

```python
from pydantic import BaseModel, Field
from typing import List

class Person(BaseModel):
    name: str = Field(..., description="Person's full name")
    age: int = Field(..., ge=0, le=150)
    occupation: str
    skills: List[str]

prompt = f"""
Extract information in this JSON schema:
{Person.schema_json()}

Text: John Smith is a 30-year-old software engineer skilled in Python, Docker, and Kubernetes.

JSON:"""

response = llm.generate(prompt)
person = Person.parse_raw(response)  # Validates and parses
```

## 9. Prompt Security & Safety

### Prompt Injection Prevention

```python
def sanitize_input(user_input):
    # 1. Escape special characters
    user_input = user_input.replace('"', '\\"')
    
    # 2. Detect injection attempts
    injection_patterns = [
        "ignore previous instructions",
        "disregard all previous",
        "new instructions:",
        "system:",
    ]
    
    for pattern in injection_patterns:
        if pattern in user_input.lower():
            raise ValueError("Potential prompt injection detected")
    
    return user_input

# Use in prompt
safe_input = sanitize_input(user_input)
prompt = f"""
Process the following user query:

---
User query: {safe_input}
---

Response:"""
```

### Input Validation

```python
def validate_input(user_input, max_length=1000):
    # Length check
    if len(user_input) > max_length:
        raise ValueError(f"Input too long (max {max_length} chars)")
    
    # Content check
    if contains_malicious_content(user_input):
        raise ValueError("Malicious content detected")
    
    return True

# Malicious content detection
from transformers import pipeline

classifier = pipeline("text-classification", model="unitary/toxic-bert")

def contains_malicious_content(text):
    result = classifier(text)[0]
    return result['label'] == 'toxic' and result['score'] > 0.7
```

## 9.5. Troubleshooting Guide

### Common Issues & Solutions

#### Problem 1: Prompt Too Long (Context Window Exceeded)

**Symptoms**: `InvalidRequestError: maximum context length exceeded`

**Solutions**:

```python
# Strategy 1: Prompt Compression with LLMLingua
from llmlingua import PromptCompressor

compressor = PromptCompressor()
compressed = compressor.compress_prompt(
    long_prompt,
    target_ratio=0.5,  # Compress to 50% of original
    preserve_structure=True
)

# Strategy 2: Chunking and Map-Reduce
def process_long_document(document, query):
    chunks = split_document(document, chunk_size=2000)
    
    # Map: Summarize each chunk
    summaries = []
    for chunk in chunks:
        prompt = f"Summarize this section:\n{chunk}"
        summary = llm.generate(prompt)
        summaries.append(summary)
    
    # Reduce: Combine summaries and answer query
    combined = "\n\n".join(summaries)
    final_prompt = f"Based on these summaries:\n{combined}\n\nAnswer: {query}"
    return llm.generate(final_prompt)

# Strategy 3: Sliding Window
def sliding_window_search(document, query, window_size=2000, overlap=200):
    best_chunk = None
    best_score = -1
    
    for i in range(0, len(document), window_size - overlap):
        chunk = document[i:i+window_size]
        # Score relevance
        score = compute_relevance(chunk, query)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    
    return llm.generate(f"{query}\n\nContext: {best_chunk}")
```

#### Problem 2: Persistent Hallucinations

**Symptoms**: Model generates plausible but incorrect information

**Solutions**:

```python
# Solution 1: Grounding with Citations
prompt = """
Answer the question using ONLY information from the provided context.
Cite the specific part of the context you used for each statement.

Context:
{context}

Question: {question}

Format:
Answer: [your answer]
Citations: [relevant quotes from context]
"""

# Solution 2: Self-Verification Loop
def verify_answer(question, answer, context):
    verification_prompt = f"""
Question: {question}
Proposed Answer: {answer}
Context: {context}

Is this answer fully supported by the context? Answer Yes or No.
If No, list the unsupported claims:"""
    
    verification = llm.generate(verification_prompt)
    
    if "No" in verification:
        # Regenerate with stricter constraints
        return generate_conservative_answer(question, context)
    return answer

# Solution 3: Confidence Scoring
prompt = f"""
{question}

Provide your answer and rate your confidence (0-100%).

Format:
Answer: [your answer]
Confidence: [0-100]%
Reasoning: [why this confidence level]
"""

# Reject low-confidence answers (<70%)
```

#### Problem 3: High Cost

**Symptoms**: Monthly bill exceeding budget

**Solutions**:

```python
# Solution 1: Prompt Caching (OpenAI)
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": long_system_message},  # Cached
        {"role": "user", "content": user_query}
    ],
    # Cached tokens are 50-90% cheaper
)

# Solution 2: Smaller Model for Simple Queries
class AdaptiveRouter:
    def route_query(self, query):
        complexity = self.estimate_complexity(query)
        
        if complexity < 3:  # Simple
            return "gpt-3.5-turbo"  # $0.002/1K tokens
        elif complexity < 7:  # Medium
            return "gpt-4-turbo"     # $0.01/1K tokens
        else:  # Complex
            return "gpt-4"           # $0.03/1K tokens
    
    def estimate_complexity(self, query):
        # Check for reasoning keywords
        reasoning_keywords = ["calculate", "analyze", "compare", "step by step"]
        return sum(1 for kw in reasoning_keywords if kw in query.lower())

# Solution 3: Batch Processing
async def batch_process(queries, batch_size=10):
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        # Process batch in parallel
        batch_results = await asyncio.gather(*[
            process_query(q) for q in batch
        ])
        results.extend(batch_results)
    return results

# 5x faster, same cost
```

#### Problem 4: High Latency

**Symptoms**: Response time >5 seconds, poor UX

**Solutions**:

```python
# Solution 1: Streaming Responses
def stream_response(prompt):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
            # Display to user immediately

# Solution 2: Parallel Retrieval
async def parallel_rag(query):
    # Start retrieval and LLM call simultaneously
    retrieval_task = asyncio.create_task(retrieve_context(query))
    initial_response_task = asyncio.create_task(
        llm.generate(f"Provide a brief initial answer: {query}")
    )
    
    # Show initial response while waiting for context
    initial = await initial_response_task
    yield initial
    
    # Refine with full context
    context = await retrieval_task
    final = await llm.generate(f"Context: {context}\n\nQuery: {query}")
    yield final

# Solution 3: Smart Caching
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_generate(prompt_hash):
    return llm.generate(prompt_hash)

def generate_with_cache(prompt):
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    return cached_generate(prompt_hash)

# 70%+ cache hit rate → 90% latency reduction
```

#### Problem 5: Inconsistent Output Format

**Symptoms**: JSON parsing errors, unexpected response structure

**Solutions**:

```python
# Solution 1: Pydantic Validation with Retry
from pydantic import BaseModel, ValidationError
import json

class Response(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

def generate_with_validation(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = llm.generate(prompt)
            parsed = Response.parse_raw(response)
            return parsed
        except ValidationError as e:
            if attempt < max_retries - 1:
                # Add explicit format reminder
                prompt += f"\n\nIMPORTANT: Output must be valid JSON matching this schema:\n{Response.schema_json()}"
            else:
                raise

# Solution 2: Constrained Decoding (Guidance)
from guidance import models, gen

gpt = models.OpenAI("gpt-4")

response = gpt + f"""
{{{{"answer": "{gen(name='answer', max_tokens=100)}",
  "confidence": {gen(name='confidence', regex='[0-9]\\.[0-9]+')},
  "sources": [{gen(name='sources', list_append=True, stop=']')}]}}}}
"""

# Guaranteed valid JSON

# Solution 3: Post-Processing Fallback
def extract_json(response_text):
    # Try direct parsing
    try:
        return json.loads(response_text)
    except:
        pass
    
    # Try finding JSON in markdown code block
    import re
    match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    
    # Try finding any JSON object
    match = re.search(r'{.*}', response_text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    
    raise ValueError("No valid JSON found")
```

### Debug Checklist

```python
DEBUG_CHECKLIST = {
    "Prompt Issues": [
        "□ Is the prompt clear and specific?",
        "□ Are examples provided (few-shot)?",
        "□ Is the output format explicitly specified?",
        "□ Are constraints and guidelines listed?",
        "□ Is the prompt under token limit?",
    ],
    "Model Issues": [
        "□ Is the right model selected for the task?",
        "□ Is temperature appropriate (0 for deterministic, 0.7 for creative)?",
        "□ Are max_tokens sufficient?",
        "□ Is top_p/frequency_penalty tuned?",
    ],
    "Integration Issues": [
        "□ Are API credentials valid?",
        "□ Is error handling implemented?",
        "□ Are rate limits respected?",
        "□ Is retry logic in place?",
        "□ Are timeouts configured?",
    ],
    "Performance Issues": [
        "□ Is caching enabled?",
        "□ Are requests batched?",
        "□ Is streaming used for long responses?",
        "□ Are parallel calls used where possible?",
    ]
}
```

## 10. Evaluation & Monitoring

### Comprehensive Evaluation Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│              PROMPT EVALUATION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────┘

   Test Set (n=100-1000)
          │
          ▼
   ┌──────────────┐
   │  Automated   │
   │  Metrics     │───┐
   └──────────────┘   │
          │            │
          ▼            ▼
   ┌──────────────┐   ┌──────────────┐
   │  Accuracy    │   │   Latency    │
   │  Precision   │   │   Cost       │
   │  Recall      │   │   Tokens     │
   │  F1 Score    │   │   Errors     │
   └──────────────┘   └──────────────┘
          │                   │
          └─────────┬─────────┘
                    ▼
          ┌──────────────────┐
          │  Human Review    │
          │  (Sample 10%)    │
          └──────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │  Quality Score   │
          │  (Weighted avg)  │
          └──────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │  A/B Test        │
          │  (Production)    │
          └──────────────────┘
```

### Key Metrics

```python
from dataclasses import dataclass
from typing import List, Dict
import time

@dataclass
class PromptMetrics:
    """Comprehensive metrics for prompt evaluation"""
    accuracy: float          # Task-specific correctness
    precision: float         # Exactitude of positive predictions
    recall: float           # Coverage of positive cases
    f1_score: float         # Harmonic mean of precision/recall
    avg_latency: float      # Average response time (seconds)
    p95_latency: float      # 95th percentile latency
    avg_cost: float         # Average cost per request ($)
    token_efficiency: float # Output tokens / input tokens
    error_rate: float       # % of failed requests
    hallucination_rate: float  # % containing factual errors
    coherence_score: float  # Human-rated coherence (1-5)
    relevance_score: float  # Human-rated relevance (1-5)

class PromptEvaluator:
    def __init__(self, test_set: List[Dict]):
        self.test_set = test_set
        self.results = []
    
    def evaluate_prompt(self, prompt_template: str) -> PromptMetrics:
        """
        Comprehensive evaluation of a prompt template
        """
        correct = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        latencies = []
        costs = []
        tokens_in = []
        tokens_out = []
        errors = 0
        hallucinations = 0
        
        for test_case in self.test_set:
            # Format prompt
            prompt = prompt_template.format(**test_case['input'])
            
            # Measure latency
            start_time = time.time()
            try:
                response = llm.generate(prompt)
                latency = time.time() - start_time
                latencies.append(latency)
                
                # Calculate cost (example for GPT-4)
                input_tokens = count_tokens(prompt)
                output_tokens = count_tokens(response)
                cost = (input_tokens / 1000 * 0.03) + (output_tokens / 1000 * 0.06)
                costs.append(cost)
                
                tokens_in.append(input_tokens)
                tokens_out.append(output_tokens)
                
                # Evaluate correctness
                is_correct = self.check_correctness(response, test_case['expected'])
                if is_correct:
                    correct += 1
                    true_positives += 1
                else:
                    false_positives += 1
                
                # Check for hallucinations
                if self.contains_hallucination(response, test_case.get('facts', [])):
                    hallucinations += 1
                    
            except Exception as e:
                errors += 1
                latencies.append(0)
                costs.append(0)
        
        # Calculate metrics
        n = len(self.test_set)
        accuracy = correct / n if n > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return PromptMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_latency=sum(latencies) / len(latencies) if latencies else 0,
            p95_latency=sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            avg_cost=sum(costs) / len(costs) if costs else 0,
            token_efficiency=sum(tokens_out) / sum(tokens_in) if sum(tokens_in) > 0 else 0,
            error_rate=errors / n if n > 0 else 0,
            hallucination_rate=hallucinations / n if n > 0 else 0,
            coherence_score=0,  # Requires human evaluation
            relevance_score=0   # Requires human evaluation
        )
    
    def check_correctness(self, response: str, expected: str) -> bool:
        """Task-specific correctness check"""
        # For classification
        if isinstance(expected, str):
            return response.strip().lower() == expected.strip().lower()
        
        # For numerical answers
        if isinstance(expected, (int, float)):
            try:
                response_num = float(response.strip())
                return abs(response_num - expected) < 0.01
            except:
                return False
        
        # For free-form text (use semantic similarity)
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb1 = model.encode(response)
        emb2 = model.encode(expected)
        similarity = util.cos_sim(emb1, emb2).item()
        return similarity > 0.8  # 80% similarity threshold
    
    def contains_hallucination(self, response: str, facts: List[str]) -> bool:
        """Check if response contradicts known facts"""
        # Use NLI model to check for contradictions
        from transformers import pipeline
        nli = pipeline("text-classification", model="facebook/bart-large-mnli")
        
        for fact in facts:
            result = nli(f"{fact} [SEP] {response}")
            if result[0]['label'] == 'CONTRADICTION' and result[0]['score'] > 0.9:
                return True
        return False

# Usage
test_set = [
    {"input": {"question": "What is 2+2?"}, "expected": "4"},
    # ... 99 more test cases
]

evaluator = PromptEvaluator(test_set)

# Compare multiple prompts
prompts = [
    "Answer: {question}",
    "Let's solve step by step: {question}",
    "You are a math expert. {question}"
]

for i, prompt in enumerate(prompts, 1):
    metrics = evaluator.evaluate_prompt(prompt)
    print(f"\nPrompt {i}:")
    print(f"  Accuracy: {metrics.accuracy:.1%}")
    print(f"  F1 Score: {metrics.f1_score:.1%}")
    print(f"  Avg Latency: {metrics.avg_latency:.2f}s")
    print(f"  Avg Cost: ${metrics.avg_cost:.4f}")
    print(f"  Error Rate: {metrics.error_rate:.1%}")
```

### Production Monitoring Dashboard

```python
import prometheus_client
from datetime import datetime

class PromptMonitor:
    def __init__(self):
        # Prometheus metrics
        self.latency_histogram = prometheus_client.Histogram(
            'prompt_latency_seconds',
            'Prompt execution time',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.accuracy_gauge = prometheus_client.Gauge(
            'prompt_accuracy',
            'Current prompt accuracy'
        )
        
        self.cost_counter = prometheus_client.Counter(
            'prompt_cost_total',
            'Total cost of prompts'
        )
        
        self.error_counter = prometheus_client.Counter(
            'prompt_errors_total',
            'Total prompt errors',
            ['error_type']
        )
    
    def track_request(self, prompt: str, response: str, latency: float, cost: float, is_correct: bool):
        """Track individual request metrics"""
        self.latency_histogram.observe(latency)
        self.cost_counter.inc(cost)
        
        # Update accuracy (rolling average)
        # Store in time-series database (InfluxDB, Prometheus, etc.)
        
    def alert_if_degraded(self, metrics: PromptMetrics, thresholds: Dict):
        """Send alerts if metrics degrade"""
        alerts = []
        
        if metrics.accuracy < thresholds.get('min_accuracy', 0.8):
            alerts.append(f"⚠️ Accuracy dropped to {metrics.accuracy:.1%}")
        
        if metrics.avg_latency > thresholds.get('max_latency', 3.0):
            alerts.append(f"⚠️ Latency increased to {metrics.avg_latency:.2f}s")
        
        if metrics.error_rate > thresholds.get('max_error_rate', 0.05):
            alerts.append(f"⚠️ Error rate: {metrics.error_rate:.1%}")
        
        if alerts:
            self.send_alerts(alerts)
    
    def send_alerts(self, alerts: List[str]):
        """Send alerts via Slack, PagerDuty, etc."""
        # Integration with alerting systems
        pass
```

## 11. Prompt Engineering Workflow

### 1. Define Success Criteria

```python
success_criteria = {
    "accuracy": 0.90,          # 90% correct answers
    "latency": 2.0,            # < 2 seconds
    "cost": 0.01,              # < $0.01 per request
    "user_satisfaction": 4.5   # > 4.5/5 rating
}
```

### 2. Create Test Set

```python
test_set = [
    {
        "input": "What is 2+2?",
        "expected_output": "4",
        "category": "math"
    },
    # ... 50-100 more examples
]
```

### 3. Iterate and Evaluate

```python
prompts_to_test = [
    "Answer the question: {question}",
    "You are a helpful assistant. {question}",
    "Solve: {question}\n\nLet's think step by step:"
]

for prompt_template in prompts_to_test:
    results = []
    for test in test_set:
        prompt = prompt_template.format(question=test["input"])
        output = llm.generate(prompt)
        correct = evaluate(output, test["expected_output"])
        results.append(correct)
    
    accuracy = sum(results) / len(results)
    print(f"Prompt: {prompt_template[:50]}... | Accuracy: {accuracy:.2%}")
```

### 4. A/B Test in Production

```python
import random

def route_request(user_query):
    if random.random() < 0.5:
        # Variant A: Original prompt
        return generate_with_prompt_a(user_query)
    else:
        # Variant B: New prompt
        return generate_with_prompt_b(user_query)

# Track metrics
# After 1000 requests:
# - Variant A: 85% satisfaction, $0.015/request
# - Variant B: 90% satisfaction, $0.012/request
# → Deploy Variant B
```

## 11. Prompt Templates Library

### Production-Ready Templates

#### 1. Multi-Class Classification

```python
classification_prompt = """
You are an expert classifier with high accuracy.

Task: Classify the following text into ONE of these categories:
{categories}

Text to classify:
"""
{text}
"""

Provide your answer in this exact format:
Category: [category name]
Confidence: [0-100]%
Reasoning: [brief explanation]

Category:"""

# Advanced: Few-shot classification
few_shot_classification = """
Classify customer feedback as: Positive, Negative, or Neutral

Examples:
Text: "The product exceeded my expectations!"
Category: Positive

Text: "Terrible customer service, very disappointed."
Category: Negative

Text: "The item arrived on time."
Category: Neutral

Now classify:
Text: "{text}"
Category:"""
```

#### 2. Advanced Summarization

```python
summarization_prompt = """
Create a {summary_type} summary of the following text.

Requirements:
- Length: {num_sentences} sentences (strict)
- Style: {style}  # e.g., "technical", "executive", "casual"
- Focus on: {focus_areas}  # e.g., "key findings, action items"
- Exclude: {exclude}  # e.g., "background information, examples"

Text:
"""
{text}
"""

Summary:"""

# Bullet-point summary
bullet_summary = """
Summarize this article as bullet points:

{text}

Provide 5-7 bullet points covering:
• Main topic/thesis
• Key arguments or findings  
• Supporting evidence
• Conclusions
• Implications or recommendations

Bullet points:"""

# Chain-of-density summary (iterative refinement)
chain_of_density = """
Create a summary in 3 iterations, each denser than the last:

Iteration 1 (Brief): {max_words} words
Iteration 2 (Detailed): {max_words*2} words  
Iteration 3 (Comprehensive): {max_words*3} words

Text: {text}

Provide all three iterations:"""
```

#### 3. Named Entity Recognition & Extraction

```python
extraction_prompt = """
Extract the following entities from the text and return as JSON:

Entities to extract:
- person_names: List of people mentioned
- organizations: Companies, institutions
- locations: Cities, countries, addresses
- dates: All dates and time references
- monetary_values: Money amounts with currency
- key_metrics: Numbers with context

Text:
{text}

JSON (use null for missing entities):
{{
  "person_names": [],
  "organizations": [],
  "locations": [],
  "dates": [],
  "monetary_values": [],
  "key_metrics": []
}}

Extracted JSON:"""

# Relationship extraction
relationship_extraction = """
Extract relationships between entities:

Text: {text}

For each relationship found, provide:
{{
  "subject": "entity1",
  "predicate": "relationship_type",
  "object": "entity2",
  "confidence": 0.95
}}

Relationships (JSON array):"""
```

#### 4. Code Generation with Tests

```python
code_generation_prompt = """
You are an expert {language} developer following best practices.

Task: Write a {language} function that {description}.

Requirements:
{requirements}

Provide:
1. Function signature with type hints
2. Complete implementation with error handling
3. Docstring (Google style)
4. 3-5 unit tests covering edge cases
5. Example usage
6. Time/space complexity analysis

Function:"""

# Advanced: Code with optimization
optimized_code_prompt = """
Implement {description} in {language}.

Provide TWO solutions:

1. Simple/Readable version:
   - Easy to understand
   - Good for maintainability
   
2. Optimized version:
   - Best time complexity
   - Best space complexity
   - Production-ready

For each, include:
- Code with comments
- Complexity analysis
- When to use this version

Code:"""
```

#### 5. Question Answering

```python
qa_prompt = """
Answer the question based ONLY on the provided context.
If the answer cannot be found in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Provide:
1. Answer: [concise answer]
2. Confidence: [High/Medium/Low]
3. Supporting quote: [relevant excerpt from context]
4. Additional context: [optional relevant details]

Answer:"""

# Multi-hop reasoning
multi_hop_qa = """
Answer this question that requires combining information from multiple sources.

Sources:
{sources}

Question: {question}

Thinking process:
1. What information do I need?
2. Where can I find each piece?
3. How do they connect?
4. What's the final answer?

Answer:"""
```

#### 6. Content Generation

```python
# Blog post generator
blog_post_prompt = """
Write a blog post about: {topic}

Audience: {audience}
Tone: {tone}  # e.g., "professional", "casual", "humorous"
Length: {word_count} words

Structure:
1. Attention-grabbing title
2. Hook (first paragraph)
3. Main points (3-5 sections with subheadings)
4. Practical examples or case studies
5. Actionable takeaways
6. Call-to-action

SEO keywords to include: {keywords}

Blog post:"""

# Email templates
email_prompt = """
Write a {email_type} email:

Recipient: {recipient_role}
Purpose: {purpose}
Tone: {tone}
Length: {length}  # e.g., "brief", "detailed"

Key points to cover:
{key_points}

Include:
- Subject line (compelling, <50 chars)
- Greeting
- Body (clear, concise)
- Call-to-action
- Professional sign-off

Email:"""
```

#### 7. Data Analysis

```python
data_analysis_prompt = """
Analyze this dataset and provide insights:

Data:
{data}

Analysis tasks:
1. Descriptive statistics (mean, median, std, min, max)
2. Identify trends and patterns
3. Detect anomalies or outliers
4. Correlation analysis
5. Key insights and recommendations

Present findings with:
- Executive summary (2-3 sentences)
- Detailed analysis
- Visualizations to create (describe)
- Actionable recommendations

Analysis:"""

# SQL query generation
sql_generation = """
Generate a SQL query for this request:

Database schema:
{schema}

Request: {natural_language_query}

Provide:
1. SQL query (PostgreSQL syntax)
2. Explanation of what it does
3. Expected output format
4. Potential performance considerations
5. Alternative query approaches (if any)

SQL:"""
```

#### 8. Translation with Context

```python
translation_prompt = """
Translate the following text from {source_lang} to {target_lang}.

Context: {context}  # e.g., "legal document", "marketing copy"
Style: {style}  # e.g., "formal", "colloquial"

Important:
- Preserve tone and intent
- Maintain formatting
- Keep technical terms in original language if appropriate
- Flag any ambiguous phrases

Text to translate:
{text}

Translation:
"""

Provide:
1. Translation
2. Confidence level
3. Notes on challenging phrases
4. Alternative translations (if applicable)

Translation:"""
```

#### 9. Sentiment Analysis (Detailed)

```python
sentiment_analysis = """
Perform detailed sentiment analysis on this text:

Text: {text}

Analyze:
1. Overall sentiment: Positive/Negative/Neutral/Mixed (-100 to +100 scale)
2. Emotion breakdown: Joy, Anger, Sadness, Fear, Surprise (0-100% each)
3. Subjectivity: Objective/Subjective (0-100% subjective)
4. Intensity: Weak/Moderate/Strong
5. Key sentiment-carrying phrases
6. Sentiment by aspect (if multiple topics discussed)

Provide as JSON:
{{
  "overall_sentiment": 75,
  "label": "Positive",
  "emotions": {{}},
  "subjectivity": 60,
  "intensity": "Strong",
  "key_phrases": [],
  "aspect_sentiments": {{}}
}}

Analysis:"""
```

#### 10. Debate/Argument Analysis

```python
argument_analysis = """
Analyze the logical structure of this argument:

Argument:
{text}

Provide:
1. Main claim/thesis
2. Supporting premises (numbered)
3. Evidence provided for each premise
4. Logical fallacies (if any)
5. Counterarguments addressed (if any)
6. Strength assessment (1-10)
7. Missing evidence or gaps

Structured analysis:"""
```

### Template Selection Guide

```python
def select_template(task_type, complexity, domain):
    template_map = {
        ("classification", "simple", "general"): classification_prompt,
        ("classification", "complex", "general"): few_shot_classification,
        ("extraction", "entities", "general"): extraction_prompt,
        ("extraction", "relationships", "general"): relationship_extraction,
        ("generation", "code", "software"): code_generation_prompt,
        ("qa", "simple", "general"): qa_prompt,
        ("qa", "complex", "general"): multi_hop_qa,
        # ... add more mappings
    }
    return template_map.get((task_type, complexity, domain))
```

## 12. Common Pitfalls & Anti-Patterns

### ❌ Pitfall 1: Vague or Ambiguous Prompts

**Bad**:
```python
prompt = "Tell me about Python"
# Too vague: Python the language? The snake? Monty Python?
```

**Good**:
```python
prompt = """
Explain the key features of Python programming language that make it 
suitable for data science, with specific examples of libraries and use cases.
"""
```

**Fix**: Be specific about context, constraints, and expected output.

---

### ❌ Pitfall 2: Forgetting Output Format

**Bad**:
```python
prompt = "Extract the person's name and age from: John is 30 years old."
# Returns: "The person's name is John and he is 30 years old."
# Hard to parse!
```

**Good**:
```python
prompt = """
Extract person's name and age from: John is 30 years old.

Return as JSON:
{"name": "...", "age": ...}

JSON:"""
# Returns: {"name": "John", "age": 30}
```

**Fix**: Always specify exact output format (JSON, CSV, specific structure).

---

### ❌ Pitfall 3: Ignoring Context/Domain

**Bad**:
```python
prompt = "Is this review positive? 'The movie was sick!'"
# Ambiguous: "sick" can mean great or terrible depending on context
```

**Good**:
```python
prompt = """
You are analyzing movie reviews from Gen-Z audience where slang is common.
"sick" = very good, "mid" = mediocre, "trash" = bad.

Is this review positive, negative, or neutral?
Review: "The movie was sick!"

Sentiment:"""
```

**Fix**: Provide domain context, definitions, and cultural nuances.

---

### ❌ Pitfall 4: Over-Engineering Simple Tasks

**Bad**:
```python
# Using Tree-of-Thoughts for simple classification
prompt = """
Classify this as spam or not spam: "Buy now!"

Explore multiple reasoning paths:
1. Linguistic analysis
2. Pattern matching
3. Contextual inference
...
"""
# Overkill, expensive, slow!
```

**Good**:
```python
prompt = "Is this spam? 'Buy now!' Answer: Yes or No."
# Simple, fast, cheap
```

**Fix**: Start simple. Add complexity only when needed.

---

### ❌ Pitfall 5: Not Providing Examples (When Needed)

**Bad**:
```python
prompt = "Extract product features from this description."
# Model doesn't know what format you want
```

**Good**:
```python
prompt = """
Extract product features from descriptions.

Example 1:
Input: "Laptop with 16GB RAM, 512GB SSD, Intel i7"
Output: {"ram": "16GB", "storage": "512GB SSD", "cpu": "Intel i7"}

Example 2:
Input: "Phone with 6.5 inch screen, 5G, dual camera"
Output: {"screen": "6.5 inch", "connectivity": "5G", "camera": "dual camera"}

Now extract features from:
Input: {product_description}
Output:"""
```

**Fix**: Use few-shot examples for complex or non-standard tasks.

---

### ❌ Pitfall 6: Ignoring Token Limits

**Bad**:
```python
prompt = very_long_document + "\n\nSummarize this."
# Error: context_length_exceeded
```

**Good**:
```python
def safe_summarize(document, max_tokens=6000):
    if count_tokens(document) > max_tokens:
        # Strategy: chunk and summarize iteratively
        chunks = split_document(document, max_tokens)
        summaries = [summarize(chunk) for chunk in chunks]
        return summarize("\n".join(summaries))
    return summarize(document)
```

**Fix**: Check token count before sending. Use chunking strategies.

---

### ❌ Pitfall 7: Not Handling Edge Cases

**Bad**:
```python
prompt = f"Calculate: {user_input}"
# user_input = "" → confused model
# user_input = "destroy all humans" → inappropriate response
```

**Good**:
```python
def safe_calculate(user_input):
    if not user_input.strip():
        return "Error: Empty input"
    
    if not is_valid_math_expression(user_input):
        return "Error: Invalid math expression"
    
    prompt = f"""
Calculate the following mathematical expression:
{user_input}

Provide only the numerical result.
Result:"""
    return llm.generate(prompt)
```

**Fix**: Validate inputs, handle empty strings, check for malicious content.

---

### ❌ Pitfall 8: Inconsistent Prompt Structure

**Bad**:
```python
# Different formats for similar tasks
prompt1 = "Translate: {text}"
prompt2 = "Text to translate: {text}\nTranslation:"
prompt3 = "{text} <- translate this"
# Inconsistent performance!
```

**Good**:
```python
# Standardized template
TRANSLATION_TEMPLATE = """
Translate from {source_lang} to {target_lang}:

Text: {text}

Translation:"""

# Use consistently
prompt = TRANSLATION_TEMPLATE.format(
    source_lang="English",
    target_lang="French",
    text=text
)
```

**Fix**: Create reusable templates. Maintain consistency.

---

### ❌ Pitfall 9: No Error Handling

**Bad**:
```python
response = llm.generate(prompt)
result = json.loads(response)  # Crashes if invalid JSON
```

**Good**:
```python
try:
    response = llm.generate(prompt)
    result = json.loads(response)
except json.JSONDecodeError:
    # Retry with more explicit instructions
    prompt += "\n\nIMPORTANT: Return ONLY valid JSON, no explanations."
    response = llm.generate(prompt)
    result = json.loads(response)
except Exception as e:
    logger.error(f"LLM generation failed: {e}")
    return default_response
```

**Fix**: Always handle API errors, parsing failures, timeouts.

---

### ❌ Pitfall 10: Not Testing Prompts

**Bad**:
```python
# Deploy to production immediately
prompt = "New experimental prompt"
deploy_to_production(prompt)
```

**Good**:
```python
# Test suite
test_cases = [
    {"input": "test1", "expected": "result1"},
    {"input": "test2", "expected": "result2"},
    # ... 50 more cases
]

accuracy = evaluate_prompt(prompt, test_cases)

if accuracy > 0.90:
    # A/B test with 10% traffic
    deploy_with_ab_test(prompt, traffic_percentage=0.1)
else:
    print(f"Accuracy too low: {accuracy:.1%}")
```

**Fix**: Create test suite, measure performance, A/B test before full rollout.

---

### Quick Anti-Pattern Checklist

```python
ANTI_PATTERNS = {
    "⚠️ Avoid These": [
        "□ Vague instructions without context",
        "□ No output format specification",
        "□ Forgetting domain/audience context",
        "□ Using complex techniques for simple tasks",
        "□ No examples for non-trivial tasks",
        "□ Ignoring token limits",
        "□ No edge case handling",
        "□ Inconsistent prompt structures",
        "□ No error handling or retries",
        "□ Deploying without testing",
        "□ Not monitoring production performance",
        "□ Hardcoding prompts (use templates instead)",
        "□ Assuming first version is optimal",
        "□ Not measuring cost/latency/accuracy",
    ]
}
```

### Best Practice Summary

| Instead of... | Do this... |
|---------------|------------|
| "Tell me about X" | "Explain X focusing on Y and Z, in 3 paragraphs for [audience]" |
| No output format | "Return as JSON: {\"key\": \"value\"}" |
| Complex technique first | Start simple, add complexity if needed |
| No examples | Provide 2-3 few-shot examples |
| Ignoring errors | Try-except with retry logic |
| Deploying blind | Test suite + A/B test |
| One-size-fits-all | Adaptive routing based on complexity |
| Static prompts | Version control + continuous optimization |

## 13. Model Comparison for Prompt Techniques

### Performance by Model & Technique

```
┌────────────────────────────────────────────────────────────────────────────┐
│                  MODEL PERFORMANCE COMPARISON (2024-2025)                 │
└────────────────────────────────────────────────────────────────────────────┘
```

| Technique | GPT-4 | GPT-3.5-Turbo | Claude 3 Opus | Claude 3 Sonnet | Gemini 1.5 Pro |
|-----------|-------|---------------|---------------|-----------------|----------------|
| **Zero-Shot** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Few-Shot** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Chain-of-Thought** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **ReAct (Tool Use)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Self-Consistency** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Tree-of-Thoughts** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Structured Output** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Long Context** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Detailed Benchmarks

#### Math Reasoning (GSM8K)

| Model | Direct | CoT | Self-Consistency | Cost/1K |
|-------|--------|-----|------------------|----------|
| **GPT-4** | 68% | 92% | 95% | $0.03 |
| **GPT-3.5-Turbo** | 18% | 41% | 55% | $0.002 |
| **Claude 3 Opus** | 70% | 95% | 96% | $0.015 |
| **Claude 3 Sonnet** | 62% | 88% | 92% | $0.003 |
| **Gemini 1.5 Pro** | 65% | 90% | 94% | $0.0035 |

**Recommendation**: Claude 3 Opus for highest accuracy, GPT-3.5-Turbo for budget.

#### Code Generation (HumanEval)

| Model | Pass@1 | Pass@10 | With Tests | Cost/1K |
|-------|--------|---------|------------|----------|
| **GPT-4** | 86% | 95% | 92% | $0.03 |
| **GPT-3.5-Turbo** | 48% | 72% | 65% | $0.002 |
| **Claude 3 Opus** | 84% | 94% | 90% | $0.015 |
| **Claude 3 Sonnet** | 73% | 88% | 82% | $0.003 |
| **Gemini 1.5 Pro** | 71% | 86% | 79% | $0.0035 |

**Recommendation**: GPT-4 or Claude Opus for production code.

#### Long Context (>100K tokens)

| Model | Max Context | Accuracy @ 100K | Latency | Cost |
|-------|-------------|-----------------|---------|------|
| **GPT-4 Turbo** | 128K | 85% | 15s | High |
| **Claude 3 Opus** | 200K | 92% | 20s | High |
| **Claude 3 Sonnet** | 200K | 88% | 12s | Medium |
| **Gemini 1.5 Pro** | 1M | 95% | 25s | Medium |

**Recommendation**: Gemini 1.5 Pro for very long contexts, Claude Opus for accuracy.

### Use Case → Model Mapping

```python
MODEL_RECOMMENDATIONS = {
    "math_reasoning": {
        "best_accuracy": "claude-3-opus",
        "best_value": "claude-3-sonnet",
        "budget": "gpt-3.5-turbo"
    },
    "code_generation": {
        "best_accuracy": "gpt-4",
        "best_value": "claude-3-sonnet",
        "budget": "gpt-3.5-turbo"
    },
    "long_context": {
        "best_accuracy": "gemini-1.5-pro",
        "best_value": "claude-3-sonnet",
        "budget": "claude-3-haiku"
    },
    "tool_use": {
        "best_accuracy": "gpt-4",
        "best_value": "claude-3-opus",
        "budget": "gpt-3.5-turbo"
    },
    "creative_writing": {
        "best_accuracy": "claude-3-opus",
        "best_value": "gpt-4",
        "budget": "claude-3-sonnet"
    },
    "structured_output": {
        "best_accuracy": "gpt-4",
        "best_value": "claude-3-opus",
        "budget": "gpt-3.5-turbo"
    },
    "multilingual": {
        "best_accuracy": "gpt-4",
        "best_value": "gemini-1.5-pro",
        "budget": "gpt-3.5-turbo"
    }
}

def select_model(use_case, priority="best_value"):
    """Select optimal model based on use case and priority"""
    return MODEL_RECOMMENDATIONS.get(use_case, {}).get(priority, "gpt-4")

# Example
model = select_model("math_reasoning", priority="best_accuracy")
print(model)  # "claude-3-opus"
```

### Cost Optimization by Model

```python
# Hybrid approach: route by complexity
class ModelRouter:
    def __init__(self):
        self.models = {
            "simple": "gpt-3.5-turbo",      # $0.002/1K
            "medium": "claude-3-sonnet",     # $0.003/1K  
            "complex": "gpt-4",              # $0.03/1K
            "very_complex": "claude-3-opus" # $0.015/1K
        }
    
    def route(self, query, task_type):
        complexity = self.assess_complexity(query, task_type)
        return self.models[complexity]
    
    def assess_complexity(self, query, task_type):
        # Simple heuristics
        if task_type in ["classification", "sentiment"]:
            return "simple"
        
        if len(query.split()) < 50 and task_type != "reasoning":
            return "simple"
        
        if task_type in ["math", "code", "reasoning"]:
            if "step by step" in query or "explain" in query:
                return "complex"
            return "medium"
        
        return "medium"

# Usage
router = ModelRouter()
model = router.route("What is 2+2?", "math")
# Returns: "gpt-3.5-turbo" (simple query)

model = router.route("Solve this calculus problem step by step...", "math")
# Returns: "gpt-4" (complex query)
```

### When to Use Which Model

| Scenario | Recommended Model | Why |
|----------|------------------|-----|
| **High-volume, simple tasks** | GPT-3.5-Turbo | Lowest cost, good enough |
| **Critical accuracy** | Claude 3 Opus or GPT-4 | Best performance |
| **Long documents (>50K tokens)** | Gemini 1.5 Pro | Largest context window |
| **Balanced cost/performance** | Claude 3 Sonnet | Sweet spot |
| **Creative writing** | Claude 3 Opus | Most natural, nuanced |
| **Structured data extraction** | GPT-4 with JSON mode | Best structured output |
| **Multilingual** | GPT-4 or Gemini | Best language coverage |
| **Real-time, low latency** | GPT-3.5-Turbo | Fastest response |
| **Research/exploration** | Claude 3 Opus | Thoughtful, detailed responses |

## Conclusion

Advanced prompt engineering techniques dramatically improve LLM performance while reducing costs and latency:

### Performance Summary

| Technique | Accuracy Gain | Cost Impact | Latency | Complexity | Production Ready |
|-----------|---------------|-------------|---------|------------|------------------|
| **Chain-of-Thought** | +15-40% | +10% tokens | +20% | Low | ✅ |
| **Self-Consistency** | +10-20% | +400% (5x) | +5x | Medium | ✅ |
| **ReAct** | +25-45% | Variable | +3-8x | High | ✅ |
| **Tree-of-Thoughts** | +60-900%* | +10-50x | +10-20x | Very High | ⚠️ |
| **Few-Shot (optimized)** | +20-35% | +5-15% | +10% | Low | ✅ |
| **Structured Output** | +30-50% | -40%** | -20% | Low | ✅ |

*On specific tasks like Game of 24  
**Reduction from fewer retries

### Key Takeaways

1. **Start Simple, Scale Smart**: Begin with zero-shot, add few-shot examples, then CoT only if needed
2. **Measure Everything**: Track accuracy, latency, cost, and user satisfaction continuously
3. **Use the Right Tool**: CoT for reasoning, ReAct for tools, Self-Consistency for high-stakes
4. **Structure Outputs**: JSON mode and function calling reduce parsing errors by 80%
5. **Iterate with Data**: A/B test prompts in production, optimize based on real user behavior
6. **Cache Aggressively**: 70%+ cache hit rate can reduce costs by 60-80%
7. **Monitor & Alert**: Set thresholds for accuracy, latency, cost; alert on degradation
8. **Security First**: Validate inputs, prevent injection, rate limit, audit logs

### ROI Analysis

```python
# Example: Customer support bot
baseline = {
    "accuracy": 0.65,
    "cost_per_query": 0.02,
    "queries_per_month": 100000,
    "human_escalation_rate": 0.35,
    "human_cost_per_escalation": 5.00
}

optimized = {
    "accuracy": 0.88,  # +35% with CoT + few-shot
    "cost_per_query": 0.025,  # +25% prompt cost
    "queries_per_month": 100000,
    "human_escalation_rate": 0.12,  # -66% escalations
    "human_cost_per_escalation": 5.00
}

# Baseline monthly cost
baseline_llm_cost = baseline['queries_per_month'] * baseline['cost_per_query']
baseline_human_cost = baseline['queries_per_month'] * baseline['human_escalation_rate'] * baseline['human_cost_per_escalation']
baseline_total = baseline_llm_cost + baseline_human_cost
print(f"Baseline: ${baseline_total:,.0f}/month (LLM: ${baseline_llm_cost:,.0f}, Human: ${baseline_human_cost:,.0f})")

# Optimized monthly cost
optimized_llm_cost = optimized['queries_per_month'] * optimized['cost_per_query']
optimized_human_cost = optimized['queries_per_month'] * optimized['human_escalation_rate'] * optimized['human_cost_per_escalation']
optimized_total = optimized_llm_cost + optimized_human_cost
print(f"Optimized: ${optimized_total:,.0f}/month (LLM: ${optimized_llm_cost:,.0f}, Human: ${optimized_human_cost:,.0f})")

# ROI
savings = baseline_total - optimized_total
roi_percentage = (savings / baseline_total) * 100
print(f"\nMonthly Savings: ${savings:,.0f} ({roi_percentage:.1f}% reduction)")
print(f"Annual Savings: ${savings * 12:,.0f}")

# Output:
# Baseline: $177,000/month (LLM: $2,000, Human: $175,000)
# Optimized: $62,500/month (LLM: $2,500, Human: $60,000)
# Monthly Savings: $114,500 (64.7% reduction)
# Annual Savings: $1,374,000
```

### 2024-2025 Trends & Future Directions

**1. Prompt Caching (OpenAI, Anthropic)**
- Cache system messages and context
- 50-90% cost reduction for repeated queries
- Sub-10ms latency for cached prompts

**2. Constitutional AI Prompting**
- Self-critique and self-improvement loops
- "Before answering, check if this violates [principles]"
- 40% reduction in harmful outputs

**3. Multimodal Prompting**
- Text + image + audio in single prompt
- "Analyze this chart and explain the trend"
- Vision-language models (GPT-4V, Gemini Pro Vision)

**4. Meta-Prompting**
- LLM generates optimal prompts for other LLMs
- "Generate a prompt to extract product specs from reviews"
- 20-30% better than human-written prompts

**5. Prompt Compression**
- Compress long contexts into dense representations
- LLMLingua, AutoCompressors
- 80% token reduction, <5% accuracy loss

**6. Adaptive Prompting**
- Dynamic prompt selection based on query complexity
- Simple queries → zero-shot, Complex → CoT + tools
- Optimal cost/accuracy trade-off

### Mastering Prompt Engineering Enables You To

✅ **Extract Maximum Value**: Get GPT-3.5 to perform like GPT-4 at 10x lower cost  
✅ **Reduce Hallucinations**: From 30% to <5% with structured prompting  
✅ **Build Reliable Systems**: 99.9% uptime with proper error handling  
✅ **Optimize Costs**: 60-80% reduction through caching and efficient prompting  
✅ **Scale Confidently**: Handle 1M+ requests/day with monitoring and optimization  
✅ **Competitive Advantage**: Better prompts = better products = happier users  

### Final Thought

> "The difference between a $10K/month LLM bill and a $2K/month bill is often just better prompt engineering. The difference between 70% accuracy and 95% accuracy is the same thing."

**Remember**: Great prompt engineering can make a smaller model outperform a larger one at a fraction of the cost. Master these techniques, measure rigorously, iterate continuously, and you'll build LLM applications that delight users while staying within budget.


