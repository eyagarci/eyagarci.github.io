---
layout: post
comments: true
title:  "Introduction to Tensorflow - 01 Basics"
excerpt: "Florens Greßner - While giving an overview of Python, NumPy and 
TensorFlow I want to create a basic understanding of TensorFlows graph 
representation for the reader."
date:   2017-08-11
mathjax: true
---


## Introduction
We will explore the different approaches to programming using Python, NumPy and 
TensorFlow. This won't be a beginner tutorial that talks about variable 
initialization, loops, function, classes, library calls and saying hello to a
world.

I want to give you a framework for how to think about the different approaches 
such that you can use the provided tools on your own.

Learn how to read documentations and find all tools needed. Think
about your problems and search for the gadgets that are needed for 
solving the problem. Don't only think about the tools that you already know and try to figure 
out a solution with static knowledge. You may be successful upto a certain 
point but your code can never be efficient and most importantly - beautiful. 
**Own your code!**  
Please refer to the following documentations:
- [python docs](https://docs.python.org/3/)
- [numpy docs](https://docs.scipy.org/doc/)
- [tensorflow docs](https://www.tensorflow.org/api_docs/python/)

## Python, NumPy, TensorFlow?
![python](https://raw.githubusercontent.com/neurocats/neurocats.github.io/master/assets/tensorflow/python-logo.png)
![python](https://raw.githubusercontent.com/neurocats/neurocats.github.io/master/assets/tensorflow/numpy.jpg)
![python](https://raw.githubusercontent.com/neurocats/neurocats.github.io/master/assets/tensorflow/tensorflow.png)

We first have to find out what kind of tools and supports you have for mathematical programming 
especially for building AI Software.   
What is used for which purpose? Which problems are more efficiently solved 
with which tool?

### Python
> Python is a programming language that lets you **work quickly** and 
**integrate systems** more effectively. 
**- python.org**

This just means that **Python** is
- multi-purpose, (especially cross-platform) and
- can easily combine different systems.

In short: It can do everything everywhere but nothing right or in the most 
efficient way. It is slow and hard to parallelize, because it doesn't live 
deep inside the system like e.g. C++. However it is elegant and easy to 
use.

### NumPy
> NumPy is the fundamental **package** for scientific computing with Python. 
**- numpy.org**

Python is loved by the scientific community because of the variety of 
scientific computing packages e.g. **NumPy** and SciPy. This gives the 
developer the availability of mathematical tools, e.g. matrix computation, in 
the elegant style of Python.
 

### TensorFlow
> An open-source software **library** for machine intelligence.
**- tensorflow.org**

Tensorflow is a C++ library. In more detail, think of TensorFlow as an API. With a
Python interface. It is fast, it can give you access to GPU computation and it can 
parallelize computations. TensorFlow has a more general 
respresentation of computations. Thus it is able to compute a symbolical 
derivative of a mathematical function and knows which part of the 
computation can work in parallel. 

Have a look at the [Whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf) 
if you can't wait to know how all that's done.

### Conclusion
Basically what we are doing is using Pythons ability to integrate
systems by integrating the use of TensorFlow (via Python API). We are 
communicating with the GPU over TensorFlow controlled by Python. We are 
combining easy to use Python with fast but complicated written C++ and taking 
the best properties from both worlds.

## TensorFlow Philosophy
I will give an overview of how TensorFlow works and I will give a practical 
example of the concrete difference to pure Python in the next section.

### It's all about Computation Graphs
One field in theoretical computer science focuses on representing mathematical 
operations in the form of a graph. TensorFlow is a tool for building 
computation graphs and provides functions like differentiation that make the use
of a formula easier. TensorFlow provides nodes that you can feed with NumPy 
objects and gives the possibility to evaluate each nodes value given the 
input (or constant).

Never forget: You are building a graph. When you want to evaluate a variable
(graph node) you can only do it in a so called 'TensorFlow session'. Outside
such a session you will just see an abstract graph object that lives in your
desired architecture.

### Graph or not a Graph
You will always need **Tensors/Arrays**. So let's start with the smallest 
non-trivial example: A tensor of rank 2 or: A Matrix.

```python
import numpy as np
import tensorflow as tf

# create a matrix
npMatrix = np.zeros(shape=(3, 3), dtype=np.float32)
tfMatrix = tf.zeros(shape=(3, 3), dtype=tf.float32, name="tfMatrix")

# print out results
print("Numpy Matrix:\n", npMatrix, "\n")
print("Tensorflow node:\n", tfMatrix, "\n")

# or maybe in the right way for tensorflow
with tf.Session() as sess:
    node = sess.run(tfMatrix)
    print("Tensorflow node value:\n", node, "\n")
```
Outputs:
```
Numpy Matrix:
 [[ 0.  0.  0.]
  [ 0.  0.  0.]
  [ 0.  0.  0.]] 
  
Tensorflow node:
 Tensor("zeros_1:0", shape=(3, 3), dtype=float32)
 
Tensorflow node value:
 [[ 0.  0.  0.]
  [ 0.  0.  0.]
  [ 0.  0.  0.]] 
```

Obviously, the TensorFlow node is completely different to its 
value. Like I mentioned TensorFlow creates an abstract graph object. We can 
build a computation graph via Python using TensorFlow and can evaluate the 
nodes value assigned to the current computation in a TensorFlow session. 
- `TensorFlow node` is a representation of the abstract 
node of the computation graph that was build at the initialization step.  
- `TensorFlow node value` is the value of the node that 
was assigned to this computation. It was evaluated in a TensorFlow session.

To make that clear please analyse the outcome of the next example. We will 
now add up 2 matrices.

```python
import numpy as np
import tensorflow as tf

# create the first matrix
npMatrix = np.zeros(shape=(3, 3), dtype=np.float32)
tfMatrix = tf.constant(npMatrix, dtype=tf.float32, name="tfMatrix")

# create the second matrix
npMatrix2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
tfMatrix2 = tf.constant(npMatrix2, dtype=tf.float32, name="tfMatrix2")

# add them up
npResult = np.add(npMatrix, npMatrix2)
tfResult = tf.add(tfMatrix, tfMatrix2, name="tfResult")

# Print out the sum
print("Numpy result:\n", npResult, "\n")
print("Tensorflow result node:\n", tfResult, "\n")

# or maybe in the right way for tensorflow
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    node = sess.run(tfResult)
    print("Tensorflow result:\n", node, "\n")
    
    # for deeper understanding
    sumand1, sumand2 = sess.run([tfMatrix, tfMatrix2])
    print("Summands:\n", "first:\n", sumand1, "second:\n", summand2)
writer.close()
```
Output:

```
Numpy result:
 [[ 1.  0.  0.]
  [ 0.  1.  0.]
  [ 0.  0.  1.]] 

Tensorflow result node:
 Tensor("tfResult:0", shape=(3, 3), dtype=float32) 

Tensorflow result:
 [[ 1.  0.  0.]
  [ 0.  1.  0.]
  [ 0.  0.  1.]] 
  
Summands:
 first:
  [[ 0.  0.  0.]
   [ 0.  0.  0.]
   [ 0.  0.  0.]] 
 second:
  [[ 1.  0.  0.]
   [ 0.  1.  0.]
   [ 0.  0.  1.]]

```

#### What happened?
Tensorflow has two workflows.
1. Building the computation graph.
```python
# Summand
tfMatrix = tf.constant(npMatrix, dtype=tf.float32, name="tfMatrix")
# Summand
tfMatrix2 = tf.constant(npMatrix2, dtype=tf.float32, name="tfMatrix2")
# Sum
tfResult = tf.add(tfMatrix, tfMatrix2, name="tfResult")
```
Notice that what seemingly is just an initialization step for a variable is also the birth of a 
node in the computation graph, that you feed with the properties/parameters, 
even with a name.

2. Execute the actual computation in a TensorFlow session.
```python
# Starting the session
with tf.Session() as sess:
    # initialize all variables
    sess.run(tf.global_variables_initializer())
    # provide the results for tensorboard (a visualization tool)
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # evaluate the value of a certain node
    node = sess.run(tfResult)
    
    # evaluate the value of any node of the computation graph
    sumand1, sumand2 = sess.run([tfMatrix, tfMatrix2])
writer.close()
```
Computation graph:  
![ex_II](https://raw.githubusercontent.com/neurocats/neurocats.github.io/master/assets/tensorflow/ex_II.png)

As shown in the image of the computation graph you have created 3 nodes 
(summand, summand, sum) if you ignore the `init` node. We can evaluate every 
assigned value of a node given the current computation. TensorFlow will 
calculate it via C++ API, not with native Python like in the NumPy case.

I created that graph with TensorBoard which is a tool that converts
so called TensorFlow summaries into a visualization. With that there arise 
more new concepts like namescoping, building summaries etc. I will 
elaborate on that in the next tutorial while I show you how TensorFlow can 
compute symbolic derivatives, impossible for pure Python.

Have a great code. See you in the next tutorial.
