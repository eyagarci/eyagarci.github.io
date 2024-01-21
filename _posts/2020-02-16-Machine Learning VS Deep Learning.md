---
title:  "Chapter 2: Machine Learning VS Deep Learning"
date:   2020-02-16 23:00:00
categories: [ai]
tags: [docs,ai, MachineLearning, DeepLearning]    
image:
  path: /assets/imgs/headers/MLvsDL.webp
---

## Machine Learning

Machine Learning (ML) is a branch of computer science where we develop algorithms that make a machine learn to do something without actually making computations about it. The basic premise of machine learning is to build algorithms that can receive input data and use statistical analysis to predict an output while updating outputs as new data becomes available.
Machine Learning is classified into 3 types of algorithms -
Supervised Learning — [Link coming soon in a future blog]
Unsupervised Learning — [Link coming soon in a future blog]
Reinforcement Learning — [Link coming soon in a future blog]

## Difference between Artificial Intelligence, Machine Learning, and Deep Learning

Nowadays many misconceptions are there related to the words machine learning, deep learning and artificial intelligence(AI), most of the people think all these things are the same whenever they hear the word AI, they directly relate that word to machine learning or vice versa, well yes, these things are related to each other but not the same. Let’s see how.
AI means to replicate a human brain, the way a human brain thinks, works and functions. The truth is we are not able to establish a proper AI till now but we are very close to establish it, one of the examples of AI is Sophia, the most advanced AI model present today. The main goal here is to increase the success rate of an algorithm instead of increasing accuracy. It works like a computer program that does smart work.

Machine learning is one subfield of AI. The core principle here is that machines take data and “learn” for themselves. It’s currently the most promising tool in the AI kit for businesses. The main goal here is to increase the accuracy of an algorithm instead of its success rate.
There are some steps involved in machine learning which are a prediction, classification, recommendations, clustering and decision making. When all these five work together we call it artificial intelligence.

Deep Learning is a subset of ML. The main difference between deep and machine learning is, machine learning models become better progressively but the model still needs some guidance. If a machine learning model returns an inaccurate prediction then the programmer needs to fix that problem explicitly but in the case of deep learning, the model does it by himself. Automatic car driving system is a good example of deep learning.


## Overview of Supervised Learning
In supervised learning, the algorithm is provided with a finite set of data which contains the right answers for each of the input values. The machine has the task to predict the right answers by analyzing the dataset correctly.

### Example of Supervised Learning.
As shown in the above example, we have initially taken some data and marked them as ‘Tom’ or ‘Jerry’. This labeled data is used by the training supervised model, this data is used to train the model.
Once it is trained we can test our model by testing it with some test new mails and checking of the model can predict the right output.
Types of Supervised Learning
Regression: It is a type of problem where the output variable is a real value, such as “dollars” or “weight”.
Classification: It is a type of problem where the output variable is a category, such as “red” or “blue” or “disease” and “no disease”.
Overview of Unsupervised Learning
In unsupervised learning, the algorithm is provided with an unlabelled dataset and it predicts a pattern in the data.

### Example of Unsupervised Learning
In the above example, we have given some characters to our model which are ‘Ducks’ and ‘Not Ducks’. In our training data, we don’t provide any label to the corresponding data. The unsupervised model can separate both the characters by looking at the type of data and models the underlying structure or distribution in the data to learn more about it.
Types of Unsupervised Learning
Clustering: A clustering problem is where we group similar data according to a pattern in data, such as grouping customers by purchasing behavior.
Association: An association rule learning problem is where we want to discover rules that describe large portions of your data, such as people that buy X also tend to buy Y.
Overview of Reinforcement Learning
In reinforcement learning, the algorithm learns by interacting with the environment. The algorithm adjusts itself based on feedback.

### Example of Reinforcement Learning
In the above example, we can see that the agent is given 2 options i.e. a path with water or a path with fire. A reinforcement algorithm works on reward a system i.e. if the agent uses the fire path then the rewards are subtracted and the agent tries to learn that it should avoid the fire path. If it had chosen the water path or the safe path then some points would have been added to the reward points, the agent then would try to learn what path is safe and what path isn’t.

## Summary
In this blog, I have presented the basic concepts of machine learning. I hope this blog is helpful for beginners and will motivate them to get interested in this topic.