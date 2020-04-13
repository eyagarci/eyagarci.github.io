---
layout: post
comments: true
title:  "Evolutionary algorithms"
excerpt: "Oswald Berthold - Quick introduction to evolutionary algorithms"
date:   2017-10-10
mathjax: true
hard_wrap: false
---

## Table of Contents ##

1.  [Overview](#orgcabe70e)
2.  [Core algorithm](#org41fbfe2)
3.  [Filling in the details](#org5209254)
4.  [Basic types of evolutionary algorithms](#org9031262)
5.  [Advanced techniques](#org0e6ea8f)
6.  [Relations to other fields](#orgb95580f)
7.  [Examples](#org1bf579d)
8.  [References](#org885094c)

<span style="color:red">WARNING: I will have to do the graphics on the board.</span>  


<a id="orgcabe70e"></a>

## Overview ##

Evolutionary algorithms (EAs) are a family of bioinspired algorithms <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup> that generalize various techniques invented some time between 1930 and the 1960s. Since then EAs have been in widespread use in many fields of science, engineering, and art, among others. Evolutionary algorithms are continually present among the state of the art in many problems of optimization, machine learning, robotics, or artificial life. There is a large community that drives the field, with >6000 authors counted in 2005 <sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup>. A major international conference is GECCO - The Genetic and Evolutionary Computation Conference <sup><a id="fnr.3" class="footref" href="#fn.3">3</a></sup>, with 13 tracks and 180 full papers in 2017.  

Why do we want to model natural evolution? Because obviously it is a powerful search technique. We would like to replicate this functionality and apply it to our own difficult search and optimization problems.  


<a id="org41fbfe2"></a>

## Core algorithm ##

In (Floreano & Mattiussi, 2008) the four pillars of evolutionary theory are *population*, *diversity*, *heredity*, and *selection*. From this, a simple mathematical model of natural evolution is a tuple  

$$ \begin{equation} \text{EA} = (e, p, f, o, s) \end{equation} $$  

with environment $e$, population $p$, fitness $f$, operators $o$ and some generation statistics $g$. A minimal algorithm for improving the performance of $p$ with respect to environment and fitness $f(p, e)$ based on the model (1) looks like this:  

    1 evolutionary_algorithm:
    2   initialize e, p, f, o, s, p_
    3   do min(maximum iterations, forever)
    4     for each individual i in p
    5       compute fitness f_i of i in e
    6     for each slot i_ in p_
    7       randomly select individuals i,j from p with probability weighted by fitness 
    8       apply operators o to pair and store results in slot
    9     store entire state in s
    10    update p <- p_

Let's step through the algorithm line by line. Line 2 initializes the components, with p\_ some temporary memory. Line 3 then iterates the following block for a maximum number of iterations. In line 5 an individual is evaluated in the environment and its fitness computed, which is done for each member of the population in line 4. Line 6 creates a new population by sampling pairs of last generation individuals weighted by their fitness in line 7 and computing a new individual from the pair using the genetic operators in line 8. Line 9 updates the overall statistics and line 10 replaces the old population with a new one and jumps back to line 4.  


<a id="org5209254"></a>

## Filling in the details ##

Why and how should this work? Let's have a look at it in more detail.  

The *population* is a list of individuals and its size is an important parameter controlling *diversity* and search space coverage. *Individuals* represent candidate solutions to the problem we would like to solve. They usually consist of a *genotype* and a *phenotype*. The genotype can be a string of bits, reals, or a more complex structure like a nested list or a program trees.  

Usually, the fitness cannot be evaluated directly on the genotype but is translated first into a corresponding phenotype. The phenotype is then put into the environment of the problem and is evaluated by being allowed an episode of 'doing its thing'. The genotype encoding and genotype to phenotype mapping are another fundamental aspect of EAs. For example in function optimization, the genotype can directly encode the vector arguments at which the function should be evaluated. When tuning the controller of a simulated or real robot, the genotype might encode the controller's parameters. For evaluation the parameterized controller needs to be run on the system for a given amount of time in order to compute a meaningful fitness (policy search). A parameterized policy can be made even more expressive by introducing generative mappings, for example by interpreting the genome as a program, as a parameterized *construction rule* for policies (CPPNs) or by encoding synaptic plasticity rules in a neural network instead of synaptic weights directly.  


### Environment ###

The environment takes an individual $p_i\text{'s}$ phenotype as an input and outputs an episode of data corresponding to that individual. In many cases the returned data will be randomly distributed conditioned on the phenotype input, due to domain complexity, which has to be taken into account be the algorithm. Very similar to policy search.  


### Fitness ###

A fitness function $f$ is a function mapping from the combined space of individuals and their data episodes to the reals, e.g. $f(p_i, x) \rightarrow \mathbb{R}$. The function encodes, how an individual's fitness with respect to some desired configuration or behavior is computed from the data returned by the environment.  

> &#x2026; while in natural evolution the fitness of an individual  
> is defined by its reproductive success (number of offspring), in artificial evo-  
> lution the fitness of an individual is a function that measures how well that  
> individual solves a predefined problem. (Floreano & Mattiussi, 2008, pg. 1)  


### Genetic operators ###

The operators $o$ are stochastic transformations mapping from the space of $n$-tuples of individuals to the space of individuals. The operators produce new solutions from existing ones.  

The three main operators are *selection*, *mutation*, and *crossover*, also known as recombination. Genetic operators take one or more genomes as input and return a single new genome. The simplest type of EA results from using only selection and mutation. Selection retains a memory of solutions and their fitness, and mutation provides the necessary exploration around the fittest candidates. This work well in many situations but including crossover in between selection and mutation can a) accelerate the search (the real benefit of sexual reproduction) and b) enable *jump* access to regions in configuration space which are unreachable using mutation alone due to local minima. Mutation and combination are local and global exploration functions respectively. Both a) and b) depend on appropriate genotype to phenotype mapping.  


### Hyperparameters ###

Even a simple EA quickly accumulates many hyperparameters like population size, genetic representation, geno-pheno mapping, mutation rate, combination rules, mixing coefficients in composite fitness functions, and so on.  

The main tools for choosing hyperparameters are defaults, empirical estimates, first principles, or optimization. For example, the statistics $s$ can be used to modulate hyperparameters like mutation rate, recombination rules, or fitness weights. In addition, $s$ is the basis for graphical analyses of the experiment.  


<a id="org9031262"></a>

## Basic types of evolutionary algorithms ##

There are a few basic types of EAs which are distinguished by the type of genome they use.  

*Genetic algorithms* use strings of bits as the basis of the genetic encoding. The bits could correspond to pixel in a bitmap, inclusion of options in a configuration, or turns to make traversing a binary tree.  

*Evolution strategies* are based on real-valued vectors of parameters owing to parametric design methods of hardware evolution.  

*Genetic programming* is slightly different and includes an implicit layer of a developmental encoding. In GP the genome encodes a computation graph and the algorithm explores the space of a family of programs which can bump the expressive power of the genome by oom.  


<a id="org0e6ea8f"></a>

## Advanced techniques ##

Non-exhaustive list of some advanced topics and techniques in evolutionary computation.  

-   Open-ended evolution
-   Coevolution
-   Diversity
-   Modularity
-   Probabilistic genome encoding, for example CMA-ES
-   Developmental encoding, for example CPPNs, NEAT, HyperNEAT, map-elites, &#x2026;
-   Genetic regulatory pathways
-   Distributed representation, genetic drift

and some particularly interesting subfields within evolutionary computation  

-   Evolutionary robotics
-   Evolvable hardware, intrinsic evolution, in-silico evolution
-   Black-box optimization, e.g. hyperparameters
-   Neuroevolution
-   Theory of evolution


<a id="orgb95580f"></a>

## Relations to other fields ##

Evolutionary methods are closely related with other computational methods, for example, EAs can be framed and understood in terms of stochastic optimization, black-box optimization, particle based methods, or policy search.  


<a id="org1bf579d"></a>

## Examples ##

A few examples of current and classic applied evolutionary methods  

-   Fernando and others, 2017, PathNet: Evolution Channels Gradient Descent in Super Neural Networks, <https://arxiv.org/abs/1701.08734>, <https://www.youtube.com/watch?v=tmQaj0ZqmiE>
-   Lipson, Bongard and others, 2016, Evolving Swimming Soft-Bodied Creatures,  
    <https://www.youtube.com/watch?v=4ZqdvYrZ3ro>, Kriegmann and others,  
    2017, The Evolution of Development in Soft Robots, <https://www.youtube.com/watch?v=gXf2Chu4L9A>
-   Cully & other, 2015, Robots that can adapt like animals (Nature  
    cover article), <https://www.nature.com/nature/journal/v521/n7553/full/nature14422.html>, <https://www.youtube.com/watch?v=T-c17RKh3uE>
-   Tonelli and Mouret, 2013, On the Relationships between Generative Encodings, Regularity, and Learning Abilities when Evolving Plastic Artificial Neural Networks, <10.1371/journal.pone.0079138>
-   Clune, Mouret & Lipson, 2012, The evolutionary origins of  
    modularity, <https://arxiv.org/abs/1207.2743>,
-   Koutnik and others, 2010, Evolving neural networks in compressed weight space, <https://dl.acm.org/citation.cfm?id=1830596>, <https://www.lri.fr/~hansen/proceedings/2013/GECCO/proceedings/p1061.pdf>
-   Hornby and other, 2006, Automated Antenna Design with Evolutionary  
    Algorithms,  
    <http://alglobus.net/NASAwork/papers/Space2006Antenna.pdf>, NASA antenna <https://en.wikipedia.org/wiki/Evolved_antenna>
-   de Nardi, Holland and others, 2006, Evolution of Neural Networks  
    for Helicopter Control: Why Modularity Matters, <http://julian.togelius.com/DeNardi2006Evolution.pdf>
-   Bird and Layzell, 2002, The Evolved Radio and its Implications for Modelling the Evolution of Novel Sensors, <https://people.duke.edu/~ng46/topics/evolved-radio.pdf>, FPGA evolved radio-receiver, ?
-   Lichtensteiger and Eggenberger, 1999, Evolving the Morphology of a  
    Compound Eye on a Robot, <https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated2/lichtensteiger_compound_eye.pdf>,
-   Funes and Pollack, 1997, Computer Evolution of Buildable Objects, <http://www.demo.cs.brandeis.edu/papers/other/cs-97-191.html>
-   Thompson, 1995, An evolved circuit, intrinsic in silicon, entwined  
    with physics, <https://link.springer.com/chapter/10.1007/3-540-63173-9_61>, FPGA circuit evolution, Thompson
-   Karl Sims, 1994, Evolved Virtual Creatures, Evolution Simulation, <https://www.youtube.com/watch?v=JBgG_VSP7f8>
-   Stanley & others, Picbreeder: collaborative evolutionary art, <http://picbreeder.org/>
-   Dawkins, Blind watchmaker, <http://www.dailymotion.com/video/x1jprj5>
-   Self-replicating robot, Fumiya Iida, <http://www.cam.ac.uk/research/discussion/opinion-how-we-built-a-robot-that-can-evolve-and-why-it-wont-take-over-the-world>


<a id="org885094c"></a>

## References ##

In addition to the papers and examples above here's additional reading  

-   Floreano & Mattiussi, 2008, Bio-inspired artificial intelligence
-   Nolfi & Floreano, 2000, Evolutionary robotics - The biology, intelligence, and technology of self-organizing machines
-   David B. Fogel, 2000, Evolutionary computation
-   Mitchell & Tayler, 1999, Evolutionary Computation : An Overview
-   John Holland, 1975, Adaptation in natural and artificial systems
-   Ingo Rechenberg, 1973, Evolutionsstrategie


## Footnotes ##

<sup><a id="fn.1" href="#fnr.1">1</a></sup> Inspired by the theory of natural evolution.

<sup><a id="fn.2" href="#fnr.2">2</a></sup> Cotta and Merelo, 2007, Where is evolutionary computation going? A temporal analysis of the EC community, <https://doi.org/10.1007/s10710-007-9031-0>

<sup><a id="fn.3" href="#fnr.3">3</a></sup> Website of SIGEVO, the ACM Special Interest Group on Genetic and Evolutionary Computation <http://sig.sigevo.org/index.html/tiki-index.php#&panel1-1>
