---
layout: post
comments: true
title:  "Introduction to Blockchain Technology"
excerpt: "Frank Selensky - This post gives a brief introduction to the blockchain technologies as they are used in cryptocurrencies like bitcoin."
date:   2017-09-12
mathjax: true
---

## Cryptographic Hash Function

Since the whole concept of a blockchain is strongly based on *hash functions*, we'll start with defining these.

A **hash function**  $H$ is a function from the space of all strings to the space of data which has a fixed size. Additonaly a hash function has to be efficently computable, i.e. computing $ H(x) $ with $ x$ being an $n$-bit string should have a running time of $O(n)$. We will call $H(x)$ the **hash** of $x$, for a given string $x$.

A **cryptographic hash function** is a hash funtion with three additional properties, which are:
* collision-resistance
* hiding
* puzzle-friendliness

We call a hash function $H$ **collision-resistant**, if it is infeasible to find two strings $x$ and $y$ such that $ x \neq y$ and $H(x)=H(y)$.

Notice, that collision-resistance does not mean, that there are not two different strings, which will hash to the same value. Since the input space is infinte while the output space is finite, a hash function can not be injective. So there are collisions, but your are not able to find them. 

We call a hash function $H$ *hiding*, if when a secret value $r$ is choosen from a probability distribution that has *high min-entropy* (i.e the distribution is very spread out, and it is not predictable which output the random variable has), then given $ H(r \Vert x) $ it is infeasible to find $ x $. ( $ a \Vert b $ denote the concatenation of $ a $ and $ b $ )

Notice that the secret value $r$ is needed to ensure, that the input value for the hash fuction is in some sense very spread out.

We call a hash function $H$ **puzzle-friendly**, if for every $n$-bit output value $y$, if $k$ is choosen from a probablility distribution with high min-entropy, then it is infreasible to find $x$ such that $ H(k \Vert x)=y $ in significantly less time than $2^n$.

With a puzzle friendly hash function we are able to construct a *search puzzle* which we will need later when trying to find consensus between nodes.

A **search puzzle** consists of 
* A hash function $H$ 
* a randomly choosen value $z$ (which could be called the **puzzle id**)
* a **target set** $Y$ which is a subset of the value space of $H$.

A **solution** to the search puzzle is a value $x$, such that:
$$ H(z \Vert x)\in Y $$

The *puzzle-friendliness* of the hash function ensures that there is no shorter way or no more efficient way to solve the puzzle than just trying.

## Hash Pointer, Blockchain 

A **hash pointer** is the hash value of some information, which acts as a reference to that information (it points to that information).

From now on we assume, that we have fixed a cryptographic-hash-function $H$.

Now we have all we need to give the first initial definition of our thing of interest.

A **blockchain** is a list of information, where each item is linked to the previous one with a hash pointer. We will call each item in this list a **block**.

![blockchain]( https://viking-frank.github.io/assets/intro_blk_chain/def_blockchain.png )

Notice that each block in the blockchain tells us which block came before it and additionally the hash value gives us the ability to check if the previous block was altered or not. Appart from that we only need to store the last blocks hash to check the consistency of the blockchain. 

This is the very basic definition of a blockchain. And although the definition holds for the Bitcoin blockchain, there are several features in it and in the whole Bitcoin system which make it much more interesting and powerful than just a linked list of items.

To dive more into how the Bitcoin blockchain works we need also to take a look at how the transactions work and what essentially is a *coin* in a cryptocurrency.

## Transactions
Before we take a look at what transactions are, let us recall *public-key-cryptography* and *digital signatures*
### Public-key-cryptography
**Public-key-cryptography** uses pairs of keys. Each pair consists of a **public-key** and a **private-key**. There are two main functions you can accomplish with these kinds of systems:
* encryption: if someone encrypts data with a public-key, only the holder of the corresponding private key is able to decrypt the data.
* signing: that is with a secret-key you can take some sort of data and generate a **signature** for that data (this signature is some kind of string). Now everyone who has the public-key can verify, that the data was signed using the corresponding secret-key and not changed afterwards. 

The signing mechanism plays a key role in cryptocurrency transactions, as we will see after the next heading. 

Notice that you have to know the private-key to generate a signature. This is a property of good public-key-cryptography systems (its called **unforgeability**). 

By the way: the system used in Bitcoin is the ECDSA.


## Simple Transaction Schemes
The transactions defined in this part are simplified, but the concept is the same as in the real world.

First of all: There is nothing like a user-acount in cryptocurrencies like Bitcoin. If you want to receive Bitcoins, you have to generate a key-pair and get someone to send coins to your public-key. After this is done, you need your corresponding private-key to spend the coins. So a transaction transfers coins from one public-key to another (in the simplest case). Notice that you are not limited to use just one public-key, you can create and use as many as you like. 

There are mainly two types of transactions in cryptocurrencies like Bitcoin: **normal transactions** and **generation transactions**.

A **normal transaction** is basically a data set of inputs with signatures and outputs with public-keys and values. We can assign a unique **transaction id** to it by simply hashing the whole transaction. 
An **output** specifies a **value** and a *public-key*, which should receive this value.
An **input** uses the output of a previous transaction (by giving a hash pointer to that transaction and the index of the outbut) and it has to be signed with the *secret-key* which corresponds to the *public-key* which is specified in the output.

So in an abstract manner, a normal transaction looks like this:

<table>
  <tr>
    <td colspan="6">tx_id: 29</td>
  </tr>
  <tr>
    <td colspan="3">inputs</td>
    <td colspan="3">outputs</td>
  </tr>
  <td>prev_tx</td>
  <td>output_index</td>
  <td>signature</td>
  <td>index</td>
  <td>value</td>
  <td>pub_key</td>
  <tr>
  <td> 3</td>
  <td> 2</td>
  <td> sig_1</td>
  <td>0</td>
  <td>0.3</td>
  <td>pub_key_1</td>
  </tr>
  <tr>
  <td> 6</td>
  <td> 0</td>
  <td> sig_2</td>
  <td>1</td>
  <td>1.4</td>
  <td>pub_key_2</td>
  </tr>
</table>

In order for the transaction to be valid, all the signatures have to be valid and the sum of the output values have to be less or equal the sum of the input values (which are the values for the previous transaction outputs specified in the inputs).

Notice that an input is not linked to a special output, so it is not necessary to have as many inputs as outputs and vice versa.

You maybe came to the question: "When all inputs are claiming previous outputs, where does the whole thing start?"

Therefore, it exists another type of transaction: The **generation transaction**.

A **generation transaction** is the same as a **normal transaction**, but it has no inputs. So it doesn't transfer coins, it *generates* them. So these transactions are the ones where the transaction chains begin. 

After all, we can say that a cryptocurrency-coin is nothing what is passed around like in the real world. It is just a chain of transactions (This is just like Satoshi Nakamoto defined his *bitcoin*).

Now we can generate and transfer "coins". But there are two big problems. The first one is, that we need a mechanism to determine who can create and broadcast a generation transaction. The second one is the so called *double-spending-problem*. 

The **double-spending-problem** is the following: Assume that I know the secret-key, which I could use to spend the output 0 of transaction 20. So I could create a normal transaction with output 0 of transaction 20 as input and specify for example Alice (one pub_key of Alice) as an output, and sign the whole thing with my secret key. This would be a total valid transaction. Now Alice can start and do whatever she likes with that coin. But notice, that I still know the secret-key. And there is nothing that prevents me from creating the same transaction as above, but specifying Bob as the receiver in the output. Now there are two valid chains of transactions, which grant the ownership of that specific coin to two different people (or public keys).

It turns out that both problems are solved by maintaninig a distributed ledger of all transactions in a blockchain and using some kind of consensus protocol. The way the transactions are stored in the blockchain, but mainly the distributed consensus of Bitcoin was the coolest idea in the Bitcoin whitepaper by Satoshi Nakamoto.

## Distributed Ledger (The Blockchain) ##
To keep track of what transactions happened we use a blockchain. Let's say we collect all transactions and store all of them, which are valid and are no double-spents, in a block, which has a hash-pointer to the previous one, every 10 minutes. the resulting blockchain would look like this (The transactions are identified by their hash):

![blockchain with transactions]( https://viking-frank.github.io/assets/intro_blk_chain/blk_chain_with_tx.png )

So now we have a way to determine which transactions are valid for us even if there are double spents. But if somebody else does the same thing and builds his own blockchain, its very likely that this would not be the same and therefore the set of valid transactions will differ depending on which blockchain you consider. So we need a way to choose only one blockchain, everyone accepts as the public ledger.

One way this could be done is to have a central authority, for example us, who maintains the blockchain and publishes it to all participants. But this would have some obvious disadvantages. The main problem is, that everybody has to trust us and we are too powerful. We could do everything we like with the blockchain. For example exclude users (freeze their public-addresses, by simply not letting our lockchain contain transactions with this key) or, in the worst case, just stop maintaining the blockchain and kill the currency. 

Because of that we need a way to aggree on one blockchain decentralized between all participants. So that every participant has the same blockchain, which is then a true distributed-ledger of our cryptocurrency.

## Distributed Consensus

From now on we assume that we have a **peer-to-peer network** of the participants of our cryptocurrency and we will call a participant a **node**. Every node stores the blockchain and to choose what will be the next block we use the following protocol:

 1. every node collects all transactions, it hears about (which it receives by other nodes) and stores them in a pool of transactions not included in the blockchain yet (called the **mempool**).  
 2. every node forms a block of valid transactions from its *mempool* .
 3. every 10 minutes one of the nodes gets randomly choosen, which then broadcasts its block as a candidate for the next block of the blockchain.
 4. The other nodes check if the block is valid and if it is they append it to their blockchain and delete the transactions which are in the block from their mempool. If the block is not valid, they refuse the block.
 
This system results in a distributed blockchain which is maintained by all nodes in the network. So we have solved our *double-spending-problem*, since everyone can check if the transaction he or she waits for is included in the distributed blockchain.

Aditionaly, we can solve our problem of the *generation transactions* by simply defining the first transaction in every block to be a *generation transaction*. Typically, every node would include its own public key in the generation-transaction in the block it broadcasts. So since it is randomly choosen who broadcasts the next block, it is also randomly choosen who can create new coins in each step. To prevent the nodes from creating an arbitrary amount of coins, we set the rule, that a block is only valid if the *generation-transaction* does create a previously fixed amount of new coins. Our blockchain now looks like this:

![blockchain with generation transaction]( https://viking-frank.github.io/assets/intro_blk_chain/blk_chain_with_gen_tx.png )

So how do we choose a node randomly with no central authority?

So the last main aspect - and this was the ingenious idea in the Bitcoin whitepaper - to determine is: How do we randomly choose a node to publish the next block in a distributed peer-to-peer network, where everyone wants to create blocks, because he or she then can create a *generation-transaction*? 

## Proof-of-work
A **proof-of-work** is simply a solution to a *cryptographic-search-puzzle*. As we mentioned before there is no other way to solve a search-puzzle than trying. So when you have found a solution to a given puzzle you can show it to another person, who can then easily check it. So as long as you are the first to publish the solution to the puzzle, it is very likely, that you've done some work to solve the puzzle, so it is a **proof of work**.

How do we use that to choose a block broadcaster randomly? 

We start by including a **nonce** in the block, which is an integer, which can be choosen arbitrarily. Additionaly we require the hash of the block to be less than a given **target** value for the block to be valid. The blockchain now looks like this:

![blockchain with ](https://viking-frank.github.io/assets/intro_blk_chain/blk_chain_with_nonce.png )
(In this picture the block hashes need to start with three zeros in order to be valid.)

So it is a search-puzzle for each node to find a nonce, such that the hash of his block is below the given target. Since every node would use his own public-key in the generation transaction and might include different transactions or another order of them, every node would have a different search-puzzle. Now whenever a node finds a solution to its search-puzzle, it can broadcast its block. The other nodes check it for validity, what means, checking:
* all transactions are valid
* no double spents occur
* the hash of the block is below the target

If it is valid, the node appends the new block to his blockchain and starts searching for the next block.

Since the hash values are somehow random, to choose the first block which has a valid hash is an effective way to randomly choose a node to publish the next block depending on its computing power.

### Using the Target to Adjust Difficulty
We can use the *target* value which is an upper bound for the block hash, in order for the block to be valid, to adjust the time between two blocks. As you may have noticed, the estimated time after which a block is found depends on the sum of the computing power of all nodes in the network. So if we had a fixed *target*, the time between two blocks would decrease with the computing-power of the network increasing. 
To avoid this and keep the estimated block time stable at about 10 minutes, we will adjust the target value every 2016 blocks. 2016 blocks would take 2 weeks with a block every 10 minutes. We will adjust the target with the following formula:

$$ \frac{\text{time for the last 2016 blocks in seconds}}{2016 \cdot 10 \cdot 60 \text{s}} \cdot \mathrm{currentTarget} = \mathrm{newTarget}$$

So if it took less than two weeks to generate the last 2016 blocks, the fraction on the left would be below 1 and therefore the target would decrease. A smaller target makes it more difficult to find new blocks. 

The target value can be computed by every node itself. What keeps the nodes from cheating in this computation is, that their blocks wouldn't be accepted by other nodes if they didn't fit the target.

This is the same way the target is adjusted in *Bitcoin*.

## Other Consensus Types
Since *proof-of-work* also has some disadvantages (it consumes very much electricity), there emerged other consensus types. 

### Proof-of-stake
In a proof of stake system, the node to broadcast the next block is choosen based on the amount of the currency it holds instead of the computing power it has. For example **peercoin** works with a hybrid system. It is quite similar to the *proof-of-work* system we mentioned above, but you can use **coin-age** to higher the *target* dramatically. **Coin-age** is simply the product of the amount of coins you hold times the time you hold them. This results in the *coin-age* beeing much more important than the computing power.

### Proof-of-useful-work
Since *proof-of-work* consumes a lot of computing power to compute "useless" hash values, the basic idea is to replace this computing of hashes with an equivalent system, which instead produces "useful" results, besides maintaining the network and the currency. It turns out, that it is not simple to find such a *proof-of-useful-work*, which is adjustable as *proof-of-work* or *proof-of-stake* while producing "useful" results. One currency using *proof-of-useful-work* is **primecoin**, where a node has to find prime numbers in a given large space. 

### Proof-of-storage
This is a system, where your probability of broadcasting the next block depends on how much of a large file you have stored. This could be a useful way to store files, which are very large and important. But it gets quite complicated, when the files change over time.

## Conclusion

These were the key principles, presented in a very compact way, to give you an idea of how Bitcoin and other cryptocurrencies work.  If you have any remarks, questions or ideas how to improve this post, feel free to write me an email <fs@neurocat.ai>. Hope you enjoyed reading. Stay curious.
