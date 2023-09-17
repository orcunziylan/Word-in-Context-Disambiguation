# Polysemous Word Disambiguation

## Table of Contents

1. [Introduction](#1-introduction)
2. [Preprocessing](#2-preprocessing)
   - [2.1 Lemmatization](#21-lemmatization)
   - [2.2 Stop Words](#22-stop-words)
   - [2.3 Word Frequencies](#23-word-frequencies)
   - [2.4 Keywords Tokenization](#24-keywords-tokenization)
   - [2.5 Number Tokenization](#25-number-tokenization)
3. [Model Architectures](#3-model-architectures)
   - [3.1 Multilayer Perceptron (MLP)](#31-multilayer-perceptron-mlp)
     - [3.1.1 Mixed Embeddings](#311-mixed-embeddings)
     - [3.1.2 Separated Embeddings](#312-separated-embeddings)
   - [3.2 Long Short-Term Memory (LSTM)](#32-long-short-term-memory-lstm)
4. [Pre-trained Embeddings](#4-pre-trained-embeddings)
5. [Extras](#5-extras)
6. [Summary](#6-summary)
6. [Tables](#7-tables)


## 1. Introduction
The objective of this project is to compare the usage of polysemous words in different sentences and determine if they are used in the same context or not. To tackle this problem, various experiments were conducted involving different preprocessing techniques, network architectures, and pre-trained embeddings, all of which will be discussed in the following sections.

## 2. Preprocessing
When dealing with a limited dataset, challenges like overfitting due to unknown and low-frequency words arise, leading to generalization problems. Various techniques are experimented with to mitigate these challenges and improve validation accuracy (see Table 1).

### 2.1 Lemmatization
Lemmatization is employed as a practical approach to address the generalization problem. It simplifies words to their base form, increasing token examples and aiding in generalization.

### 2.2 Stop Words
Common stop words like "the," "is," and "and" are removed from the dataset to focus on words making a significant difference in generalization.

### 2.3 Word Frequencies
Word frequencies were used to identify unique (lower frequency) words. Two common methods were explored: removal of these words and labeling them as unknown. The choice between these methods depended on experimentation, with the former predominantly used.

### 2.3 Word Frequencies
Word frequencies are used to identify unique words. Two common methods are explored: removal of these words and labeling them as unknown (<UNK>). 

### 2.4 Keywords Tokenization
Different keyword tokenization methods ("None," "Lemma," "Pos," "Flag") are employed to emphasize keywords and reduce complexity, independently of dataset lemmatization.

### 2.5 Number Tokenization
To handle the variety of numbers in sentences, separate tokenization techniques ("None," "Discard," "Flag") are used to either keep, remove, or flag them as "<NUMBER>.".

## 3. Model Architectures
### 3.1 Multilayer Perceptron (MLP)
In this project, MLP processed the output of the embedding layer. A word-embedding aggregation layer was introduced to lower dimensionality and extract new features for smoother training.

## 3. Model Architectures
### 3.1 Multilayer Perceptron (MLP)
An additional word-embedding aggregation layer is introduced to reduce dimensionality and extract new features, contributing to smoother training. Subsequently, the output is passed to a Multilayer Perceptron (MLP).

#### 3.1.1 Mixed Embeddings
An experiment involving mixed embeddings is conducted, where all word embeddings from two sentences are combined into a single vector. However, due to substantial information loss, this approach achieves only limited accuracy, approximately 56%.

#### 3.1.2 Separated Embeddings
Following the mixed embeddings' failure, separate mean vectors for each sentence are computed, considering individual sentence lengths to prevent information loss. Four methods ("separated" to use both vectors, "subtract" to have a single vector, and both are also combined with "weighted" to highlight the keyword) are developed to combine these vectors, resulting in higher accuracy scores (around 68%) (see Table 2).

### 3.2 Long Short-Term Memory (LSTM)
LSTMs, designed for sequential data, are also employed for experimenting further. Hyperparameter tuning and early stopping are applied to prevent overfitting but tested LSTM models consistently exhibit overfitting, with the best accuracy of around 65%.

## 4. Pre-trained Embeddings
Two pre-trained embeddings, GLOVE and Word2Vec, are considered. GLOVE, favored for its various embedding dimensions, leads to quicker experiments. Word2Vec is utilized in the project's final stages to evaluate its impact on the best-performing models (See Table 3).


## 5. Extras
- Utilization of pre-trained embeddings
- Grid search for hyperparameter tuning
- Various preprocessing techniques
- Comparative results presented in plots and tables

## 6. Summary
In conclusion, this project offers insights into the complexities of word-in-context disambiguation. It demonstrates the critical role of preprocessing techniques, the significance of choosing appropriate tokenization methods, and the impact of neural network architecture selection on model performance. Moreover, the exploration of pre-trained embeddings enriches the understanding of their influence on the task at hand.

## 7. Tables

## Table 1: Preprocessing Techniques and Validation Accuracy

| Type        | Accuracy   | Best Epoch # | Vocab   |
|-------------|------------|--------------|---------|
| **KEYWORD** |            |              |         |
| None        | 61.31      | 44           | 27,008  |
| Flag        | 60.81      | 50           | 25,247  |
| Lemma   | **61.41**  | 50       | 26,073 |
| Pos         | 61.11      | 50           | 25,250  |
| **NUMBER**  |            |              |         |
| None        | 61.31      | 44           | 27,008  |
| Flag        | 61.51      | 48           | 26,493  |
| Discard     | **63.00**| 50       | 26,492 |
| **STOP WORDS** |        |              |         |
| None        | 61.31      | 44           | 27,008  |
| Discard     | **66.67**| 49       | 26,961 |
| **FREQUENCY (>1)** |   |              |         |
| None        | 61.31      | 44           | 27,008  |
| **Low Freq- Del** | **61.41** | 50    | 18,518 |
| Low Freq - Unk | **61.61** | 46        | 18,519  |
| **DATASET** |           |              |         |
| None        | 61.31      | 44           | 27,008  |
| Lemma       | **62.40**  | 50           | 23,758  |

## Table 2: Validation Accuracies for Different Methods in Embedding Aggregation.

| Type       | Accuracy | Epoch |
|------------|----------|-------|
| Separated  | 68.38    | 96    |
| Weighted Separated  | **68.85**    | 113   |
| Subtract   | 68.35    | 77    |
| Weighted Subtract   | 67.16    | 37    |

## Table 3: The Effectiveness of Different Pre-trained Word Embeddings

| Embedding         | Accuracy (%) | Best Epoch | Embedding Dimension |
|-------------------|--------------|------------|---------------------|
| Random            | 52.7         | 44         | 300                 |
| Glove             | **64.1**         | 48         | 300                 |
| Word2Vec          | 59.0         | 49         | 300                 |
