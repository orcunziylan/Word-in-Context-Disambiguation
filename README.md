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

## 1. Introduction
The objective of this project is to compare the usage of polysemous words in different sentences and determine if they are used in the same context or not. To tackle this problem, various experiments were conducted involving different preprocessing techniques, network architectures, and pre-trained embeddings, all of which will be discussed in the following sections.

## 2. Preprocessing
While dealing with a small dataset, there were no strict rules for data preparation. However, having a limited dataset posed challenges like overfitting due to unknown and low-frequency words, which led to generalization problems. Various techniques were experimented with to mitigate these challenges and improve validation accuracy (see Table 1). N-grams were not considered as a preprocessing technique due to their potential to increase the number of unique tokens unnecessarily.

### 2.1 Lemmatization
Lemmatization was employed as an effective approach to address the generalization problem. It simplifies words to their base form, increasing token examples and aiding in generalization.

### 2.2 Stop Words
Common stop words like "the," "is," and "and" were removed from the dataset to focus on words making a significant difference in generalization.

### 2.3 Word Frequencies
Word frequencies were used to identify unique (lower frequency) words. Two common methods were explored: removal of these words and labeling them as unknown. The choice between these methods depended on experimentation, with the former predominantly used.

### 2.4 Keywords Tokenization
Different keyword tokenization methods ("None," "Lemma," "Pos," "Flag") were employed to emphasize keywords and reduce complexity, independently of dataset lemmatization.

### 2.5 Number Tokenization
To handle the variety of numbers in sentences, separate tokenization techniques ("None," "Discard," "Flag") were used to either keep, remove, or flag them as "<NUMBER>."

## 3. Model Architectures
### 3.1 Multilayer Perceptron (MLP)
In this project, MLP processed the output of the embedding layer. A word-embedding aggregation layer was introduced to lower dimensionality and extract new features for smoother training.

#### 3.1.1 Mixed Embeddings
Initially, mixed embeddings were attempted by combining word embeddings from two sentences into a single vector, but this method yielded limited accuracy (approximately 56%).

#### 3.1.2 Separated Embeddings
Following the mixed embeddings' failure, separate mean vectors for each sentence were computed, considering individual sentence lengths to prevent information loss. Three methods ("weighted," "separated," "subtract") were developed to combine these vectors, resulting in higher accuracy scores (around 68%).

### 3.2 Long Short-Term Memory (LSTM)
LSTMs, designed for sequential data, were employed. Hyperparameter tuning and early stopping were applied to combat overfitting, but LSTM models consistently exhibited overfitting, with the best accuracy around 65%.

## 4. Pre-trained Embeddings
Two pre-trained embeddings, GLOVE and Word2Vec, were considered. GLOVE, favored for its various embedding dimensions, facilitated quicker experiments. Word2Vec was utilized in the project's final stages to evaluate its impact on the best-performing models.

## 5. Extras
- Utilization of pre-trained embeddings
- Grid search for hyperparameter tuning
- Various preprocessing techniques
- Comparative results presented in plots and tables

| Category          | Type            | Accuracy | Best Epoch # | Vocab  |
|-------------------|-----------------|----------|--------------|--------|
| **Keyword**       | None            | 61.31    | 44           | 27,008 |
|                   | Flag            | 60.81    | 50           | 25,247 |
|                   | **Lemma**       | **61.41**| **50**       | **26,073** |
|                   | Pos             | 61.11    | 50           | 25,250 |
| **Number**        | None            | 61.31    | 44           | 27,008 |
|                   | Flag            | 61.51    | 48           | 26,493 |
|                   | **Discard**     | **63.00**| **50**       | **26,492** |
| **Stop Words**    | None            | 61.31    | 44           | 27,008 |
|                   | **Discard**     | **66.67**| **49**       | **26,961** |
| **Frequency (>1)**| None            | 61.31    | 44           | 27,008 |
|                   | **Low Freq- Del** | **61.41** | **50**    | **18,518** |
|                   | Low Freq- Unk   | 61.61    | 46           | 18,519 |
| **Dataset**       | None            | 61.31    | 44           | 27,008 |
|                   | **Lemma**       | **62.40**| **50**       | **23,758** |



