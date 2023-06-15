---
layout: post
title: "NLP-Driven Predictive Modeling of Clinical Trial Status"
author: "Diep Nguyen"
tags: [nlp, ml,neural network]
categories: journal
image: bg_clinical.jpg
---

# Introduction/ Motivation
Clinical trials are research studies conducted to evaluate the safety and effectiveness of new medical interventions, such as drugs, treatments, or medical devices. These trials involve the participation of human volunteers who may benefit from innovative healthcare solutions. However, not all clinical trials proceed as planned, and in some cases, they may be terminated prematurely. Given the considerable financial, logistical, and human resources invested in these trials, the termination of a clinical trial carries significant implications beyond research setbacks, encompassing economic, ethical, and scientific consequences. 


Many clinical trials have been terminated due to failure to meet the required accrual rate. The success of clinical trials relies on the participation and enrollment of volunteers. However, each trial has specific inclusion and exclusion criteria that potential volunteers must meet in order to participate. Setting overly specific or strict criteria may result in low enrollment rates. This article aims to explore the relationship between inclusion/exclusion criteria and the termination of clinical trials. To achieve this, the text data will be processed using the Continuous Bag of Words method to generate embeddings, and the trial status will be classified using a random forest algorithm. Additionally, pre-trained word embeddings will be utilized in a Long Short-Term Memory (LSTM) network, a type of Recurrent Neural Network (RNN), for the classification task. 

# Data Preprocessing and Exploration
The data used for this project is available on the [ClinicalTrials.gov](ClinicalTrials.gov) website. Unfortunately, the inclusion/exclusion criteria cannot be downloaded directly. Due to this limitation, the site was scraped to obtain this feature. The project will specifically focus on clinical trials for cancer. After removing any NaN values, a total of $17,065$ cancer-related clinical trials (both interventional and observational) were considered. Out of these trials, $8,168$ were terminated, while $8,897$ trials were completed. Below is a summary of the data used in the analysis. 

![alt text](https://github.com/dnnguyen99/dnnguyen99.github.io/blob/gh-pages/assets/img/data_illustration.png?raw=true){:width="600px"}
*Each clinical trial is identified by a unique NCTID. The "Criteria" column includes both the inclusion criteria (requirements for participating in a trial) and exclusion criteria (conditions that disqualify individuals from participating). The trial status is categorized into two categories: Terminated and Completed.*

Now, the text data will be pre-processed. The inclusion/exclusion criteria for each observation, referred to as "criteria," will be examined. Each clinical trial has a list of criteria, which we will call a "document." To start, newline characters and digits will be removed from each document. Then, punctuation will be eliminated, and all letters will be converted to lowercase. The text will then be tokenized, breaking down the sentences into individual words. This step enables us to apply word embeddings to the sentences. Lastly, it is necessary to ensure that all documents have the same length for LSTMs.

To achieve this,  the total number of words in each document can be analyzed. By examining the statistics of the dataset, it is observed that the average number of words in the document is approximately $304$ words. This means that we should truncate documents that are longer than 300 words and add padding to documents that have fewer than $300$ words. This threshold strikes a balance between capturing sufficient context and ensuring computational efficiency.

# Generate Word Embeddings

Machine learning models, such as random forests, typically take numerical input. Textual data can be represented in a numerical format using word embeddings, enabling these models to process it. Unlike the sparse vectors obtained from methods like one-hot encoding or TF-IDF, word embeddings provide a dense vector representation of a word. This dense vector captures the semantic meaning of the word and facilitates the identification of similar words. One popular algorithm for generating word embeddings is Word2Vec, which offers two main architectures: Continuous Bag of Words (CBOW) and Skip-gram. For this project, we will use the CBOW architecture to construct word embeddings. In CBOW, the model predicts the target word based on its context words, allowing us to create meaningful embeddings for our textual data.

To train the Word2Vec model using the words from our criteria, we will use the `gensim` library. The training process involves inputting the list of tokenized sentences obtained from the data preprocessing step:

`w2v = Word2Vec(df["tokenized_text"], vector_size=100, window=10, min_count=1, epochs=5)`

The "window" parameter refers to the maximum distance between the predicted current word and the words around it in a sentence, within which the Word2Vec algorithm considers the context for learning word embeddings. To understand the concept of window size, let's consider an example document: “Ability to swallow oral tablets.”  If we set the window size to 2, the Word2Vec algorithm will look at the word before and after the current word in the sentence as its context. So, for the word "oral," the algorithm will consider "tablets" and "swallow" as the context words.

The parameter “vector_size” sets the dimensionality of the word embeddings to 100, meaning each word will be represented by a dense vector of length 100. The “min_count” parameter specifies the minimum frequency threshold for a word to be included in the vocabulary. Words with a frequency below this threshold will be disregarded. In the context of medical data analysis, it is common for specific medical terms to be rare and still carry valuable information. Therefore, to avoid excluding any potentially important words, we will set the "min_count" parameter to 1, ensuring that all words, regardless of frequency, are included in the vocabulary during training.

The Word2Vec model provides embeddings for all the words in a document. With a dimension set to $100$, each word is represented by a $100$-dimensional vector, resulting in $100$ embeddings for each word. While we can utilize all of these embeddings as features for our model, there might be an interest in obtaining a "combined" embedding for each word. To achieve this, we can employ a method for aggregating the word embeddings and generating a document vector. Consider the following example:

Assume that for an observation $n$, we have the following criteria: “written consent and health record”. Then, the list of tokenized text after removing punctuations and stopwords will be: [written, consent, health, record]. For each word, assume we have the following word embeddings with dimensions $1 \times 100$ for each word:

Written: $[0.14, -1.21, 0.48, …]$

Consent: $[0.95, 1.64, -0.43, …]$

Health: $[-0.02, 1.85, -0.94, …]$

Record: $[1.54, 0.73, -1.08, …]$

Then, we obtain the following embeddings for observation $n$:

$$[\quad [0.14, -1.21, 0.48, …],  [0.95, 1.64, -0.43, …],  [-0.02, 1.85, -0.94, …], [1.54, 0.73, -1.08, …] \quad ]$$ 

We want to use these numeric embeddings to train our model. While one approach is to input all individual embeddings as separate features, we might prefer to have a combined embedding for each word. According to the Word2Vec documentation, a common method is to create a document vector for each observation by averaging the embeddings of all the words within the document. For instance, if we have:

$mean(0.14, -1.21, 0.48, …) = a1$

$mean(0.95, 1.64, -0.43, …) = a2$

$mean(-0.02, 1.85, -0.94, …) = a3$

$mean(1.54, 0.73, -1.08, …) = a4$

Then, the document vector for this particular observation will be

$$[a1, \quad a2, \quad a3, \quad a4]$$

We can use this as input for our machine-learning model. Each of the $a_i$ values represents the $i^{th}$ feature/predictor for this particular observation. 

It is important to note that averaging the word embeddings, as described earlier, is just one approach to aggregating the information. However, this method may lead to a loss of relevant word information that could be valuable for the classification task. For instance, in a bag-of-words representation, if a few words are highly relevant to the classification task while most words are irrelevant, the classifier can effectively learn this distinction. However, by averaging the vectors of all the words in the document, the classifier loses this opportunity.

Considering this limitation, it is worth exploring alternative methods such as Doc2Vec, which captures the document-level context along with word embeddings. Another promising solution, in my opinion, is to leverage Recurrent Neural Networks (RNNs), which will be discussed in the LSTM section. RNNs can effectively capture sequential information and are well-suited for processing textual data, allowing the model to potentially learn more nuanced patterns and dependencies within the text.


# Word2Vec Results
Our model obtained an accuracy of $62.13$% and an AUC score of $0.6202$. An AUC score of $0.6202$ suggests that the model possesses some discriminatory power, although its overall performance can be considered fair. To gain further insight into the model's shortcomings, let's examine the confusion matrix below. By analyzing the instances of false positives and false negatives, we can better understand the specific areas where our model struggles.

## False positive: when the model predicts a trial was terminated but it was completed

The model's prediction that the trial was terminated can be justified due to the specific criteria set for participant selection ([NCT01895491](https://clinicaltrials.gov/ct2/show/NCT01895491)). These criteria, like the need for "prior use of six chemotherapies" or specific immunohistochemistry (IHC) requirements, might suggest difficulty in finding eligible participants who meet all the criteria.
However, it's important to note that the trial only required a small number of participants, just 9 individuals. The small participant requirement made it more feasible to identify and enroll volunteers who met all the criteria. Consequently, the trial was able to complete its enrollment process.

## False negative: when the model predicts a trial was completed but it was terminated
In cases where the trial should not have been terminated but was, we observe that the criteria used do not appear overly restrictive or exclusive. Based on these criteria, it would be reasonable to expect the trial to reach completion. Interestingly, the actual reason for termination, as indicated on the website, was because of the death of a participant ([NCT01547923](https://clinicaltrials.gov/ct2/show/NCT01547923)). It is noteworthy that many of these false negative observations had criteria that seemed reasonable and concise. However, the termination of these trials was primarily influenced by factors unrelated to the criteria themselves.
For instance, some trials were terminated due to strategic considerations ([NCT03515551](https://clinicaltrials.gov/ct2/show/NCT03515551),[ NCT03484520](https://clinicaltrials.gov/ct2/show/NCT03484520)) or encountered challenges related to rare diseases ([NCT02137096](https://clinicaltrials.gov/ct2/show/NCT02137096)), leading to poor enrollment. 
In summary, despite the criteria suggesting that the trials should have been completed, external factors such as participant fatalities, strategic decisions, or challenges related to rare diseases influenced their status. These findings highlight the need to incorporate additional features like disease types or trial sponsors into our model to improve its predictive capabilities.

# LSTM 
Now, we will discuss Long Short-Term Memory Networks (LSTMs) a variety of recurrent neural networks (RNNs). We can think of these as recurrent units, or networks with loops in them, allowing information to pass from one unit to the next. Below is a simplified explanation of how LSTM processes the text and performs the classification task.

Before exploring how LSTMs work, we need to construct an embedding matrix of size $n \times m$, where $n$ is the size of the vocab (the number of tokens) and $m$ is the embedding dimension (in this case, there are $100$ embedding dimensions). Note that we are using the actual embeddings and not document vectors to build the embedding matrix. Each row of the embedding matrix corresponds to the embeddings, or the dense vector representation, of a word. We can think of the embedding matrix as a lookup table that maps each word to its dense vector representation. This matrix enables LSTMs to efficiently retrieve the corresponding embeddings for each word. 

Our RNN consists of many LSTM units. In each LSTM unit,  there are two key components: the hidden state and the cell state. These components work together to capture and propagate information through the recurrent connections of the LSTM. Additionally, an LSTM unit contains multiple interacting components, such as the input gate, forget gate, and output gate, which allow it to selectively retain or forget information over time. 


At each time step, the LSTM unit considers the embeddings of a word (retrieved using the embedding matrix) and processes them while taking into account the previous words and their contextual information. The hidden state in each unit captures the information and context of the word input at a specific time step $t$.  The LSTM unit takes the current word embeddings and the previous hidden state as inputs to compute the updated hidden state for that time step. In the first time step ($t = 0$), the first LSTM unit initializes the hidden state based on the input word embeddings. Once the unit processes the first embeddings, it updates the hidden state and passes it to the next LSTM unit at the next time step. LSTMs continue to process the subsequent words in the input sequence, updating the hidden state at each time step. 

![alt text](https://github.com/dnnguyen99/dnnguyen99.github.io/blob/gh-pages/assets/img/lstm.png?raw=true){:width="600px"}
*An overview of how LSTM units work to give binary classification*

To update the cell state, LSTMs use a series of operations involving gates. To update the hidden state, LSTMs use the current input, the previous hidden state, and the current cell state.

![alt text](https://github.com/dnnguyen99/dnnguyen99.github.io/blob/gh-pages/assets/img/lstm_unit.png?raw=true){:width="600px"}
*An LSTM unit. Image credits to: https://towardsdatascience.com/lstm-networks-a-detailed-explanation-8fae6aefc7f9*

In an LSTM unit, several components work together to process information. The forget gate decides what information from the previous cell state should be forgotten based on the previous hidden state and current input. The input gate determines which parts of the current input should be stored in the cell state, while also considering the previous hidden state. These gates help update the cell state.
Next, the output gate decides how much information from the updated cell state should be used to compute the updated hidden state. It uses the previous hidden state and current input to control the output of the cell state. In the illustration above, the updated cell state is passed through a $tanh$ activation function and multiplied by the output gate to obtain the updated hidden state. 
By performing these computations at each time step, the hidden state of the LSTM evolves and captures the context, patterns, and information from the current input and the history of the input sequence up to that point. In summary, the LSTM unit selectively forgets, stores, and outputs information, allowing it to capture and retain relevant context throughout the sequence.

Once the LSTM unit has processed the entire input sequence, the final hidden state will capture the LSTM's encoded understanding of the entire input text. This final hidden state serves as the input to a fully connected layer. In order to obtain a binary classification, we apply an activation function, such as the sigmoid function. The output of this function will be a value ranging from 0 to 1, and by applying a threshold, such as 0.5, we can make a prediction of whether the trial is completed or terminated.

# LSTM Results

The model demonstrates a notable performance with 71.21% accuracy and an AUC score of 0.7119, surpassing our random forest model. An AUC value of 0.7 or higher signifies the model's good performance and its moderate capability to differentiate between terminated and completed trials. To further enhance the model's performance, one promising approach is to incorporate pre-trained word embeddings derived from extensive health records or medical articles. Our current embeddings were trained solely on the criteria of our trials, which might not capture relationships between rare medical terms or nuanced concepts. By using embeddings from a broader medical context ([such as this](https://github.com/ncbi-nlp/BioSentVec#biowordvec)), the model could potentially gain a deeper understanding of the medical domain, leading to a more accurate classification of trial outcomes.
