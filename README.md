# Suicide Tweets Detection using NLP

## Overview

This research project focuses on the detection of suicidal tweets using Natural Language Processing (NLP). The goal is to develop models capable of identifying tweets that express suicidal thoughts or emotions. Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) were employed for the task.

## Dataset

The model was trained and evaluated on a dataset containing labeled tweets, where each tweet was either expressing suicidal tendencies or not. The dataset was preprocessed to suit the input requirements of the models.

## Models

### 1. Convolutional Neural Network (CNN)

The CNN model used in this project is designed for detecting suicidal tweets using a one-dimensional convolutional neural network (CNN). It takes advantage of word embeddings, specifically GloVe embeddings, to represent the input text. The architecture consists of an embedding layer, a 1D convolutional layer with a kernel size of 3 and padding of 1, followed by max-pooling and linear layers with ReLU activation. Dropout is applied for regularization, and the final output layer is a linear layer with a sigmoid activation function to produce binary classification results (1 for suicidal, 0 for non-suicidal).

In summary, this CNN model processes input embeddings through convolutional and pooling layers, capturing relevant patterns in the text, and then passes them through fully connected layers to make a prediction about the presence of suicidal content in the input tweet. The model architecture is designed to extract hierarchical features from the text data and has been configured with hyperparameters such as the hidden size, dropout rate, and kernel size to optimize performance.

The CNN model achieved its highest F1 score of 0.9327 when utilizing a classification threshold of 0.4800. This threshold represents a well-balanced point between precision and recall, making it an optimal choice for the binary classification of suicidal and non-suicidal tweets.

### 2. Recurrent Neural Network (RNN)
The RNN model employed in this project is designed for detecting suicidal tweets using a Recurrent Neural Network (RNN) architecture. Similar to the CNN model, it utilizes GloVe word embeddings for representing the input text. The model comprises an embedding layer, an RNN layer with a hidden size of 64 and batch-first set to True, followed by a combination of average pooling and max pooling operations. The concatenated result of these pooling operations is passed through a linear layer with ReLU activation, dropout for regularization, and a final output layer with a sigmoid activation for binary classification (1 for suicidal, 0 for non-suicidal).

In summary, this RNN model processes input embeddings sequentially through the RNN layer, capturing contextual information from the text. The combination of average and max pooling enables the model to capture different aspects of the temporal information. The fully connected layers then transform the pooled features for the final prediction. The model's architecture is configured with hyperparameters such as hidden size, dropout rate, and utilizes both average and max pooling to leverage sequential information effectively.

The RNN model, at the identified threshold of 0.4900, demonstrates a remarkable F1 score of 0.9333, indicating its high discriminative power in distinguishing between suicidal and non-suicidal tweets. This performance metric reflects the model's ability to effectively capture the nuanced patterns in the textual data, showcasing its reliability in identifying tweets expressing suicidal tendencies while minimizing misclassifications.
