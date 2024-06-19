# SPAM-SMS-DETECTION
Overview:
SMS spam classification using machine learning is a powerful approach that leverages computational techniques to analyze and interpret text data. This project focuses on developing an AI model to classify SMS messages as either spam or legitimate, using techniques like TF-IDF or word embeddings with classifiers such as Naive Bayes, Logistic Regression, or Support Vector Machines (SVM).

Objective:
The primary goal of this project is to contribute to the detection and prevention of spam messages by enabling timely and accurate classification. By utilizing historical SMS data, the machine learning model aims to learn patterns and relationships within the data, allowing it to accurately identify spam messages.

Dataset:
The project utilizes a carefully curated dataset that includes SMS messages labeled as spam or legitimate. The dataset is crucial for training and evaluating the machine learning model. 
The dataset : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Data Preprocessing:
Prior to model training, the dataset undergoes a thorough preprocessing phase. This involves handling missing values, text cleaning (removing punctuation, converting to lowercase), tokenization, and transforming the text data using techniques like TF-IDF or word embeddings.

Libraries:
  The following libraries were used in this project:
pandas
numpy
sklearn
nltk
matplotlib

Machine Learning Model:
  The machine learning models employed in this project are:
Naive Bayes
Logistic Regression
Support Vector Machine (SVM)

The text data is transformed using:
TF-IDF (Term Frequency-Inverse Document Frequency)
Word embeddings (such as Word2Vec or GloVe)
The choice of model and transformation technique depends on the nature of the data and the specific requirements of the spam classification task.

Training:
The model is trained on a subset of the dataset, where it learns to recognize patterns and relationships between input features (transformed text data) and SMS classifications. During the training phase, hyperparameters are tuned to optimize the model's performance.

Evaluation:
To assess the model's effectiveness, various evaluation metrics are used. Common metrics include accuracy, precision, recall, F1 score, and area under the receiver operating characteristic (ROC) curve. These metrics provide a comprehensive understanding of the model's performance in classifying SMS messages as spam or legitimate.

