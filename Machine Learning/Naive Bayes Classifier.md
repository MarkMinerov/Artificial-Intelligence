# Naive Bayes Classifier

The Naive Bayes classifier is based on two essential assumptions:

- **Conditional Independence** - All features are independent of each other. This implies that one feature does not affect the performance of the other. This is the sole reason behind the ‘Naive’ in ‘Naive Bayes.’

- **Feature Importance** - All features are equally important. It is essential to know all the features to make good predictions and get the most accurate results.

## Bernoulli Naive Bayes

This alhorithm is used for binary classification when our features are binary. It is based on Bernoulli's binary distribution and assess probabilities of appearing each feature in each class.

**Bernoulli Naive Bayes** is a part of the Naive Bayes family. It is based on the Bernoulli Distribution and accepts only binary values, i.e., 0 or 1. If the features of the dataset are binary, then we can assume that Bernoulli Naive Bayes is the algorithm to be used.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

# training data
train_data = [
    'this is a spam email',
    'this is not a spam email',
    'this is a spam message',
    'this is a test message not a spam'
]

# class labels
train_labels = [1, 0, 1, 0] # 1 represents spam, 0 represents not spam

# create a vectorizer object to transform text into word frequency vectors
vectorizer = CountVectorizer(binary=True)

# transform the training data into binary word presence vectors
train_vectors = vectorizer.fit_transform(train_data)

# # create a Bernoulli Naive Bayes classifier and train it on the training data
classifier = BernoulliNB()
classifier.fit(train_vectors, train_labels)

# # test data
test_data = ['this is a test message']

# # convert the test data into a binary word presence vector
test_vector = vectorizer.transform(test_data)

# # predict the class of the test data using the trained classifier
predicted_class = classifier.predict(test_vector)
print(predicted_class)
```

## Multinomial Naive Bayes

**Multinomial Naive Bayes (MNB)** is a naive Bayes classifier used for text data classification. It is based on the multinomial distribution model, where each document is represented as a vector of word frequencies in it. This algorithm is used in tasks such as email topic determination, language detection, sentiment analysis of reviews, and more.

Let's consider a simple example to understand how MNB works. Suppose we have a set of documents that we want to classify into two classes - spam and not spam. We can use MNB to determine which class each document belongs to. To do this, we need to build a model based on our data. First, we split our dataset into training and testing sets. Then, we calculate the frequency of each word in each document in the training set and use it to determine the probability of each word in each class. The probability of each class is also calculated based on the number of documents in each class.

When we get a new document that needs to be classified, we calculate its frequency vector using the same words as in the training set. Then, we use our model to determine the probability that this document belongs to each class. Finally, we choose the class with the highest probability and assign this document to that class.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# training data
train_data = ['this is a spam email', 'this is not a spam email', 'this is a spam message']

# class labels
train_labels = ['spam', 'not spam', 'spam']

# create a vectorizer object to transform text into word frequency vectors
vectorizer = CountVectorizer()

# transform the training data into word frequency vectors
train_vectors = vectorizer.fit_transform(train_data)

# create a Multinomial Naive Bayes classifier and train it on the training data
classifier = MultinomialNB()
classifier.fit(train_vectors, train_labels)

# test data
test_data = ['this is a test message']

# convert the test data into a word frequency vector
test_vector = vectorizer.transform(test_data)

# predict the class of the test data using the trained classifier
predicted_class = classifier.predict(test_vector)
print(predicted_class)
```

## Gaussian Naive Bayes

**Gaussian Naive Bayes (GNB)** is a variant of the Naive Bayes algorithm that assumes that the features (input variables) are normally distributed. It is particularly useful for classification problems where the features are continuous (i.e., real-valued) and have a bell-shaped distribution.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load the iris dataset
iris = load_iris()

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# create a GNB classifier and fit it to the training data
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# predict the classes of the test data
y_pred = gnb.predict(X_test)

# calculate the accuracy of the classifier
accuracy = (y_pred == y_test).sum() / len(y_test)
print("Accuracy:", accuracy)
```
