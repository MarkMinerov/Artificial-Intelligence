# Natural Language Processing

Here I describe how to create models to solve **text classification** problem. What layers should we use for this sort of problem, what is [projector.tensorflow.org](https://projector.tensorflow.org) and how to use it. What is `encoder`, `decoder` and so on!

## Text processing

In Machine Learning, there are many (_except of Transformers_) methods using what we can process text, here is a list of the the most used technics and layers that we can use:

- **Text vectorization**: Text vectorization is a powerful algorithm in Machine Learning which is used to convert text or characters into vector of numbers. Vectorization is very useful when we want to use _embeddings_, to extract number representation of a string from embedding layer we first have to vectorize this text using _TextVectorization_ layer.

```python
from tensorflow.keras.layers import TextVectorization

text_vectorizer = TextVectorization(
  max_tokens=vocabulary_len,
  output_sequence_length=output_length,
  output_mode='int',
)

text_vectorizer.adapt(text_data)

print(text_vectorizer([text_data[0]]))
```

- **Embedding**: _Word-embedding_, _character embedding_, _sentence embedding_, _positional embedding_.
- **Recurrent Neural Networks**: using **RNN** we can make our model remember some aspects while learning. **RNN** are not used very widely due to vanishing and exploding gradient problems.

![RNN](./images/RNN.png)

- **Long-Short term memory**: This machine in based on RNN and in widely used in text and speech recognition problem. **LSTM** machine is a complicated structure which consists of **LSTM** cells. Here is how **LSTM** cell looks from inside:

![LSTM](./images/LSTM.webp)

- **Gated Recurrent Unit**: This machine was introduced in 2014 as a machine which is very similar to the LSTM one. You should consider which machine is better for you based on your needs and experiments.

![GRU](./images/GRU.jpg)

- **Bidirectional layer**: Bidirectional layer which cab be based on LSTM or GRU is a way to process input data bidirectionally. It means that we process input data two times:

  - First time we process it from left to right
  - Second time we process it from right to left

  Bidirectional layer can use either LSTM or GRU machine. It is important to say that this machine use two instances of either LSTM or GRU model under the hood. First machine is used to process data in _forward direction_ and second one is used to process data in _backward direction_.

![bidirectional](./images/bidirectional.webp)

- **1D Convolution**: This layer can be used for text processing. The layer works the same way like a `Conv2D` layer. `Conv1D` also has number of filters and kernel size. We can also use `padding` parameter as well as other parameters which are used for convolutional layers.

![conv1d](./images/conv1d.png)

- **Transfer Learning**: It is believed that one of the best ways to create a sentence embedding layer is to use Transfer Learning. There are many already pre-trained models on `TensorHub`, but one of the most useful is `universal-sentence-encoder`, this transfer learning embedding layer has output vector with length of 512 and can be used to embed sentences instead of words.

![USE](./images/USE.png)

- **Transformers**: `TODO`

## Text vectorization

There are several types of vectorization techniques commonly used in NLP, including:

- **Bag-of-Words (BoW)**: BoW is a simple and widely used technique for vectorizing text data. It represents each document as a vector of word counts, where each element of the vector corresponds to a particular word in the vocabulary. This technique ignores the order and structure of the words in the document, but can be effective in many NLP tasks, such as text classification and sentiment analysis.

- **TF-IDF**: Term Frequency-Inverse Document Frequency (TF-IDF) is a variation of BoW that assigns weights to each word in a document based on how often it appears in the document and how common it is across all documents. This technique can help to reduce the importance of common words like "the" and "and" while highlighting more meaningful words.

- **Word embeddings**: Word embeddings are dense, low-dimensional representations of words that capture their meaning and semantic relationships. They are typically learned from large amounts of text data using neural network models like Word2Vec, GloVe, and FastText. Word embeddings are often used as input to deep learning models for NLP tasks such as language modeling, text classification, and machine translation.

- **Character embeddings**: Character embeddings represent words or subwords as sequences of character embeddings, rather than as single tokens. This can be useful for handling out-of-vocabulary words, as well as for modeling morphological variations and spelling variations in text.

- **Sentence embeddings**: Sentence embeddings represent entire sentences as fixed-length vectors that capture their meaning and context. They can be generated using pre-trained models like the Universal Sentence Encoder or InferSent, or learned from scratch using neural network models like the SkipThought model.

- **Positional Embedding**: Positional Embedding is another part of embeddings. This layer has no weights and does not get optimized while learning. This layer encode a sequence of words into a matrix of numbers where each row represents a vector of numbers with predefined length of n.
