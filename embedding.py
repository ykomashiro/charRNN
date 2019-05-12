import numpy as np
import tensorflow as tf
import collections
import random
import math
import zipfile


vocabulary_size = 2000
batch_size = 128
embedding_size = 128
skip_window = 2
num_skips = 2
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_steps = 10000


def read_data(filename):
    """Extract the file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, num_words):
    """
    Process the raw input into a vocabulary dataset.
        :param words: the raw dataset.
        :param num_words: the dataset contain (num_words) command words
    """
    data  =
    count = []
    count.extend(collections.Counter(words).most_common(num_words-1))
    word2num = dict()
    for word, _ in count:
        word2num[word] = len(word2num)
    data = list()
    unk_count = 0
    for word in words:
        if word in word2num:
            index = word2num[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    num2word = dict(zip(word2num.values(), word2num.keys()))
    return data, word2num, num2word, count


word_index = 0


def generate_batch(data, batch_size, num_skip=1, skip_window=2):
    global word_index
    assert batch_size % num_skip == 0
    assert num_skip <= 2*skip_window
    batch = np.ndarray(shape=(batch_size))
    context = np.ndarray(shape=(batch_size, 1))
    span = 2*skip_window+1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[word_index])
        word_index = (word_index+1) % len(data)
    for i in range(batch_size//num_skip):
        target = skip_window
        target_to_avoid = [skip_window]
        for j in range(num_skip):
            while target in target_to_avoid:
                target = random.randint(0, span-1)
            target_to_avoid.append(target)
            batch[i*num_skip+j] = buffer[skip_window]
            context[i*num_skip+j, 0] = buffer[target]
        buffer.append(data[word_index])
        word_index = (word_index+1) % len(data)
    word_index = (word_index + len(data) - span) % len(data)
    return batch, context


filename = './text8.zip'
vocabulary = read_data(filename)
print(vocabulary[:7])
data, word2num, num2word, count = build_dataset(vocabulary, vocabulary_size)


train_input = tf.placeholder(tf.int32, shape=[batch_size])
train_label = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


# Look up embeddings for inputs.
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_input)

# Construct the variables for the softmax
weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                          stddev=1.0 / math.sqrt(embedding_size)))
biases = tf.Variable(tf.zeros([vocabulary_size]))
hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases


train_context = tf.one_hot(train_label, vocabulary_size)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=hidden_out, labels=train_context))
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)


norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalize_embeddings = embeddings/norm
valid_embeddings = tf.nn.embedding_lookup(normalize_embeddings, valid_dataset)
similarity = tf.matmul(
    valid_embeddings, normalize_embeddings, transpose_b=True)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print('initialized')

    average_loss = 0.0
    for step in range(num_steps):
        batch_input, batch_context = generate_batch(data, batch_size)
        feed_dict = {train_input: batch_input, train_label: batch_context}
        _, loss_val = session.run(
            [optimizer, cross_entropy], feed_dict=feed_dict)
        average_loss += loss_val
        if (step+1) % 2000 == 0:
            average_loss /= 2000
            print('Average loss at step ', step+1, ': ', average_loss)
            average_loss = 0
        if (step+1) % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = num2word[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = num2word[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
