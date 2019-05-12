import numpy as np
import tensorflow as tf
from collections import namedtuple, Counter


class DataSet:
    def __init__(self, filename):
        self.load_data(filename)

    def load_data(self, filename):
        """
        Process the raw input into a vocabulary dataset.
        """
        with open(filename, encoding='utf-8') as fn:
            words = list(fn.read())
        count = list()
        count.extend(Counter(words).most_common())
        self.word2num = dict()
        self.data = list()
        for word, _ in count:
            self.word2num[word] = len(self.word2num)
        for word in words:
            index = self.word2num[word]
            self.data.append(index)
        self.num2word = dict(zip(self.word2num.values(), self.word2num.keys()))

    def next_batch(self, batch_size, num_seq):
        X = np.zeros((batch_size, num_seq), dtype=np.int32)
        y = np.zeros((batch_size, num_seq), dtype=np.int32)
        start_index = np.random.choice(len(self.data)-(num_seq+1), batch_size)
        for idx, index in enumerate(start_index):
            X[idx] = self.data[index:index+num_seq]
            y[idx] = self.data[index+1:index+num_seq+1]
        return X, y

    def index2word(self, index):
        if isinstance(index, int):
            return self.num2word[index]
        else:
            words = []
            for idx in index:
                words.append(self.num2word[idx])
            return words

    @property
    def size(self):
        return len(self.data)

    @property
    def words(self):
        return self.num2word.values()

    @property
    def vocab_size(self):
        return len(self.num2word.values())


class CharRNN:
    def __init__(self,
                 num_classes=10000,  # vocabulary size
                 batch_size=128,  # batch size
                 num_steps=20,  # the number of time steps
                 lstm_size=128,  # the number of hidden units in lstm model
                 num_layers=2,  # the number of lstm layers
                 learning_rate=0.01,  # learning_rate
                 grad_clip=5,  # regularize gradients by norm
                 is_training=True,  # ensure the model for train or test.
                 train_keep_prob=0.5,  # the keep probability of dropout layer
                 use_embedding=False,  # ensure you need embedding or not
                 embedding_size=128  # the number of hidden units in embedding model
                 ):
        if is_training is False:
            batch_size, num_steps = 1, 1
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.is_training = is_training
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        self._loss_history = []
        self.dataset = None

        tf.reset_default_graph()
        self.session = tf.Session()
        self.build_inputs()
        self._build_net()
        self.build_loss()
        self.build_optimizer()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)
        self.saver.save(
            self.session, './checkpoint/model.ckpt', global_step=1000)

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.batch_size, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(
                self.batch_size, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable(
                        'embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(
                        embedding, self.inputs)

    def _cell(self, lstm_size, keep_prob):
        net = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        if self.is_training and self.train_keep_prob < 1.0:
            net = tf.nn.rnn_cell.DropoutWrapper(
                net, output_keep_prob=keep_prob)
        return net

    def _build_net(self):
        with tf.name_scope('lstm'):
            # deep rnn for num_layers
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [self._cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)])
            self.initial_state = cell.zero_state(self.batch_size, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
                cell, self.lstm_inputs, initial_state=self.initial_state)

            # 通过lstm_outputs得到概率
            # seq_output of shape (batch_size,time_step,num_class)
            # x of shape (batch_size*time_step,num_class)
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])
            self.logits = tf.layers.dense(
                x, units=self.num_classes, name='dense')
            self.prob_prediction = tf.nn.softmax(
                self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=y_reshaped))

    def build_optimizer(self):
        # 使用clipping gradients
        self.global_steps = tf.Variable(0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_steps)

    def train(self, max_steps=5000, log_every_n=200):
        if self.dataset == None:
            raise Exception("The dataset is None, you must load data first.")
        sess = self.session
        # Train network
        new_state = sess.run(self.initial_state)
        for step in range(max_steps):
            x, y = self.dataset.next_batch(self.batch_size, self.num_steps)
            feed = {self.inputs: x,
                    self.targets: y,
                    self.keep_prob: self.train_keep_prob,
                    self.initial_state: new_state}
            batch_loss, new_state, _ = sess.run(
                [self.loss, self.final_state, self.optimizer], feed_dict=feed)
            if (step+1) % 100 == 0:
                current_step = sess.run(self.global_steps)
                self._loss_history.append((current_step, batch_loss))
                if current_step % 2000 == 0:
                    self.save()
            # control the print lines
            if (step+1) % log_every_n == 0:
                print('current step: {}/{}... '.format(step+1, max_steps),
                      'loss: {:.4f}... '.format(batch_loss),
                      '\ttotal step: {}'.format(sess.run(self.global_steps)))

    def sample(self, n_samples, prime):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((self.num_classes, ))  # for prime=[]
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.prob_prediction, self.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, self.num_classes)
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for _ in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.prob_prediction, self.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, self.num_classes)
            samples.append(c)
        return np.array(samples)

    def load_data(self, filepath):
        self.dataset = DataSet(filepath)

    def index2word(self, index):
        return self.dataset.index2word(index)

    def save(self):
        self.saver.save(self.session, './checkpoint/model.ckpt',
                        global_step=self.global_steps)

    def restore(self):
        self.saver.restore(
            self.session, tf.train.latest_checkpoint('./checkpoint'))

    def close(self):
        self.session.close()

    @property
    def loss_history(self):
        return self._loss_history

    @property
    def current_step(self):
        return self.session.run(self.global_steps)


def pick_top_n(preds, vocab_size, top_n=5):
    """
    to select appropriate word.
        :param preds: probability prediction of shape (1, num_classes)
        :param vocab_size: the size of vocabulary
        :param top_n: candidate words
    """
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    # select word randomly.
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
