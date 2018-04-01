import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers.core import Dense

emb_size = 50
batch_size = 5
eos_id = 1

embedings = np.loadtxt("vocab_embedings.txt");
vocab = np.loadtxt("vocab.txt", dtype="str");
vocab_size = vocab.size

dict = {word: embeding for (word, embeding) in zip(vocab, embedings)}

emb_mat = tf.constant(embedings)

vocab = lookup_ops.index_table_from_file(
    "vocab.txt", default_value=0)

reverse_vocab = tf.contrib.lookup.index_to_string_table_from_file("vocab.txt", default_value='unk')

src_dataset = tf.data.TextLineDataset("text.txt")
tgt_dataset = tf.data.TextLineDataset("summary.txt")

dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

# string to token
dataset = dataset.map(
    lambda src, tgt: (
        tf.string_split([src]).values, tf.string_split([tgt]).values),
    num_parallel_calls=2)

# word to index
dataset = dataset.map(
    lambda src, tgt: (tf.cast(vocab.lookup(src), tf.int32),
                      tf.cast(vocab.lookup(tgt), tf.int32)),
    num_parallel_calls=2)

dataset = dataset.map(
    lambda src, tgt: (src,
                      tf.concat((tgt, [eos_id]), 0)),
    num_parallel_calls=2)

# add length
dataset = dataset.map(
    lambda src, summary: (
        src, summary, tf.size(src), tf.size(summary)),
    num_parallel_calls=2)


def batching_func(x):
    return x.padded_batch(
        batch_size,  # batch size
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),
            tf.TensorShape([]),
            tf.TensorShape([])),  # src_len
        padding_values=(
            eos_id,  # src
            eos_id,  # src
            0,
            0))  # len


dataset = batching_func(dataset)

iterator = dataset.make_initializable_iterator()
(inputs_index, label_index, input_sequence_length, labels_sequence_length) = iterator.get_next()

# To get embedings
encoder_emb_inp = tf.nn.embedding_lookup(
    emb_mat, [1, 4, 5])

with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer, feed_dict=None)
    # res = sess.run(iterator.get_next())

    # inputs_index = y[0]

    embedded_inputs = tf.nn.embedding_lookup(
        emb_mat, inputs_index)

    embedded_labels = tf.nn.embedding_lookup(
        emb_mat, label_index)


    def get_embeding(ids):
        return tf.nn.embedding_lookup(
            emb_mat, ids)


    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)

    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float64,
        sequence_length=input_sequence_length,
        inputs=embedded_inputs)

    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)

    projection_layer = Dense(units=vocab_size, use_bias=False)

    helper = tf.contrib.seq2seq.TrainingHelper(
        embedded_labels, [tf.reduce_max(labels_sequence_length) for _ in range(batch_size)]
        , time_major=False)

    tf.contrib.seq2seq.GreedyEmbeddingHelper(
        get_embeding, [eos_id for _ in range(batch_size)], eos_id)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, encoder_state,
        output_layer=projection_layer)

    outputs, output_states, output_seq_length = tf.contrib.seq2seq.dynamic_decode(
        decoder, output_time_major=False,
        swap_memory=False
    )

    # # calculate loss
    logits = outputs.rnn_output
    train_prediction = outputs.sample_id

    # print("loggiiiittts :", logits.shape)
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_index, logits=logits)
    train_loss = (tf.reduce_sum(crossent
                                * tf.sequence_mask(labels_sequence_length, dtype=logits.dtype)) /
                  batch_size)

    saver = tf.train.Saver()
    saver.restore(sess, "/tmp/model.ckpt")

    for step in range(1000):
        l, pred, o_i = sess.run([train_loss, train_prediction, label_index], feed_dict=None)
        if step == 0:
            print("step 1 loss::", l)
        print(".", step)

        if step % 100 == 0:
            x = reverse_vocab.lookup(tf.constant(pred, tf.int64))
            print(o_i)
            print(pred)
            print([[word for word in x] for x in sess.run(x)])

    print("Loss::", l)
