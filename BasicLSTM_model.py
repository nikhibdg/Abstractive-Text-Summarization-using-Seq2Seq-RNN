import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers.core import Dense

emb_size = 50
batch_size = 10
eos_id = 1
sos_id = 2

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
                      tf.concat(([sos_id], tgt), 0),
                      tf.concat((tgt, [eos_id]), 0)),
    num_parallel_calls=2)

# add length
dataset = dataset.map(
    lambda src, target_input, summary: (
        src, target_input, summary, tf.size(src), tf.size(summary)),
    num_parallel_calls=2)


def batching_func(x):
    return x.padded_batch(
        batch_size,  # batch size
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([]),
            tf.TensorShape([])),  # src_len
        padding_values=(
            eos_id,  # src
            eos_id,  # src
            eos_id,  # src
            0,
            0))  # len


# num_buckets = 5
#
# if num_buckets > 1:
#
#     def key_func(unused_1,unused_2,unused_3, src_len):
#         # Calculate bucket_width by maximum source sequence length.
#         # Pairs with length [0, bucket_width) go to bucket 0, length
#         # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
#         # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
#         # if src_max_len:
#         #     bucket_width = (src_max_len + num_buckets - 1) // num_buckets
#         # else:
#         bucket_width = 10
#
#         # Bucket sentence pairs by the length of their source sentence and target
#         # sentence.
#         bucket_id = src_len // bucket_width
#         return tf.to_int64(tf.minimum(num_buckets, bucket_id))
#
#
#     def reduce_func(unused_key, windowed_data):
#         return batching_func(windowed_data)
#
#
#     dataset = dataset.apply(
#         tf.contrib.data.group_by_window(
#             key_func=key_func, reduce_func=reduce_func, window_size=5))

dataset = batching_func(dataset)

iterator = dataset.make_initializable_iterator()
(inputs_index, target_input, label_index, input_sequence_length, labels_sequence_length) = iterator.get_next()

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

    # print("chkkk --->",inputs_index.shape)
    # print("conversion", embedded_inputs.shape)

    # input_sequence_length = res[2]
    # print("seq shape", res[2].shape)
    # label_index = res[1]
    embedded_labels = tf.nn.embedding_lookup(
        emb_mat, target_input)
    # print("label indexxx------->", label_index.shape)
    # print("eeeeeeeemmmm   label indexxx------->", embedded_labels.shape)

    # labels_sequence_length = res[3]
    # print("out seq shape ", res[3].shape)

    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)
    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float64,
        sequence_length=input_sequence_length,
        inputs=embedded_inputs)

    # sess.run(tf.global_variables_initializer())
    # output = sess.run([encoder_output, encoder_state], feed_dict=None)
    # print(output)

    # result = tf.contrib.learn.run_n(
    #     {"outputs": encoder_output, "last_states": encoder_state},
    #     n=1,
    #     feed_dict=None)
    # print(result[0].get('outputs').shape)

    # #
    # # print(result[0].get('last_states')[0].shape)
    # #
    # # print(result[0].get('last_states')[1].shape)
    #
    # # Decoder

    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)

    projection_layer = Dense(units=vocab_size, use_bias=False)

    helper = tf.contrib.seq2seq.TrainingHelper(
        embedded_labels, [tf.reduce_max(labels_sequence_length) for _ in range(batch_size)]
        , time_major=False)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, encoder_state,
        output_layer=projection_layer)
    """Dnamic decoder"""
    outputs, output_states, output_seq_length = tf.contrib.seq2seq.dynamic_decode(
        decoder, output_time_major=False,
        swap_memory=False
    )

    # result = tf.contrib.learn.run_n(
    #     {"outputs": outputs, "states": output_states},
    #     n=1,
    #     feed_dict=None)

    # sess.run(tf.global_variables_initializer())
    #
    # output = sess.run([outputs], feed_dict=None)
    # print(output[0])
    #
    # output = sess.run([outputs], feed_dict=None)
    # print(output[0])
    #
    # output = sess.run([outputs], feed_dict=None)
    # print(output[0])

    # print(result[0].get('outputs').rnn_output.shape)
    # x = reverse_vocab.lookup(tf.constant(result[0].get('outputs').sample_id, tf.int64))
    # print([[word.decode() for word in x] for x in sess.run(x)])
    #
    #
    # # calculate loss
    logits = outputs.rnn_output

    # print("loggiiiittts :", logits.shape)
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_index, logits=logits)
    train_loss = (tf.reduce_sum(crossent
                                * tf.sequence_mask(labels_sequence_length, dtype=logits.dtype)) /
                  batch_size)

    global_step = tf.Variable(0, trainable=False)
    inc_gstep = tf.assign(global_step, global_step + 1)
    learning_rate = tf.train.exponential_decay(
        0.03, global_step, decay_steps=10, decay_rate=0.9, staircase=True)

    # with tf.variable_scope('Adam'):
    adam_optimizer = tf.train.AdamOptimizer(learning_rate)

    adam_gradients, v = zip(*adam_optimizer.compute_gradients(train_loss))
    adam_gradients, _ = tf.clip_by_global_norm(adam_gradients, 25.0)
    adam_optimize = adam_optimizer.apply_gradients(zip(adam_gradients, v))
    train_prediction = outputs.sample_id

    sess.run(tf.global_variables_initializer())

    average_loss = 0;
    for epoch in range(100):
        sess.run(iterator.initializer, feed_dict=None)
        average_loss = 0;
        for step in range(1000):
            _, l, pred,t_i, o_i = sess.run([adam_optimize, train_loss, train_prediction,target_input, label_index], feed_dict=None)
            average_loss += l;
            if step == 0:
                print("step 1 loss::", l)
            if step%100 > 90:
                print(".", step)

            if step % 100 == 0:
                x = reverse_vocab.lookup(tf.constant(pred, tf.int64))
                print("label ::",o_i)
                print("target input ::", t_i)
                print(pred)
                print([[word for word in x] for x in sess.run(x)])

        saver = tf.train.Saver()
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Epoch::", epoch, "average loss::", average_loss/1000)

    #
    # print(train_loss.eval())
    # print(result[1].size)

    # res =sess.run(encoder_emb_inp)
    # print(res)
