import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers.core import Dense

emb_size = 50
batch_size = 32
sos_id = 1
eos_id = 2

########################## Data loading and preprocessing #######################

text_vocab = lookup_ops.index_table_from_file(
    "text_vocab.txt", default_value=0)

summary_vocab = lookup_ops.index_table_from_file(
    "summary_vocab.txt", default_value=0)

summary_vocab_size = np.loadtxt("summary_vocab.txt", dtype="str").size

text_embedings = np.loadtxt("text_vocab_embedings.txt");
text_emb_mat = tf.constant(text_embedings)

summary_embedings = np.loadtxt("summary_vocab_embedings.txt");
summary_emb_mat = tf.constant(summary_embedings)

print summary_vocab

reverse_text_vocab = tf.contrib.lookup.index_to_string_table_from_file("text_vocab.txt", default_value='<unk>')
reverse_summary_vocab = tf.contrib.lookup.index_to_string_table_from_file("summary_vocab.txt", default_value='<unk>')

src_dataset = tf.data.TextLineDataset("raw_text.txt")
tgt_dataset = tf.data.TextLineDataset("raw_summary.txt")

dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

# string to token
dataset = dataset.map(
    lambda src, tgt: (
        tf.string_split([src]).values, tf.string_split([tgt]).values),
    num_parallel_calls=2)

# word to index
dataset = dataset.map(
    lambda src, tgt: (tf.cast(text_vocab.lookup(src), tf.int32),
                      tf.cast(summary_vocab.lookup(tgt), tf.int32)),
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

0
with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer, feed_dict=None)


    ########################## Encoder #######################


    embedded_inputs = tf.nn.embedding_lookup(
        text_emb_mat, inputs_index)

    embedded_labels = tf.nn.embedding_lookup(
        summary_emb_mat, target_input)


    cell = tf.nn.rnn_cell.LSTMCell(num_units=128)
    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float64,
        sequence_length=input_sequence_length,
        inputs=embedded_inputs)


    ########################## Decoder #######################


    decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=128)

    projection_layer = Dense(units=summary_vocab_size, use_bias=False)

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


    ########################## Loss and back propogation #######################


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



    ########################## inference #######################


    def get_embeding(ids):
        return tf.nn.embedding_lookup(
            summary_emb_mat, ids)

    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    get_embeding, [sos_id for _ in range(batch_size)], eos_id)

    infer_decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, infer_helper, encoder_state,
        output_layer=projection_layer)

    infer_outputs = tf.contrib.seq2seq.dynamic_decode(
        infer_decoder, output_time_major=False,
        swap_memory=False
    )

    infer_prediction = infer_outputs[0].sample_id


    ################# Training #####################


    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("/tmp/summary/basicModel/1")
    writer.add_graph(sess.graph)
    average_loss = 0;
    for epoch in range(20):
        sess.run(iterator.initializer, feed_dict=None)
        average_loss = 0;
        for step in range(4000): # with batch size 100 this will be 400k data points.
            _, l, pred,t_i, o_i = sess.run([adam_optimize, train_loss, train_prediction,target_input, label_index], feed_dict=None)
            average_loss += l;
            if step%500 == 0 and step != 0:
                print("step "+ step +" loss::", l)
            # if step%100 > 90:
            #     print(".", step)

            if step % 100 == 0:
                x = reverse_summary_vocab.lookup(tf.constant(pred, tf.int64))
                # print("label ::",o_i)
                # print("target input ::", t_i)
                # print(pred)
                print([[word for word in x] for x in sess.run(x)])

        saver = tf.train.Saver()
        save_path = saver.save(sess, "./tmp/model.ckpt")
        print("Epoch::", epoch, "average loss::", average_loss/4000)

