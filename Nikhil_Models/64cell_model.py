
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers.core import Dense
from nltk.translate.bleu_score import corpus_bleu

emb_size = 50
batch_size = 16
sos_id = 1
eos_id = 2
test_steps = 30
epochs = 12
steps = 3000


tf.reset_default_graph()


text_vocab = lookup_ops.index_table_from_file(
    "drive/ML_FINAL/data/text_vocab.txt", default_value=0)

summary_vocab = lookup_ops.index_table_from_file(
    "drive/ML_FINAL/data/summary_vocab.txt", default_value=0)

summary_vocab_size = np.loadtxt("drive/ML_FINAL/data/summary_vocab.txt", dtype="str").size

text_embedings = np.loadtxt("drive/ML_FINAL/data/text_vocab_embedings.txt");
text_emb_mat = tf.constant(text_embedings)

summary_embedings = np.loadtxt("drive/ML_FINAL/data/summary_vocab_embedings.txt");
summary_emb_mat = tf.constant(summary_embedings)

reverse_text_vocab = tf.contrib.lookup.index_to_string_table_from_file("drive/ML_FINAL/data/text_vocab.txt", default_value='<unk>')
reverse_summary_vocab = tf.contrib.lookup.index_to_string_table_from_file("drive/ML_FINAL/data/summary_vocab.txt", default_value='<unk>')

src_dataset = tf.data.TextLineDataset("drive/ML_FINAL/data/raw_text.txt")
tgt_dataset = tf.data.TextLineDataset("drive/ML_FINAL/data/raw_summary.txt")

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

dataset = batching_func(dataset)

iterator = dataset.make_initializable_iterator()
(inputs_index, target_input, label_index, input_sequence_length, labels_sequence_length) = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer, feed_dict=None)

    ########################## Encoder #######################

    embedded_inputs = tf.nn.embedding_lookup(
        text_emb_mat, inputs_index)

    embedded_labels = tf.nn.embedding_lookup(
        summary_emb_mat, target_input)

    cell = tf.nn.rnn_cell.LSTMCell(num_units=64)
    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float64,
        sequence_length=input_sequence_length,
        inputs=embedded_inputs)

    ########################## Decoder #######################

    decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=64)

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
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("drive/ML_FINAL/basicModel/1")
    writer.add_graph(sess.graph)
    average_loss = 0;
    for epoch in range(epochs):
        sess.run(iterator.initializer, feed_dict=None)
        average_loss = 0;
        for step in range(steps):  # with batch size 100 this will be 400k data points.
            _, l, pred, t_i, o_i = sess.run([adam_optimize, train_loss, train_prediction, target_input, label_index],
                                            feed_dict=None)
            average_loss += l;
            if step % 500 == 0 and step != 0:
                print("step " + str(step) + " loss::", average_loss / step)
            # if step%100 > 90:
            #     print(".", step)

            if step % 500 == 0:
                x = reverse_summary_vocab.lookup(tf.constant(pred, tf.int64))
                # print("label ::",o_i)
                # print("target input ::", t_i)
                # print(pred)
                print([[word for word in x] for x in sess.run(x)])

        # saver = tf.train.Saver()
        save_path = saver.save(sess, "drive/ML_FINAL/saved_model/model_{}.ckpt".format(epoch))
        print("Epoch::", epoch, "average loss::", average_loss / steps)

    ########################## BLUE score ###########################

    total_blue_score = 0;
    for step in range(test_steps):  # with batch size 32 this will be 1k data points.
        pred, o_i, seq_lens = sess.run([infer_prediction, label_index, labels_sequence_length], feed_dict=None)
        x = reverse_summary_vocab.lookup(tf.constant(pred, tf.int64))

        pred_sentence = [[word for word in x] for x in sess.run(x)]

        y = reverse_summary_vocab.lookup(tf.constant(o_i, tf.int64))
        label_sentence = [[word for word in o] for o in sess.run(y)]
        label_sentence =  [s[:l] for s, l in zip(label_sentence, seq_lens)]

        try:
            total_blue_score += corpus_bleu(pred_sentence, label_sentence)
        except:
            print("divison error")

    print("average BLUE SCORE :: ", total_blue_score / test_steps)

