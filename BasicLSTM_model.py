import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

emb_size = 50
eos_id = 1

embedings = np.loadtxt("vocab_embedings.txt");
vocab = np.loadtxt("vocab.txt", dtype="str");

dict = {word: embeding for (word, embeding) in zip(vocab, embedings)}

emb_mat = tf.constant(embedings)

vocab = lookup_ops.index_table_from_file(
  "vocab.txt", default_value= 0)


src_dataset = tf.data.TextLineDataset("text.txt")
tgt_dataset = tf.data.TextLineDataset("summary.txt")

dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

#string to token
dataset = dataset.map(
  lambda src, tgt: (
      tf.string_split([src]).values, tf.string_split([tgt]).values),
  num_parallel_calls=2)

#word to index
dataset = dataset.map(
  lambda src, tgt: (tf.cast(vocab.lookup(src), tf.int32),
                    tf.cast(vocab.lookup(tgt), tf.int32)),
  num_parallel_calls=2)


dataset = dataset.map(
  lambda src, tgt: (src,
                    tf.concat((tgt, [eos_id]), 0)),
  num_parallel_calls=2)

#add length
dataset = dataset.map(
    lambda src, summary: (
        src, summary, tf.size(src), tf.size(summary)),
    num_parallel_calls=2)

def batching_func(x):
    return x.padded_batch(
        5, #batch size
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
y = iterator.get_next()

#To get embedings
encoder_emb_inp = tf.nn.embedding_lookup(
          emb_mat, [1, 4, 5])

with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer, feed_dict=None)
    res =sess.run(y)
    print(res[0].shape, res)
    res =sess.run(y)
    print(res[0].shape, res)
    res =sess.run(y)
    print(res[0].shape, res)
    res =sess.run(y)
    print(res[0].shape, res)
    res =sess.run(y)
    print(res[0].shape, res)


    res =sess.run(encoder_emb_inp)
    print(res)
