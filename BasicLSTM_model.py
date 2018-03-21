import pickle
import numpy as np
import tensorflow as tf

texts = np.array(pickle.load(open("vec_texts", "rb")))

print(texts.dtype)

texts_placeholder = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices((texts_placeholder))

dataset = dataset.map(
    lambda src: (
        tf.string_split([src], delimiter=":")).values,
    num_parallel_calls=2)

dataset = dataset.map(
    lambda src: (
        tf.string_to_number(tf.map_fn(lambda x: tf.string_split([x], delimiter=" ").values, src), out_type=tf.float32)
    ),
    num_parallel_calls=2)

dataset = dataset.map(
    lambda src: (
        src, tf.cast(tf.divide(tf.size(src), 50), tf.int32)),
    num_parallel_calls=2)


def batching_func(x):
    return x.padded_batch(
        5, #batch size
        padded_shapes=(
            tf.TensorShape([None,50]),  # src
            tf.TensorShape([])),  # src_len
        padding_values=(
            0.0,  # src
            0))  # len

num_buckets = 5

if num_buckets > 1:

    def key_func(unused_1, src_len):
        # Calculate bucket_width by maximum source sequence length.
        # Pairs with length [0, bucket_width) go to bucket 0, length
        # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
        # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
        # if src_max_len:
        #     bucket_width = (src_max_len + num_buckets - 1) // num_buckets
        # else:
        bucket_width = 10

        # Bucket sentence pairs by the length of their source sentence and target
        # sentence.
        bucket_id = src_len // bucket_width
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))


    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)


    dataset = dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=5))

#dataset = batching_func(dataset)

iterator = dataset.make_initializable_iterator()
y = iterator.get_next()


with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={texts_placeholder: texts})
    res =sess.run(y)
    print(res[0].shape, res[1])
    res =sess.run(y)
    print(res[0].shape, res[1])
    res =sess.run(y)
    print(res[0].shape, res[1])
    res =sess.run(y)
    print(res[0].shape, res[1])
    res =sess.run(y)
    print(res[0].shape, res[1])
