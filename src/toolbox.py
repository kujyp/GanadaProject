import tensorflow as tf
from config.config import HYPARMS


def fill_feed_dict(data_set, images_pl, labels_pl, keep_prob, evalflag=False):
    images_feed, labels_feed = data_set.next_batch(HYPARMS.batch_size)
    if evalflag:
        dropout_rate = 1
    else:
        dropout_rate = HYPARMS.dropout_rate
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        keep_prob: dropout_rate,
    }
    return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            keep_prob,
            data_set):

  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // HYPARMS.batch_size
  num_examples = steps_per_epoch * HYPARMS.batch_size
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder,
                               keep_prob,
                               True)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

