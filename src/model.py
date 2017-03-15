# -*- coding: utf-8 -*-

import tensorflow as tf
from config.config import HYPARMS
import os, time
from .toolbox import fill_feed_dict, do_eval
from .graph import placeholder_inputs,graph_model,calcul_loss,training,evaluation


class Model:
    # train_set = None
    # test_set = None
    # logits = None
    # recog_placebundle = None
    # train_placebundle = None
    # loss = None
    # train_op = None
    # eval_correct = None
    # saver = None

    def __init__(self, recognizer_flag = False):
        if recognizer_flag:
            # make batch_size = 1
            self.init(1)
        else:
            self.init()

    def init(self, batch_size=HYPARMS.batch_size):
        self.sess = tf.Session()
        self.placebundle = placeholder_inputs(batch_size)
        self.load_graph()
        self.init_saver()

        init = tf.initialize_all_variables()
        self.sess.run(init)

    def load_data(self, train_data_set, test_data_set):
        self.train_set = train_data_set
        self.test_set = test_data_set

    def predict(self, input):
        with self.sess.as_default():
            return self.classified.eval(feed_dict = {self.placebundle.x: input,
                                                    self.placebundle.keep_prob: 1})

    def load_graph(self):
        self.placebundle = placeholder_inputs(HYPARMS.batch_size)
        self.logits = graph_model(self.placebundle)
        self.loss = calcul_loss(self.logits, self.placebundle)
        self.train_op = training(self.loss, HYPARMS.learning_rate)
        self.eval_correct = evaluation(self.logits, self.placebundle)

        self.logits = graph_model(self.placebundle)
        self.sftmax = tf.nn.softmax(self.logits)
        self.classified = tf.argmax(self.sftmax,1)

    def init_saver(self):
        self.checkpoint_file = os.path.join(HYPARMS.ckpt_dir, HYPARMS.ckpt_name)
        self.saver = tf.train.Saver()

    def save_saver(self):
        self.saver.save(self.sess, self.checkpoint_file, global_step=self.step)

    def restore_saver(self):
        ckpt = tf.train.get_checkpoint_state(HYPARMS.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model loaded")
        else:
            print("No checkpoint file found")

    def train(self):
        init = tf.initialize_all_variables()
        self.sess.run(init)
        for step in range(HYPARMS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(self.train_set,
                                       self.placebundle.x,
                                       self.placebundle.y_,
                                       self.placebundle.keep_prob,
                                       HYPARMS)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = self.sess.run([self.train_op, self.loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                # summary_str = sess.run(summary, feed_dict=feed_dict)
                # summary_writer.add_summary(summary_str, step)
                # summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == HYPARMS.max_steps:
                self.step = step+1
                self.save_saver()
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(self.sess,
                        self.eval_correct,
                        self.placebundle.x,
                        self.placebundle.y_,
                        self.placebundle.keep_prob,
                        self.train_set)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(self.sess,
                        self.eval_correct,
                        self.placebundle.x,
                        self.placebundle.y_,
                        self.placebundle.keep_prob,
                        self.test_set)