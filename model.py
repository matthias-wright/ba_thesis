# -*- coding: utf-8 -*-

"""
This module implements a Generative Adversarial Network with TensorFlow 1.8.0
"""

__author__ = 'Matthias Wright'

import tensorflow as tf
import numpy as np
import data
import metrics


class Model:

    def __init__(self, layers_d, layers_g):
        assert len(layers_d) > 1
        assert len(layers_g) > 1
        assert layers_d[-1] == 1
        assert layers_g[-1] == layers_d[0]
        self.input_size = layers_d[0]
        self.latent_dim = layers_g[0]
        self.layers_d = layers_d
        self.layers_g = layers_g
        self.X = tf.placeholder(tf.float32, shape=[None, self.input_size], name='X')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name='Z')
        self.keep_prob_d = tf.placeholder(tf.float32)
        self.keep_prob_g = tf.placeholder(tf.float32)
        self.theta_d = None
        self.theta_g = None

    def train_model(self, training_set_file, epochs, iter_d, iter_g, mini_batch_size, keep_prob_d, keep_prob_g, model_name):
        """
        Trains the Generative Adversarial Network.
        :param epochs: (int) number of training epochs.
        :param iter_d: (int) number of optimization steps of the discriminator.
        :param iter_g: (int) number of optimization steps of the generator.
        :param mini_batch_size: (int) size of the mini-batches.
        """
        self.theta_d = Model.__init_theta(self.layers_d, is_discriminator=True)
        self.theta_g = Model.__init_theta(self.layers_g, is_discriminator=False)

        sample_g = self.__generator(self.Z, self.keep_prob_g)
        real_d, logit_real_d = self.__discriminator(self.X, self.keep_prob_d)
        fake_d, logit_fake_d = self.__discriminator(sample_g, self.keep_prob_d)

        loss_real_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_real_d, labels=tf.ones_like(logit_real_d)))
        loss_fake_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake_d, labels=tf.zeros_like(logit_fake_d)))
        loss_d = loss_real_d + loss_fake_d
        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake_d, labels=tf.ones_like(logit_fake_d)))

        optimizer_d = tf.train.AdamOptimizer().minimize(loss_d, var_list=self.theta_d)
        optimizer_g = tf.train.AdamOptimizer().minimize(loss_g, var_list=self.theta_g)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        data_ = data.Data(training_set_file)

        for i in range(epochs):

            mini_batches = data_.get_mini_batches(mini_batch_size)

            for j in range(len(mini_batches)):

                for k in range(iter_d):
                    # add Gaussian noise
                    X_mb = mini_batches[j] + np.random.normal(0, 0.1, (mini_batches[j].shape[0], self.input_size))
                    _, _loss_d = sess.run([optimizer_d, loss_d], feed_dict={self.X: X_mb, self.Z: Model.__sample_noise(mini_batch_size, self.latent_dim), self.keep_prob_d: keep_prob_d, self.keep_prob_g: keep_prob_g})
                for k in range(iter_g):
                    _, _loss_g = sess.run([optimizer_g, loss_g], feed_dict={self.Z: Model.__sample_noise(mini_batch_size, self.latent_dim), self.keep_prob_d: keep_prob_d, self.keep_prob_g: keep_prob_g})

            print('Epoch: ' + str(i))
            print('D loss: ' + str(_loss_d))
            print('G loss: ' + str(_loss_g))
            print('')
            print('Samples:')
            samples = sess.run(sample_g, feed_dict={self.Z: Model.__sample_noise(1, self.latent_dim), self.keep_prob_g: keep_prob_g})
            print(samples)
            print('')
            saver.save(sess, 'saved_models/' + model_name)

    def evaluate(self, model_name, test_set_normal, test_set_abnormal):
        """
        Tests the trained model.
        :return: loss_normal: the losses produced by the test set containing normal data,
                 loss_abnormal: the losses produced by the test set containing abnormal data,
                 roc_auc: area under the ROC curve.
        """
        self.theta_d, sess = Model.__restore_theta(self, model_name, is_discriminator=True)
        real, logit_real = self.__discriminator(self.X, tf.constant(1.0, tf.float32))
        test_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_real, labels=tf.ones_like(logit_real))

        data_normal = data.Data(test_set_normal)
        data_abnormal = data.Data(test_set_abnormal)
        X_normal = data_normal.get_all_data()
        X_abnormal = data_abnormal.get_all_data()
        sess.run(tf.global_variables_initializer())
        loss_normal = sess.run(test_loss, feed_dict={self.X: X_normal})
        loss_abnormal = sess.run(test_loss, feed_dict={self.X: X_abnormal})
        roc_auc = metrics.get_auc(loss_normal, loss_abnormal)
        return loss_normal, loss_abnormal, roc_auc

    def draw_samples(self, model_name, num_samples):
        """
        Draws num_samples samles from the generator of the model.
        :param model_name: saved model.
        :param num_samples: number of samples.
        :return: sample batch of size num_samples
        """
        self.theta_g, sess = Model.__restore_theta(self, model_name, is_discriminator=False)
        sample = self.__generator(self.Z, tf.constant(1.0, tf.float32))
        samples = sess.run(sample, feed_dict={self.Z: self.__sample_noise(num_samples, self.latent_dim)})
        return samples

    def __discriminator(self, x, keep_prob):
        """
        Propagates the input x through discriminator.
        :param x: input vector.
        :param keep_prob: (float) for Dropout: probability with which each neuron in every layer 'survives'.
        :return: output vector.
        """
        a_prev = x
        L = len(self.theta_d)
        for l in range(0, L-2, 2):
            W = self.theta_d[l]
            b = self.theta_d[l+1]
            a = tf.nn.tanh(tf.matmul(a_prev, W) + b)
            a = tf.nn.dropout(a, keep_prob)
            a_prev = a
        W = self.theta_d[L-2]
        b = self.theta_d[L-1]
        z_out = tf.matmul(a_prev, W) + b
        a_out = tf.nn.sigmoid(z_out)
        return a_out, z_out

    def __generator(self, z, keep_prob):
        """
        Propagates the noise z through generator.
        :param z: noise vector.
        :param keep_prob: (float) for Dropout: probability with which each neuron in every layer 'survives'.
        :return: generated sample.
        """
        a_prev = z
        L = len(self.theta_g)
        for l in range(0, L-2, 2):
            W = self.theta_g[l]
            b = self.theta_g[l+1]
            a = tf.nn.tanh(tf.matmul(a_prev, W) + b)
            a = tf.nn.dropout(a, keep_prob)
            a_prev = a
        W = self.theta_g[L-2]
        b = self.theta_g[L-1]
        z_out = tf.matmul(a_prev, W) + b
        return z_out

    @staticmethod
    def __sample_noise(m, n):
        """
        Samples a (m x n)-batch from the Uniform distribution with support [-1, 1].
        :param m: (int) size of the mini-batch.
        :param n: (int) size of the sample.
        :return:
        """
        return np.random.uniform(-1., 1., size=[m, n])

    @staticmethod
    def __init_theta(layers, is_discriminator):
        """
        Initializes the weights and bias units of the network.
        :param layers: (list) specifies the number of layers, as well as the number of neurons for each layer.
        :param is_discriminator: (bool) whether we want to initialize the discriminator network or the generator network.
        :return: (list) all the weights and bias units.
        """
        if is_discriminator:
            name_prefix = 'D'
        else:
            name_prefix = 'G'
        theta = []
        L = len(layers)
        for l in range(L-1):
            W = tf.Variable(tf.random_normal([layers[l], layers[l+1]]), name=name_prefix + '_W' + str(l+1))
            b = tf.Variable(tf.zeros(shape=[layers[l+1]]), name=name_prefix + '_b' + str(l+1))
            theta.append(W)
            theta.append(b)

        return theta

    def __restore_theta(self, model_name, is_discriminator):
        """
        Restores the saved weights and bias units of the model.
        :param model_name: name of the saved model.
        :param is_discriminator: whether we want to restore the parameters of the discriminator or the generator.
        :return:
        """
        tf.reset_default_graph()
        graph = tf.train.import_meta_graph('saved_models/' + model_name + '/model.meta')
        sess = tf.Session()
        graph.restore(sess, tf.train.latest_checkpoint('saved_models/' + model_name))
        if is_discriminator:
            name_prefix = 'D'
            num_layers = len(self.layers_d)
        else:
            name_prefix = 'G'
            num_layers = len(self.layers_g)

        theta = []

        for l in range(1, num_layers):
            theta.append(sess.run(name_prefix + '_W' + str(l) + ':0'))
            theta.append(sess.run(name_prefix + '_b' + str(l) + ':0'))

        self.X = sess.graph.get_tensor_by_name(name='X:0')
        self.Z = sess.graph.get_tensor_by_name(name='Z:0')

        return theta, sess




