# -*- coding: utf-8 -*-

import model


model_ = model.Model([101, 50, 50, 50, 1], [60, 80, 80, 80, 101])
model_.train_model('ltstdb_training.csv', epochs=10000, iter_d=2, iter_g=1, mini_batch_size=1024, keep_prob_d=0.5, keep_prob_g=0.5, model_name='model')
a, b, c = model_.evaluate('ltstdb', 'mitdb_normal.csv', 'mitdb_abnormal.csv')
samples = model_.draw_samples('ltstdb', 1)