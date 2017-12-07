import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


DIMS = 10
TRAINING_STEPS = 20
LAYERS = 2
STATE_SIZE = 20

def f(x):
    x = scale*x
    return tf.reduce_sum(x*x)


def g_sgd(grad, state, lr=0.1):
    return -lr*grad, state


def g_rms(grad, state, lr=0.1, dr=0.99):
    if state == None:
        state = tf.zeros(DIMS)
    state = dr*state + (1-dr)*tf.pow(grad,2)
    update = -lr*grad / (tf.sqrt(state)+1e-5)
    return update, state


def learn(optimizer):
    losses = []
    x = initial_pos
    state = None
    for _ in range(TRAINING_STEPS):
        loss = f(x)
        losses.append(loss)
        grads, = tf.gradients(loss, x)

        update, state = optimizer(grads, state)
        x += update
    return losses


scale = tf.random_uniform([DIMS], 0.5, 1.5)
initial_pos = tf.random_uniform([DIMS], -1, 1)

sgd_losses = learn(g_sgd)
rms_losses = learn(g_rms)


# ==========================
# Meta-learner here
# ==========================
LAYERS = 2
STATE_SIZE = 20

def g_rnn(grad, state):
    grad = tf.expand_dims(grad, axis=1)

    if state is None:
        state = [[tf.zeros([DIMS, STATE_SIZE])] * 2] * LAYERS
    update, state = cell(grad, state)
    return tf.squeeze(update, axis=[1]), state

cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(STATE_SIZE) for _ in range(LAYERS)])
cell = tf.contrib.rnn.InputProjectionWrapper(cell, STATE_SIZE)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
cell = tf.make_template('cell', cell)

rnn_losses = learn(g_rnn)
sum_losses = tf.reduce_sum(rnn_losses)

def optimize(loss):
    optimizer = tf.train.AdamOptimizer(.0001)
    grad, v = zip(*optimizer.compute_gradients(loss))
    grad, _ = tf.clip_by_global_norm(grad, 1.)
    return optimizer.apply_gradients(zip(grad,v))

apply_update = optimize(sum_losses)


# ==========================
# TF Session here
# ==========================

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


x = range(TRAINING_STEPS)
for _ in range(3):
    sgd_l, rms_l, rnn_l = sess.run([sgd_losses, rms_losses, rnn_losses])
    p1, = plt.plot(x, sgd_l, label = 'SGD')
    p2, = plt.plot(x, rms_l, label = 'RMS')
    p3, = plt.plot(x, rnn_l, label = 'RNN')
    plt.legend(handles=[p1,p2, p3])
    plt.title('Losses')
    plt.show()