import unittest
from typing import Optional, Callable
import tensorflow as tf
import numpy as np
import os
# from tensorflow_asr.mwer.monotonic_rnnt_loss import monotonic_rnnt_loss, MonotonicRnntData
from tensorflow_asr.mwer.mwer_loss import mwer_loss, MWERLossData

os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.config.run_functions_eagerly(True)

def finite_difference_gradient(func: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor, epsilon: float) -> tf.Tensor:
    """Approximates gradient numerically

    Source: https://github.com/alexeytochin/tf_seq2seq_losses/
    """
    input_shape = tf.shape(x)[1:]
    input_rank = input_shape.shape[0]
    dim = tf.reduce_prod(input_shape)
    dx = tf.reshape(tf.eye(dim, dtype=x.dtype), shape=tf.concat([tf.constant([1]), tf.reshape(dim, [1]), input_shape], axis=0))
    # shape = [1, dim] + input_shape

    pre_x1 = tf.expand_dims(x, 1) + epsilon * dx
    # shape = [batch_size, dim] + input_shape
    x1 = tf.reshape(pre_x1, shape=tf.concat([tf.constant([-1], dtype=tf.int32), input_shape], axis=0))
    # shape = [batch_size * dim] + input_shape
    x0 = tf.tile(x, multiples=[dim] + [1] * input_rank)

    pre_derivative = (func(x1) - func(x0)) / epsilon
    # shape = [batch_size * dim]
    derivative = tf.reshape(pre_derivative, shape=tf.concat([tf.constant([-1]), input_shape], axis=0))
    # shape = [batch_size] + input_shape
    return derivative


def generate_inputs():
    batch_size = 1
    max_u = 4
    max_t = 6
    vocab_size = 4
    beam_size = 2
    logits = tf.random.uniform(shape=[batch_size, beam_size, max_t, max_u, vocab_size],
                                   minval=0.1, maxval=0.8, seed=42)
    risk_vals = tf.random.uniform([batch_size, beam_size], minval=0.0, maxval=0.99, seed=42)

    labels = []
    labels_length = tf.random.uniform([beam_size], minval=max_u-1, maxval=max_u, dtype=tf.int32)
    labels_length = tf.concat([labels_length, tf.random.uniform([batch_size-1], minval=4, maxval=max_u, dtype=tf.int32)], axis=0)
    maxlen = tf.math.reduce_max(labels_length)
    for i in range(2):
        elem = tf.random.uniform([labels_length[i]], minval=1, maxval=vocab_size, dtype=tf.int32)
        elem = tf.pad(elem, [[0, maxlen - labels_length[i]]])
        labels.append(elem)
    labels = tf.stack(labels)
    labels = tf.expand_dims(labels, axis=0)
    logits_length = tf.constant([max_t, max_t] * batch_size)

    return logits, risk_vals, labels, logits_length, labels_length


class TestMWERLoss(unittest.TestCase):
    def assert_tensors_almost_equal(self, first: tf.Tensor, second: tf.Tensor, places: Optional[int]):
        self.assertAlmostEqual(first=0, second=tf.norm(first - second, ord=np.inf).numpy(), places=places)

    def test_gradient_with_autodifferentiation(self):
        logits, risk_vals, labels, logits_length, labels_length = generate_inputs()

        with tf.GradientTape() as tape:
            tape.watch(logits)
            loss = MWERLossData(logits, labels, logits_length, labels_length, risk_vals).loss_value
        autograd_grad = tape.gradient(loss, logits)

        with tf.GradientTape() as tape:
            tape.watch(logits)
            loss = mwer_loss(logits, risk_vals, labels, logits_length, labels_length)
        grad = tape.gradient(loss, logits)

        self.assert_tensors_almost_equal(grad, autograd_grad, 1)

    def test_gradient_with_finite_difference(self):
        logits, risk_vals, labels, logits_length, labels_length = generate_inputs()

        def loss_fn(logit):
            return mwer_loss(tf.expand_dims(logit, 0), risk_vals, labels, logits_length, labels_length)

        gradient_numerical = finite_difference_gradient(
            func=lambda logits_: tf.vectorized_map(fn=loss_fn, elems=logits_),
            x=logits,
            epsilon=1e-1
        )

        with tf.GradientTape() as tape:
            tape.watch(logits)
            loss = mwer_loss(logits, risk_vals, labels, logits_length, labels_length)

        gradient_analytic = tape.gradient(loss, sources=logits)

        self.assert_tensors_almost_equal(gradient_numerical, gradient_analytic, 1)

if __name__ == "__main__":
    unittest.main()