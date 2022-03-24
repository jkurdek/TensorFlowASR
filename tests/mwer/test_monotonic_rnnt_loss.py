import unittest
from typing import Optional, Callable
import tensorflow as tf
import numpy as np
import os
from tensorflow_asr.mwer.monotonic_rnnt_loss import monotonic_rnnt_loss, MonotonicRnntData

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def finite_difference_gradient(func: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor, epsilon: float) -> tf.Tensor:
    """Approximates gradient numerically using finite difference gradient.

    Inspired by @alexeytochin seq2seq losses tests (github.com/alexeytochin/tf_seq2seq_losses/).
    """
    input_shape = tf.shape(x)[1:]
    dim = tf.reduce_prod(input_shape)
    dx = tf.reshape(
        tf.eye(dim, dtype=x.dtype), shape=tf.concat([tf.constant([1]), tf.reshape(dim, [1]), input_shape], axis=0)
    )
    # shape = [1, dim] + input_shape

    pre_x1 = tf.expand_dims(x, 1) + epsilon * dx
    # shape = [batch_size, dim] + input_shape

    x1 = tf.experimental.numpy.swapaxes(pre_x1, 0, 1)
    # shape = [dim, batch_size] + input_shape

    x0 = tf.expand_dims(x, axis=0)

    pre_derivative = (func(x1) - func(x0)) / epsilon
    # shape = [dim, batch_size]
    pre_derivative = tf.transpose(pre_derivative)
    # shape = [batch_size, dim]
    derivative = tf.reshape(pre_derivative, shape=tf.shape(x))
    # shape = [batch_size] + input_shape
    return derivative


def generate_inputs(batch_size: int, max_t: int, max_u: int, vocab_size: int):

    labels_len = tf.random.uniform(shape=[batch_size - 1], minval=1, maxval=max_u, dtype=tf.int32)
    labels_len = tf.concat([labels_len, tf.constant(max_u, shape=[1])], axis=0)

    inputs_len = tf.random.uniform(shape=[batch_size - 1], minval=1, maxval=max_t, dtype=tf.int32)
    inputs_len = tf.concat([inputs_len, tf.constant(max_t, shape=[1])], axis=0)

    # Monotonic loss requires inputs_len >= labels_len
    inputs_len = tf.math.maximum(inputs_len, labels_len + 1)

    labels = tf.random.uniform(
        shape=[batch_size, max_u], minval=1, maxval=vocab_size + 1, dtype=tf.int32
    ) * tf.sequence_mask(labels_len, maxlen=max_u, dtype=tf.int32)

    logits = tf.random.uniform(shape=[batch_size, max_t, max_u + 1, vocab_size + 1], minval=0.0, maxval=0.99, seed=42)
    return logits, labels, labels_len, inputs_len


class TestRnntLoss(unittest.TestCase):
    def assert_tensors_almost_equal(self, first: tf.Tensor, second: tf.Tensor, places: Optional[int]):
        self.assertAlmostEqual(first=0, second=tf.norm(first - second, ord=np.inf).numpy(), places=places)

    def test_alpha_beta(self):
        """Checks whether bottom left element of beta == top right element of alpha * prob_blank

        From the definition of forward backward variables the loss should be equal to
        the bottom left element of beta and top right element of alpha multiplied by the probability to output blank.
        """
        logits, labels, labels_len, inputs_len = generate_inputs(batch_size=2, max_t=4, max_u=3, vocab_size=3)
        loss_data = MonotonicRnntData(logits, labels, inputs_len, labels_len)

        beta_final = loss_data.log_loss

        idx = tf.stack([inputs_len - 1, labels_len], axis=1)
        alpha_final = tf.gather_nd(loss_data.alpha, idx, batch_dims=1) + tf.gather_nd(
            loss_data.blank_probs, idx, batch_dims=1
        )

        self.assert_tensors_almost_equal(beta_final, alpha_final, 3)

    def test_gradient_with_finite_difference(self):

        logits, labels, labels_len, inputs_len = generate_inputs(batch_size=2, max_t=4, max_u=3, vocab_size=3)

        def loss_fn(logit):
            return monotonic_rnnt_loss(logit, labels, labels_len, inputs_len)

        gradient_numerical = finite_difference_gradient(
            func=lambda logits_: tf.map_fn(fn=loss_fn, elems=logits_), x=logits, epsilon=1e-2
        )

        with tf.GradientTape() as tape:
            tape.watch([logits])
            loss = loss_fn(logits)

        gradient_analytic = tape.gradient(loss, sources=logits)

        self.assert_tensors_almost_equal(gradient_numerical, gradient_analytic, 1)

    def test_small_gradient(self):
        logits = tf.constant(
            [
                [
                    [[0.1, 0.6, 0.1], [0.8, 0.2, 0.3], [0.7, 0.2, 0.1]],
                    [[0.6, 0.2, 0.1], [0.2, 0.8, 0.3], [0.1, 0.8, 0.3]],
                    [[0.1, 0.7, 0.1], [0.1, 0.2, 0.6], [0.1, 0.9, 0.3]],
                ]
            ],
            dtype=tf.float32,
        )

        labels = tf.convert_to_tensor([[1, 2]])
        labels_len = tf.convert_to_tensor([2])
        inputs_len = tf.convert_to_tensor([3])

        expected_loss = tf.constant([-3.5545435])

        loss_data = MonotonicRnntData(logits, labels, inputs_len, labels_len)

        self.assert_tensors_almost_equal(loss_data.log_loss, expected_loss, places=6)
