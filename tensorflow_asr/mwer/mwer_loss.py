import tensorflow as tf
from typing import Callable, Dict
from tensorflow_asr.mwer.monotonic_rnnt_loss import monotonic_rnnt_loss
from functools import lru_cache


class MWERLoss:
    def __init__(self,
                 risk_obj: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 global_batch_size=None,
                 blank=0,
                 name=None):
        self._risk_obj = risk_obj
        self._global_batch_size = global_batch_size
        self._blank = blank
        self._name = name

    def __call__(self,
                 prediction: Dict[str, tf.Tensor],
                 label: Dict[str, tf.Tensor],
                 hypotheses: Dict[str, tf.Tensor],
                 ) -> tf.Tensor:
        sentences = hypotheses["sentences"]  # [batch_size, beam_size]
        ground_truths = hypotheses["labels"]  # [batch_size, beam_size]

        batch_beam_dim = tf.shape(ground_truths)

        sentences = tf.reshape(sentences, [-1])
        ground_truths = tf.reshape(ground_truths, [-1])

        risk_vals = tf.py_function(self._risk_obj, [sentences, ground_truths], Tout=tf.float32)
        risk_vals = tf.reshape(risk_vals, batch_beam_dim)

        logits = prediction["logits"]
        logits_length = prediction["logits_length"]
        log_probas = hypotheses["log_probas"]
        labels = label["labels"]
        labels_length = label["labels_length"]

        loss = mwer_loss(
            logits,
            risk_vals,
            log_probas,
            labels,
            logits_length,
            labels_length,
            self._blank,
        )

        return tf.nn.compute_average_loss(loss, global_batch_size=self._global_batch_size)


@tf.function(experimental_relax_shapes=True)
def mwer_loss(
        logits: tf.Tensor,  # [batch_size, beam_size, T, U, V]
        risk_vals: tf.Tensor,  # [batch_size, beam_size]
        hypotheses_log_probas: tf.Tensor,  # [batch_size, beam_size]
        labels: tf.Tensor,  # [batch_size, beam_size, max_label_length]
        logit_length: tf.Tensor,  # [batch_size, beam_size]
        label_length: tf.Tensor,  # [batch_size, beam_size]
        blank: int = 0,
):
    @tf.custom_gradient
    def compute_grads(input_logits: tf.Tensor):
        loss_data = MWERLossData(
            logits=input_logits,
            risk_vals=risk_vals,
            hypotheses_log_probas=hypotheses_log_probas,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank=blank
        )

        return loss_data.loss_value, loss_data.grad

    return compute_grads(logits)


class MWERLossData:
    def __init__(self,
                 logits: tf.Tensor,
                 labels: tf.Tensor,
                 logit_length: tf.Tensor,
                 label_length: tf.Tensor,
                 hypotheses_log_probas: tf.Tensor,
                 risk_vals: tf.Tensor,
                 blank: int = 0
                 ):
        self._logits = logits
        self._labels = labels
        self._logit_length = logit_length
        self._label_length = label_length
        self._hypotheses_log_probas = hypotheses_log_probas
        self._risk_vals = risk_vals
        self._blank = blank
        self._batch_beam_dim = tf.shape(risk_vals)

    def grad(self, init_grad):
        return [tf.reshape(init_grad, shape=[-1, 1, 1, 1]) * self._get_grads()]
    @property
    @lru_cache()
    def loss_value(self):
        softmax_probas = tf.nn.softmax(self._hypotheses_log_probas)
        expected_risk = tf.reduce_sum(softmax_probas * self._risk_vals, axis=1)
        return expected_risk

    def _get_grads(self) -> tf.Tensor:
        logits = tf.reshape(self._logits, tf.concat([[-1], tf.shape(self._logits)[2:]], axis=0))
        labels = tf.reshape(self._labels, tf.concat([[-1], tf.shape(self._labels)[2:]], axis=0))
        logit_length = tf.reshape(self._logit_length, [-1])
        label_length = tf.reshape(self._label_length, [-1])

        probas_normalized = tf.nn.softmax(self._hypotheses_log_probas)
        risk_diffs = self._risk_vals - tf.expand_dims(self.loss_value, axis=1)
        lhs = probas_normalized * risk_diffs

        with tf.GradientTape() as tape:
            tape.watch(logits)
            rnn_loss_val = monotonic_rnnt_loss(logits,
                                               labels,
                                               label_length,
                                               logit_length,
                                               self._blank)
        rhs = tape.gradient(rnn_loss_val, logits)
        grad = tf.reshape(lhs, [-1, 1, 1, 1]) * rhs
        grad = tf.reshape(grad, tf.concat([self._batch_beam_dim, tf.shape(grad)[1:]], axis=0))

        return grad
