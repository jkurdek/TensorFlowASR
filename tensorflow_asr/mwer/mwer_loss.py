import tensorflow as tf
from typing import Callable, Dict, Tuple
from tensorflow_asr.mwer.monotonic_rnnt_loss import monotonic_rnnt_loss
from cached_property import cached_property
import fastwer


class MWERLoss:
    def __init__(self,
                 global_batch_size=None,
                 blank=0,
                 name=None):
        self._global_batch_size = global_batch_size
        self._blank = blank
        self._name = name

    def __call__(self,
                 prediction: Dict[str, tf.Tensor],
                 hypothesis: Dict[str, tf.Tensor],
                 text_transcription: Dict[str, tf.Tensor],
                 ) -> tf.Tensor:
        text_hypotheses = text_transcription["text_hypotheses"]  # [batch_size, beam_size]
        text_labels = text_transcription["text_labels"]  # [batch_size, beam_size]

        batch_beam_dim = tf.shape(text_labels)

        text_hypotheses = tf.reshape(text_hypotheses, [-1])
        text_labels = tf.reshape(text_labels, [-1])

        risk_vals = tf.vectorized_map(
            lambda x: self._wer(x[0], x[1]),
            (text_hypotheses, text_labels),
        )

        # risk_vals = tf.py_function(self._wer2, [text_hypotheses, text_labels], Tout=tf.float32)

        # risk_vals = tf.map_fn(
        #     lambda x: self._wer(x[0], x[1]),
        #     (text_hypotheses, text_labels),
        #     dtype=(tf.string, tf.string),
        #     fn_output_signature=tf.TensorSpec([], dtype=tf.float32)
        # )

        risk_vals = tf.reshape(risk_vals, batch_beam_dim)

        logits = prediction["logits"]
        logits_length = prediction["logits_length"]
        labels = hypothesis["labels"]
        labels_length = hypothesis["labels_length"]

        loss = mwer_loss(
            logits,
            risk_vals,
            labels,
            logits_length,
            labels_length,
            self._blank,
        )

        return tf.nn.compute_average_loss(loss, global_batch_size=self._global_batch_size)

    def _wer(self, hypothesis: tf.Tensor, truth: tf.Tensor) -> tf.Tensor:
        return tf.constant(fastwer.score_sent(tf.compat.as_str_any(hypothesis),
                                              tf.compat.as_str_any(truth)))

    # def _wer2(self, hypothesis: tf.Tensor, truth: tf.Tensor) -> tf.Tensor:
    #     hypothesis_arr = [s.numpy() for s in tf.unstack(hypothesis)]
    #     truth_arr = [s.numpy() for s in tf.unstack(truth)]
    #     wers = list(map(fastwer.score_sent, hypothesis_arr, truth_arr))
    #
    #     return tf.constant(wers)


@tf.function
def mwer_loss(
        logits: tf.Tensor,  # [batch_size, beam_size, T, U, V]
        risk_vals: tf.Tensor,  # [batch_size, beam_size]
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
                 risk_vals: tf.Tensor,
                 blank: int = 0
                 ):
        self._logits = logits
        self._labels = labels
        self._logit_length = logit_length
        self._label_length = label_length
        self._risk_vals = risk_vals
        self._blank = blank
        self._batch_beam_dim = tf.shape(risk_vals)

    def grad(self, init_grad):
        return [tf.reshape(init_grad, shape=[-1, 1, 1, 1, 1]) * self.grads]

    @cached_property
    def loss_value(self):
        hypotheses_probabilities, _ = self.rnnt_loss_and_grad
        softmax_probas = tf.nn.softmax(hypotheses_probabilities)
        expected_risk = tf.reduce_sum(softmax_probas * self._risk_vals, axis=1)

        return expected_risk

    @cached_property
    def grads(self) -> tf.Tensor:
        hypotheses_probabilities, rnnt_grad = self.rnnt_loss_and_grad
        probas_normalized = tf.nn.softmax(hypotheses_probabilities)
        risk_diffs = self._risk_vals - tf.expand_dims(self.loss_value, axis=1)

        lhs = probas_normalized * risk_diffs
        grad = -tf.reshape(lhs, tf.concat([self._batch_beam_dim, tf.constant([1, 1, 1])], axis=0)) * rnnt_grad

        return grad

    @cached_property
    def rnnt_loss_and_grad(self) -> Tuple[tf.Tensor, tf.Tensor]:
        logits = tf.reshape(self._logits, tf.concat([[-1], tf.shape(self._logits)[2:]], axis=0))
        labels = tf.reshape(self._labels, tf.concat([[-1], tf.shape(self._labels)[2:]], axis=0))
        logit_length = tf.reshape(self._logit_length, [-1])
        label_length = tf.reshape(self._label_length, [-1])

        with tf.GradientTape() as tape:
            tape.watch(logits)
            rnnt_loss = monotonic_rnnt_loss(logits,
                                            labels,
                                            label_length,
                                            logit_length,
                                            self._blank)
        rnnt_grad = tape.gradient(rnnt_loss, logits)
        rnnt_grad = tf.reshape(rnnt_grad, tf.concat([self._batch_beam_dim, tf.shape(rnnt_grad)[1:]], axis=0))
        rnnt_loss = tf.reshape(rnnt_loss, self._batch_beam_dim)

        return -rnnt_loss, rnnt_grad
