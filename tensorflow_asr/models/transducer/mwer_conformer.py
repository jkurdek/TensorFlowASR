from typing import Dict, Tuple

import tensorflow as tf

from tensorflow_asr.models.transducer.base_transducer import Transducer
from tensorflow_asr.models.transducer.conformer import Conformer
from tensorflow_asr.mwer.monotonic_rnnt_loss import MonotonicRnntLoss
from tensorflow_asr.mwer.mwer_loss import MWERLoss
from tensorflow_asr.utils import math_util, data_util


class MWERConformer(Conformer):
    def compile(
            self,
            optimizer,
            global_batch_size,
            blank=0,
            run_eagerly=None,
            **kwargs,
    ):
        self.rnnt_loss = MonotonicRnntLoss(blank=blank, global_batch_size=global_batch_size)
        loss = MWERLoss(blank=blank, global_batch_size=global_batch_size)

        super(Transducer, self).compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)

    def train_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]): a batch of training data

        Returns:
            Dict[tf.Tensor]: a dict of validation metrics with keys are the name of metric

        """
        inputs, y_true = batch

        with tf.GradientTape() as tape:
            # Calculating rnn-t loss
            encoder_out = self.encoder(inputs["inputs"], training=True)
            predictor_out = self.predict_net([inputs["predictions"], inputs["predictions_length"]], training=True)
            logits = self.joint_net([encoder_out, predictor_out], training=True)
            logits_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
            rnnt_input = data_util.create_logits(logits, logits_length)
            rnnt_loss = self.rnnt_loss(y_true, rnnt_input)

            # MWER loss
            with tape.stop_recording():
                # Calculating top beamsearch paths
                encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
                hypotheses, hypotheses_lengths, text_transcriptions = self._get_beam_hypotheses(encoder_out,
                                                                                                encoded_length,
                                                                                                y_true)
                ground_truths = data_util.create_labels(hypotheses, hypotheses_lengths)
                predictor_inputs, predictor_lengths = self._prepare_rnnt_input(hypotheses, hypotheses_lengths)

            # Calculating the actual loss
            encoder_out, encoder_out_lengths = self._tile_encoder_out(encoder_out, inputs["inputs_length"])
            predictor_out = self.predict_net([predictor_inputs, predictor_lengths], training=True)
            logits = self.joint_net([encoder_out, predictor_out], training=True)
            logits_length = math_util.get_reduced_length(encoder_out_lengths, self.time_reduction_factor)

            y_pred = self._parse_rnn_output(logits, logits_length, tf.shape(hypotheses_lengths))
            mwer_loss = self.loss(y_pred, ground_truths, text_transcriptions)

            # Interpolating mwer loss with rnnt loss
            loss = rnnt_loss + mwer_loss

            if self.use_loss_scale:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        if self.use_loss_scale:
            gradients = tape.gradient(scaled_loss, self.trainable_weights)
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        else:
            gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self._tfasr_metrics["loss"].update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]: a batch of validation data

        Returns:
            Dict[tf.Tensor]: a dict of validation metrics with keys are the name of metric prefixed with "val_"

        """
        inputs, y_true = batch

        # Calculating rnn-t loss
        encoder_out = self.encoder(inputs["inputs"], training=True)
        predictor_out = self.predict_net([inputs["predictions"], inputs["predictions_length"]], training=True)
        logits = self.joint_net([encoder_out, predictor_out], training=True)
        logits_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        rnnt_input = data_util.create_logits(logits, logits_length)
        rnnt_loss = self.rnnt_loss(y_true, rnnt_input)

        # Calculating mwer loss
        encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        hypotheses, hypotheses_lengths, text_transcriptions = self._get_beam_hypotheses(encoder_out, encoded_length, y_true)
        predictor_inputs, predictor_lengths = self._prepare_rnnt_input(hypotheses, hypotheses_lengths)
        ground_truths = data_util.create_labels(hypotheses, hypotheses_lengths)

        encoder_out, encoder_out_lengths = self._tile_encoder_out(encoder_out, inputs["inputs_length"])
        predictor_out = self.predict_net([predictor_inputs, predictor_lengths], training=True)
        logits = self.joint_net([encoder_out, predictor_out], training=True)
        logits_length = math_util.get_reduced_length(encoder_out_lengths, self.time_reduction_factor)
        y_pred = self._parse_rnn_output(logits, logits_length, tf.shape(hypotheses_lengths))
        mwer_loss = self.loss(y_pred, ground_truths, text_transcriptions)

        # Interpolating mwer loss with rnnt loss
        loss = rnnt_loss + mwer_loss

        self._tfasr_metrics["loss"].update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def _parse_rnn_output(self,
                          logits: tf.Tensor,
                          logits_length: tf.Tensor,
                          # rnnt_output: Dict[str, tf.Tensor],
                          batch_beam_dim: tf.Tensor
                          ) -> Dict[str, tf.Tensor]:
        logits_out_dim = tf.concat([batch_beam_dim, tf.shape(logits)[1:]], axis=0)
        logits = tf.reshape(logits, logits_out_dim)
        logits_length = tf.reshape(logits_length, batch_beam_dim)

        return data_util.create_logits(logits=logits, logits_length=logits_length)

    def _prepare_rnnt_input(self,
                            hypotheses: tf.Tensor,
                            hypotheses_lengths: tf.Tensor,
                            ) -> Tuple[tf.Tensor, tf.Tensor]:
        hypotheses = tf.reshape(hypotheses, [-1, tf.shape(hypotheses)[2]])
        hypotheses_lengths = tf.reshape(hypotheses_lengths, [-1])
        blank_slice = tf.ones([tf.shape(hypotheses)[0], 1], dtype=tf.int32) * self.text_featurizer.blank
        rnnt_input_hypotheses = tf.concat([blank_slice, hypotheses], axis=1)

        return rnnt_input_hypotheses, hypotheses_lengths + tf.constant(1)

    def _tile_encoder_out(self,
                          features: tf.Tensor,
                          features_length: tf.Tensor,
                          ) -> Tuple[tf.Tensor, tf.Tensor]:
        # copying input features lengths for each prediction
        features_length = tf.expand_dims(features_length, axis=0)
        features_length = tf.transpose(features_length)
        features_length = tf.tile(features_length, [1, self._beam_size])
        features_length = tf.reshape(features_length, [-1])

        # copying input features for each prediction
        features_shape = tf.shape(features)
        features = tf.tile(features, [self._beam_size, 1, 1])
        features = tf.reshape(features, [self._beam_size] + tf.unstack(features_shape))
        features = tf.transpose(features, [1, 0, 2, 3])
        features = tf.reshape(features, [-1] + tf.unstack(features_shape[1:]))

        return features, features_length

    def _get_beam_hypotheses(self,
                             encoded: tf.Tensor,
                             encoded_length: tf.Tensor,
                             y_true: Dict[str, tf.Tensor]
                             ) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        text_labels = self.text_featurizer.iextract(y_true["labels"])
        hypotheses, log_probas = self.beam.call(encoded=encoded,
                                                encoded_length=encoded_length,
                                                return_topk=True,
                                                parallel_iterations=1)

        with tf.device('/cpu:0'):
            text_hypotheses = tf.map_fn(self.text_featurizer.iextract, hypotheses, fn_output_signature=tf.string)

        batch_beam_dims = tf.shape(log_probas)

        text_labels = tf.expand_dims(text_labels, axis=1)
        text_labels = tf.tile(text_labels, [1, self._beam_size])

        text_transcriptions = data_util.create_hypotheses(sentences=text_hypotheses,
                                                          labels=text_labels)

        max_length = tf.shape(hypotheses)[2]
        hypotheses = tf.reshape(hypotheses, [-1, tf.shape(hypotheses)[2]])

        nonblank_tokens = tf.math.not_equal(hypotheses, self.text_featurizer.blank)
        nonblank_tokens = tf.cast(nonblank_tokens, dtype=tf.int32)

        hypotheses_length = tf.reduce_sum(nonblank_tokens, axis=1)
        hypotheses_length = tf.reshape(hypotheses_length, batch_beam_dims)

        # max_length = tf.reduce_max(hypotheses_length)

        def remove_zeros_and_pad(input: tf.Tensor):
            nonblank = tf.where(input != tf.constant(self.text_featurizer.blank))
            nonblank_count = tf.math.count_nonzero(nonblank, dtype=tf.int32)
            nonblank_tokens = tf.gather_nd(input, nonblank)

            return tf.pad(nonblank_tokens, [[0, max_length - nonblank_count]])

        hypotheses = tf.map_fn(remove_zeros_and_pad, hypotheses)
        hypotheses = tf.reshape(hypotheses, tf.concat([batch_beam_dims, [-1]], axis=0))

        return hypotheses, hypotheses_length, text_transcriptions