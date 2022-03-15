# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" https://arxiv.org/pdf/1811.06621.pdf """

import collections
from typing import Dict, Tuple
import tensorflow as tf

from tensorflow_asr.models.transducer.transducer_prediction import TransducerPrediction
from tensorflow_asr.models.transducer.transducer_joint import TransducerJoint
from tensorflow_asr.featurizers.speech_featurizers import SpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import TextFeaturizer
from tensorflow_asr.losses.rnnt_loss import RnntLoss
from tensorflow_asr.mwer.beam_search import BeamSearch
from tensorflow_asr.mwer.monotonic_rnnt_loss import MonotonicRnntLoss
from tensorflow_asr.mwer.mwer_loss import MWERLoss
from tensorflow_asr.utils import data_util, layer_util, math_util, shape_util
from tensorflow_asr.models.base_model import BaseModel

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))

BeamHypothesis = collections.namedtuple("BeamHypothesis", ("score", "indices", "prediction", "states"))


class Transducer(BaseModel):
    """Transducer Model Warper"""

    def __init__(
            self,
            encoder: tf.keras.Model,
            vocabulary_size: int,
            blank_token: int = 0,  # TODO: clean up the code such that blank is non-optional argument
            # beam_size: int = 2, # TODO: change to 1
            embed_dim: int = 512,
            embed_dropout: float = 0,
            num_rnns: int = 1,
            rnn_units: int = 320,
            rnn_type: str = "lstm",
            rnn_implementation: int = 2,
            layer_norm: bool = True,
            projection_units: int = 0,
            prediction_trainable: bool = True,
            joint_dim: int = 1024,
            joint_activation: str = "tanh",
            prejoint_linear: bool = True,
            postjoint_linear: bool = False,
            joint_mode: str = "add",
            joint_trainable: bool = True,
            kernel_regularizer=None,
            bias_regularizer=None,
            mwer_training=False,
            name="transducer",
            **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.encoder = encoder
        self.predict_net = TransducerPrediction(
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_rnns=num_rnns,
            rnn_units=rnn_units,
            rnn_type=rnn_type,
            rnn_implementation=rnn_implementation,
            layer_norm=layer_norm,
            projection_units=projection_units,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=prediction_trainable,
            name=f"{name}_prediction",
        )
        self.joint_net = TransducerJoint(
            vocabulary_size=vocabulary_size,
            joint_dim=joint_dim,
            activation=joint_activation,
            prejoint_linear=prejoint_linear,
            postjoint_linear=postjoint_linear,
            joint_mode=joint_mode,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=joint_trainable,
            name=f"{name}_joint",
        )
        self.beam = BeamSearch(
            vocabulary_size=vocabulary_size,
            predict_net=self.predict_net,
            joint_net=self.joint_net,
            blank_token=blank_token,
            name=f"{name}_beam_search"
        )
        self.time_reduction_factor = 1
        self._beam_size = None
        self._batch_size = None
        self._mwer_training = mwer_training

    def make(
            self,
            input_shape,
            prediction_shape=[None],
            batch_size=None,
    ):
        self._batch_size = batch_size
        inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        inputs_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        predictions = tf.keras.Input(shape=prediction_shape, batch_size=batch_size, dtype=tf.int32)
        predictions_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        self(
            data_util.create_inputs(
                inputs=inputs,
                inputs_length=inputs_length,
                predictions=predictions,
                predictions_length=predictions_length,
            ),
            training=False,
        )

    def summary(
            self,
            line_length=None,
            **kwargs,
    ):
        if self.encoder is not None:
            self.encoder.summary(line_length=line_length, **kwargs)
        self.predict_net.summary(line_length=line_length, **kwargs)
        self.joint_net.summary(line_length=line_length, **kwargs)
        super(Transducer, self).summary(line_length=line_length, **kwargs)

    def add_featurizers(
            self,
            speech_featurizer: SpeechFeaturizer,
            text_featurizer: TextFeaturizer,
    ):
        """
        Function to add featurizer to model to convert to end2end tflite
        Args:
            speech_featurizer: SpeechFeaturizer instance
            text_featurizer: TextFeaturizer instance
        """
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.beam.beam_size = text_featurizer.decoder_config.beam_width
        self._beam_size = text_featurizer.decoder_config.beam_width

    def compile(
            self,
            optimizer,
            global_batch_size,
            blank=0,
            run_eagerly=None,
            **kwargs,
    ):
        loss = MonotonicRnntLoss(blank=blank, global_batch_size=global_batch_size)
        super().compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)

    def call(
            self,
            inputs,
            training=False,
            **kwargs,
    ):
        enc = self.encoder(inputs["inputs"], training=training, **kwargs)
        pred = self.predict_net([inputs["predictions"], inputs["predictions_length"]], training=training, **kwargs)
        logits = self.joint_net([enc, pred], training=training, **kwargs)

        return data_util.create_logits(
            logits=logits,
            logits_length=math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor),
        )


    # -------------------------------- INFERENCES -------------------------------------

    def encoder_inference(
            self,
            features: tf.Tensor,
    ):
        """Infer function for encoder (or encoders)

        Args:
            features (tf.Tensor): features with shape [T, F, C]

        Returns:
            tf.Tensor: output of encoders with shape [T, E]
        """
        with tf.name_scope(f"{self.name}_encoder"):
            outputs = tf.expand_dims(features, axis=0)
            outputs = self.encoder(outputs, training=False)
            return tf.squeeze(outputs, axis=0)

    def decoder_inference(
            self,
            encoded: tf.Tensor,
            predicted: tf.Tensor,
            states: tf.Tensor,
            tflite: bool = False,
    ):
        """Infer function for decoder

        Args:
            encoded (tf.Tensor): output of encoder at each time step => shape [E]
            predicted (tf.Tensor): last character index of predicted sequence => shape []
            states (nested lists of tf.Tensor): states returned by rnn layers

        Returns:
            (ytu, new_states)
        """
        with tf.name_scope(f"{self.name}_decoder"):
            encoded = tf.reshape(encoded, [1, 1, -1])  # [E] => [1, 1, E]
            predicted = tf.reshape(predicted, [1, 1])  # [] => [1, 1]
            y, new_states = self.predict_net.recognize(predicted, states, tflite=tflite)  # [1, 1, P], states
            ytu = tf.nn.log_softmax(self.joint_net([encoded, y], training=False))  # [1, 1, V]
            ytu = tf.reshape(ytu, shape=[-1])  # [1, 1, V] => [V]
            return ytu, new_states

    def get_config(self):
        conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf

    # -------------------------------- GREEDY -------------------------------------

    @tf.function
    def recognize(
            self,
            inputs: Dict[str, tf.Tensor],
    ):
        """
        RNN Transducer Greedy decoding
        Args:
            features (tf.Tensor): a batch of extracted features
            input_length (tf.Tensor): a batch of extracted features length

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded = self.encoder(inputs["inputs"], training=False)
        encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        return self._perform_greedy_batch(encoded=encoded, encoded_length=encoded_length)

    def recognize_tflite(
            self,
            signal,
            predicted,
            states,
    ):
        """
        Function to convert to tflite using greedy decoding (default streaming mode)
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal
            predicted: last predicted character with shape []
            states: lastest rnn states with shape [num_rnns, 1 or 2, 1, P]

        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
            predicted: last predicted character with shape []
            states: lastest rnn states with shape [num_rnns, 1 or 2, 1, P]
        """
        features = self.speech_featurizer.tf_extract(signal)
        encoded = self.encoder_inference(features)
        hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, states, tflite=True)
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)

        return transcript, hypothesis.index, hypothesis.states

    def recognize_tflite_with_timestamp(
            self,
            signal,
            predicted,
            states,
    ):
        features = self.speech_featurizer.tf_extract(signal)
        encoded = self.encoder_inference(features)
        hypothesis = self._perform_greedy(encoded, tf.shape(encoded)[0], predicted, states, tflite=True)
        indices = self.text_featurizer.normalize_indices(hypothesis.prediction)
        upoints = tf.gather_nd(self.text_featurizer.upoints,
                               tf.expand_dims(indices, axis=-1))  # [None, max_subword_length]

        num_samples = tf.cast(tf.shape(signal)[0], dtype=tf.float32)
        total_time_reduction_factor = self.time_reduction_factor * self.speech_featurizer.frame_step

        stime = tf.range(0, num_samples, delta=total_time_reduction_factor, dtype=tf.float32)
        stime /= tf.cast(self.speech_featurizer.sample_rate, dtype=tf.float32)

        etime = tf.range(total_time_reduction_factor, num_samples, delta=total_time_reduction_factor, dtype=tf.float32)
        etime /= tf.cast(self.speech_featurizer.sample_rate, dtype=tf.float32)

        non_blank = tf.where(tf.not_equal(upoints, 0))
        non_blank_transcript = tf.gather_nd(upoints, non_blank)
        non_blank_stime = tf.gather_nd(tf.repeat(tf.expand_dims(stime, axis=-1), tf.shape(upoints)[-1], axis=-1),
                                       non_blank)
        non_blank_etime = tf.gather_nd(tf.repeat(tf.expand_dims(etime, axis=-1), tf.shape(upoints)[-1], axis=-1),
                                       non_blank)

        return non_blank_transcript, non_blank_stime, non_blank_etime, hypothesis.index, hypothesis.states

    def _perform_greedy_batch(
            self,
            encoded: tf.Tensor,
            encoded_length: tf.Tensor,
            parallel_iterations: int = 10,
            swap_memory: bool = False,
    ):
        with tf.name_scope(f"{self.name}_perform_greedy_batch"):
            total_batch = tf.shape(encoded)[0]
            batch = tf.constant(0, dtype=tf.int32)

            decoded = tf.TensorArray(
                dtype=tf.int32,
                size=total_batch,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([None]),
            )

            def condition(batch, _):
                return tf.less(batch, total_batch)

            def body(batch, decoded):
                hypothesis = self._perform_greedy(
                    encoded=encoded[batch],
                    encoded_length=encoded_length[batch],
                    predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                    states=self.predict_net.get_initial_state(),
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                )
                decoded = decoded.write(batch, hypothesis.prediction)
                return batch + 1, decoded

            batch, decoded = tf.while_loop(
                condition,
                body,
                loop_vars=[batch, decoded],
                parallel_iterations=parallel_iterations,
                swap_memory=True,
            )

            decoded = math_util.pad_prediction_tfarray(decoded, blank=self.text_featurizer.blank)
            return self.text_featurizer.iextract(decoded.stack())

    def _perform_greedy(
            self,
            encoded: tf.Tensor,
            encoded_length: tf.Tensor,
            predicted: tf.Tensor,
            states: tf.Tensor,
            parallel_iterations: int = 10,
            swap_memory: bool = False,
            tflite: bool = False,
    ):
        with tf.name_scope(f"{self.name}_greedy"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length

            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=total,
                    dynamic_size=False,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([]),
                ),
                states=states,
            )

            def condition(_time, _hypothesis):
                return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    predicted=_hypothesis.index,
                    states=_hypothesis.states,
                    tflite=tflite,
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []
                # something is wrong with tflite that drop support for tf.cond
                # def equal_blank_fn(): return _hypothesis.index, _hypothesis.states
                # def non_equal_blank_fn(): return _predict, _states  # update if the new prediction is a non-blank
                # _index, _states = tf.cond(tf.equal(_predict, blank), equal_blank_fn, non_equal_blank_fn)

                _equal = tf.equal(_predict, self.text_featurizer.blank)
                _index = tf.where(_equal, _hypothesis.index, _predict)
                _states = tf.where(_equal, _hypothesis.states, _states)
                _prediction = _hypothesis.prediction.write(_time, _predict)
                _hypothesis = Hypothesis(index=_index, prediction=_prediction, states=_states)

                return _time + 1, _hypothesis

            time, hypothesis = tf.while_loop(
                condition,
                body,
                loop_vars=[time, hypothesis],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
            )
            return Hypothesis(
                index=hypothesis.index,
                prediction=hypothesis.prediction.stack(),
                states=hypothesis.states,
            )

    def _perform_greedy_v2(
            self,
            encoded: tf.Tensor,
            encoded_length: tf.Tensor,
            predicted: tf.Tensor,
            states: tf.Tensor,
            parallel_iterations: int = 10,
            swap_memory: bool = False,
            tflite: bool = False,
    ):
        """Ref: https://arxiv.org/pdf/1801.00841.pdf"""
        with tf.name_scope(f"{self.name}_greedy_v2"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length

            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=0,
                    dynamic_size=True,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([]),
                ),
                states=states,
            )

            def condition(_time, _hypothesis):
                return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    predicted=_hypothesis.index,
                    states=_hypothesis.states,
                    tflite=tflite,
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                _equal = tf.equal(_predict, self.text_featurizer.blank)
                _index = tf.where(_equal, _hypothesis.index, _predict)
                _states = tf.where(_equal, _hypothesis.states, _states)
                _time = tf.where(_equal, _time + 1, _time)

                _prediction = _hypothesis.prediction.write(_time, _predict)
                _hypothesis = Hypothesis(index=_index, prediction=_prediction, states=_states)

                return _time, _hypothesis

            time, hypothesis = tf.while_loop(
                condition,
                body,
                loop_vars=[time, hypothesis],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
            )

            return Hypothesis(
                index=hypothesis.index,
                prediction=hypothesis.prediction.stack(),
                states=hypothesis.states,
            )

    # -------------------------------- BEAM SEARCH -------------------------------------

    def recognize_beam(
            self,
            inputs: Dict[str, tf.Tensor],
            return_tokens: bool = False,
            return_topk: bool = False,
            return_log_probas: bool = False,
            lm: bool = False,
    ):
        """
        RNN Transducer Beam Search
        Args:
            inputs (Dict[str, tf.Tensor]): Input dictionary containing "inputs" and "inputs_length"
            lm (bool, optional): whether to use language model. Defaults to False.

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded = self.encoder(inputs["inputs"], training=False)
        encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        predictions, probabilities = self.beam.call(encoded=encoded,
                                                    encoded_length=encoded_length,
                                                    return_topk=return_topk,
                                                    parallel_iterations=1)

        with tf.device('/cpu:0'):
            transcriptions = tf.map_fn(self.text_featurizer.iextract, predictions, fn_output_signature=tf.string)
        output = [transcriptions]
        if return_tokens:
            output = output + [predictions]
        if return_log_probas:
            output = output + [probabilities]

        return output

    # -------------------------------- TFLITE -------------------------------------

    def make_tflite_function(
            self,
            timestamp: bool = False,
    ):
        tflite_func = self.recognize_tflite_with_timestamp if timestamp else self.recognize_tflite
        return tf.function(
            tflite_func,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32),
                tf.TensorSpec(self.predict_net.get_initial_state().get_shape(), dtype=tf.float32),
            ],
        )
