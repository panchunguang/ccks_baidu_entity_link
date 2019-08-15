# coding=utf-8
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import math
from keras.layers import concatenate, Flatten
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.objectives import categorical_crossentropy
from keras.objectives import sparse_categorical_crossentropy
from keras.layers import Activation,Wrapper
import numpy as np
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras.callbacks import Callback
import keras

from keras.layers import *
class CRF(Layer):
    """An implementation of linear chain conditional random field (CRF).

    An linear chain CRF is defined to maximize the following likelihood function:

    $$ L(W, U, b; y_1, ..., y_n) := \frac{1}{Z} \sum_{y_1, ..., y_n} \exp(-a_1' y_1 - a_n' y_n
        - \sum_{k=1^n}((f(x_k' W + b) y_k) + y_1' U y_2)), $$

    where:
        $Z$: normalization constant
        $x_k, y_k$:  inputs and outputs

    This implementation has two modes for optimization:
    1. (`join mode`) optimized by maximizing join likelihood, which is optimal in theory of statistics.
       Note that in this case, CRF mast be the output/last layer.
    2. (`marginal mode`) return marginal probabilities on each time step and optimized via composition
       likelihood (product of marginal likelihood), i.e., using `categorical_crossentropy` loss.
       Note that in this case, CRF can be either the last layer or an intermediate layer (though not explored).

    For prediction (test phrase), one can choose either Viterbi best path (class indices) or marginal
    probabilities if probabilities are needed. However, if one chooses *join mode* for training,
    Viterbi output is typically better than marginal output, but the marginal output will still perform
    reasonably close, while if *marginal mode* is used for training, marginal output usually performs
    much better. The default behavior is set according to this observation.

    In addition, this implementation supports masking and accepts either onehot or sparse target.


    # Examples

    ```python
        model = Sequential()
        model.add(Embedding(3001, 300, mask_zero=True)(X)

        # use learn_mode = 'join', test_mode = 'viterbi', sparse_target = True (label indice output)
        crf = CRF(10, sparse_target=True)
        model.add(crf)

        # crf.accuracy is default to Viterbi acc if using join-mode (default).
        # One can add crf.marginal_acc if interested, but may slow down learning
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

        # y must be label indices (with shape 1 at dim 3) here, since `sparse_target=True`
        model.fit(x, y)

        # prediction give onehot representation of Viterbi best path
        y_hat = model.predict(x_test)
    ```


    # Arguments
        units: Positive integer, dimensionality of the output space.
        learn_mode: Either 'join' or 'marginal'.
            The former train the model by maximizing join likelihood while the latter
            maximize the product of marginal likelihood over all time steps.
        test_mode: Either 'viterbi' or 'marginal'.
            The former is recommended and as default when `learn_mode = 'join'` and
            gives one-hot representation of the best path at test (prediction) time,
            while the latter is recommended and chosen as default when `learn_mode = 'marginal'`,
            which produces marginal probabilities for each time step.
        sparse_target: Boolen (default False) indicating if provided labels are one-hot or
            indices (with shape 1 at dim 3).
        use_boundary: Boolen (default True) inidicating if trainable start-end chain energies
            should be added to model.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        chain_initializer: Initializer for the `chain_kernel` weights matrix,
            used for the CRF chain energy.
            (see [initializers](../initializers.md)).
        boundary_initializer: Initializer for the `left_boundary`, 'right_boundary' weights vectors,
            used for the start/left and end/right boundary energy.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        chain_regularizer: Regularizer function applied to
            the `chain_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        boundary_regularizer: Regularizer function applied to
            the 'left_boundary', 'right_boundary' weight vectors
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        chain_constraint: Constraint function applied to
            the `chain_kernel` weights matrix
            (see [constraints](../constraints.md)).
        boundary_constraint: Constraint function applied to
            the `left_boundary`, `right_boundary` weights vectors
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used.
            Unrolling can speed-up a RNN, although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    # Output shape
        3D tensor with shape `(nb_samples, timesteps, units)`.

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.

    """

    def __init__(self, units,
                 learn_mode='join',
                 test_mode=None,
                 sparse_target=False,
                 use_boundary=True,
                 use_bias=True,
                 activation='linear',
                 kernel_initializer='glorot_uniform',
                 chain_initializer='orthogonal',
                 bias_initializer='zeros',
                 boundary_initializer='zeros',
                 kernel_regularizer=None,
                 chain_regularizer=None,
                 boundary_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 chain_constraint=None,
                 boundary_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 unroll=False,
                 **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.learn_mode = learn_mode
        assert self.learn_mode in ['join', 'marginal']
        self.test_mode = test_mode
        if self.test_mode is None:
            self.test_mode = 'viterbi' if self.learn_mode == 'join' else 'marginal'
        else:
            assert self.test_mode in ['viterbi', 'marginal']
        self.sparse_target = sparse_target
        self.use_boundary = use_boundary
        self.use_bias = use_bias

        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.chain_initializer = initializers.get(chain_initializer)
        self.boundary_initializer = initializers.get(boundary_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.chain_regularizer = regularizers.get(chain_regularizer)
        self.boundary_regularizer = regularizers.get(boundary_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.chain_constraint = constraints.get(chain_constraint)
        self.boundary_constraint = constraints.get(boundary_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.unroll = unroll


    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight((self.input_dim, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.chain_kernel = self.add_weight((self.units, self.units),
                                            name='chain_kernel',
                                            initializer=self.chain_initializer,
                                            regularizer=self.chain_regularizer,
                                            constraint=self.chain_constraint)
        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.use_boundary:
            self.left_boundary = self.add_weight((self.units,),
                                                 name='left_boundary',
                                                 initializer=self.boundary_initializer,
                                                 regularizer=self.boundary_regularizer,
                                                 constraint=self.boundary_constraint)
            self.right_boundary = self.add_weight((self.units,),
                                                  name='right_boundary',
                                                  initializer=self.boundary_initializer,
                                                  regularizer=self.boundary_regularizer,
                                                  constraint=self.boundary_constraint)
        self.built = True

    def call(self, X, mask=None):
        if mask is not None:
            assert K.ndim(mask) == 2, 'Input mask to CRF must have dim 2 if not None'

        if self.test_mode == 'viterbi':
            test_output = self.viterbi_decoding(X, mask)
        else:
            test_output = self.get_marginal_prob(X, mask)

        self.uses_learning_phase = True
        if self.learn_mode == 'join':
            train_output = K.zeros_like(K.dot(X, self.kernel))
            out = K.in_train_phase(train_output, test_output)
        else:
            if self.test_mode == 'viterbi':
                train_output = self.get_marginal_prob(X, mask)
                out = K.in_train_phase(train_output, test_output)
            else:
                out = test_output
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.units,)

    def compute_mask(self, input, mask=None):
        if mask is not None and self.learn_mode == 'join':
            return K.any(mask, axis=1)
        return mask

    def get_config(self):
        config = {'units': self.units,
                  'learn_mode': self.learn_mode,
                  'test_mode': self.test_mode,
                  'use_boundary': self.use_boundary,
                  'use_bias': self.use_bias,
                  'sparse_target': self.sparse_target,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'chain_initializer': initializers.serialize(self.chain_initializer),
                  'boundary_initializer': initializers.serialize(self.boundary_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'activation': activations.serialize(self.activation),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'chain_regularizer': regularizers.serialize(self.chain_regularizer),
                  'boundary_regularizer': regularizers.serialize(self.boundary_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'chain_constraint': constraints.serialize(self.chain_constraint),
                  'boundary_constraint': constraints.serialize(self.boundary_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'input_dim': self.input_dim,
                  'unroll': self.unroll}
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def loss_function(self):
        if self.learn_mode == 'join':
            def loss(y_true, y_pred):
                assert self._inbound_nodes, 'CRF has not connected to any layer.'
                assert not self._outbound_nodes, 'When learn_model="join", CRF must be the last layer.'
                if self.sparse_target:
                    y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), self.units)
                X = self._inbound_nodes[0].input_tensors[0]
                mask = self._inbound_nodes[0].input_masks[0]
                nloglik = self.get_negative_log_likelihood(y_true, X, mask)
                return nloglik
            return loss
        else:
            if self.sparse_target:
                return sparse_categorical_crossentropy
            else:
                return categorical_crossentropy

    @property
    def accuracy(self):
        if self.test_mode == 'viterbi':
            return self.viterbi_acc
        else:
            return self.marginal_acc

    @staticmethod
    def _get_accuracy(y_true, y_pred, mask, sparse_target=False):
        y_pred = K.argmax(y_pred, -1)
        if sparse_target:
            y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
        else:
            y_true = K.argmax(y_true, -1)
        judge = K.cast(K.equal(y_pred, y_true), K.floatx())
        if mask is None:
            return K.mean(judge)
        else:
            mask = K.cast(mask, K.floatx())
            return K.sum(judge * mask) / K.sum(mask)

    @property
    def viterbi_acc(self):
        def acc(y_true, y_pred):
            X = self._inbound_nodes[0].input_tensors[0]
            mask = self._inbound_nodes[0].input_masks[0]
            y_pred = self.viterbi_decoding(X, mask)
            return self._get_accuracy(y_true, y_pred, mask, self.sparse_target)
        acc.func_name = 'viterbi_acc'
        return acc

    @property
    def marginal_acc(self):
        def acc(y_true, y_pred):
            X = self._inbound_nodes[0].input_tensors[0]
            mask = self._inbound_nodes[0].input_masks[0]
            y_pred = self.get_marginal_prob(X, mask)
            return self._get_accuracy(y_true, y_pred, mask, self.sparse_target)
        acc.func_name = 'marginal_acc'
        return acc

    @staticmethod
    def softmaxNd(x, axis=-1):
        m = K.max(x, axis=axis, keepdims=True)
        exp_x = K.exp(x - m)
        prob_x = exp_x / K.sum(exp_x, axis=axis, keepdims=True)
        return prob_x

    @staticmethod
    def shift_left(x, offset=1):
        assert offset > 0
        return K.concatenate([x[:, offset:], K.zeros_like(x[:, :offset])], axis=1)

    @staticmethod
    def shift_right(x, offset=1):
        assert offset > 0
        return K.concatenate([K.zeros_like(x[:, :offset]), x[:, :-offset]], axis=1)

    def add_boundary_energy(self, energy, mask, start, end):
        start = K.expand_dims(K.expand_dims(start, 0), 0)
        end = K.expand_dims(K.expand_dims(end, 0), 0)
        if mask is None:
            energy = K.concatenate([energy[:, :1, :] + start, energy[:, 1:, :]], axis=1)
            energy = K.concatenate([energy[:, :-1, :], energy[:, -1:, :] + end], axis=1)
        else:
            mask = K.expand_dims(K.cast(mask, K.floatx()))
            start_mask = K.cast(K.greater(mask, self.shift_right(mask)), K.floatx())
            end_mask = K.cast(K.greater(self.shift_left(mask), mask), K.floatx())
            energy = energy + start_mask * start
            energy = energy + end_mask * end
        return energy

    def get_log_normalization_constant(self, input_energy, mask, **kwargs):
        """Compute logarithm of the normalization constance Z, where
        Z = sum exp(-E) -> logZ = log sum exp(-E) =: -nlogZ
        """
        # should have logZ[:, i] == logZ[:, j] for any i, j
        logZ = self.recursion(input_energy, mask, return_sequences=False, **kwargs)
        return logZ[:, 0]

    def get_energy(self, y_true, input_energy, mask):
        """Energy = a1' y1 + u1' y1 + y1' U y2 + u2' y2 + y2' U y3 + u3' y3 + an' y3
        """
        input_energy = K.sum(input_energy * y_true, 2)  # (B, T)
        chain_energy = K.sum(K.dot(y_true[:, :-1, :], self.chain_kernel) * y_true[:, 1:, :], 2)  # (B, T-1)

        if mask is not None:
            mask = K.cast(mask, K.floatx())
            chain_mask = mask[:, :-1] * mask[:, 1:]  # (B, T-1), mask[:,:-1]*mask[:,1:] makes it work with any padding
            input_energy = input_energy * mask
            chain_energy = chain_energy * chain_mask
        total_energy = K.sum(input_energy, -1) + K.sum(chain_energy, -1)  # (B, )

        return total_energy

    def get_negative_log_likelihood(self, y_true, X, mask):
        """Compute the loss, i.e., negative log likelihood (normalize by number of time steps)
           likelihood = 1/Z * exp(-E) ->  neg_log_like = - log(1/Z * exp(-E)) = logZ + E
        """
        input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
        if self.use_boundary:
            input_energy = self.add_boundary_energy(input_energy, mask, self.left_boundary, self.right_boundary)
        energy = self.get_energy(y_true, input_energy, mask)
        logZ = self.get_log_normalization_constant(input_energy, mask, input_length=K.int_shape(X)[1])
        nloglik = logZ + energy
        if mask is not None:
            nloglik = nloglik / (K.sum(K.cast(mask, K.floatx()), 1) + 1e-6) # modified !!! add '+1e-6'
        else:
            nloglik = nloglik / (K.cast(K.shape(X)[1], K.floatx()) + 1e-6) # modified !!!
        return nloglik

    def step(self, input_energy_t, states, return_logZ=True):
        # not in the following  `prev_target_val` has shape = (B, F)
        # where B = batch_size, F = output feature dim
        # Note: `i` is of float32, due to the behavior of `K.rnn`
        prev_target_val, i, chain_energy = states[:3]
        t = K.cast(i[0, 0], dtype='int32')
        if len(states) > 3:
            if K.backend() == 'theano':
                m = states[3][:, t:(t + 2)]
            else:
                m = K.tf.slice(states[3], [0, t], [-1, 2])
            input_energy_t = input_energy_t * K.expand_dims(m[:, 0])
            chain_energy = chain_energy * K.expand_dims(K.expand_dims(m[:, 0] * m[:, 1]))  # (1, F, F)*(B, 1, 1) -> (B, F, F)
        if return_logZ:
            energy = chain_energy + K.expand_dims(input_energy_t - prev_target_val, 2)  # shapes: (1, B, F) + (B, F, 1) -> (B, F, F)
            new_target_val = K.logsumexp(-energy, 1)  # shapes: (B, F)
            return new_target_val, [new_target_val, i + 1]
        else:
            energy = chain_energy + K.expand_dims(input_energy_t + prev_target_val, 2)
            min_energy = K.min(energy, 1)
            argmin_table = K.cast(K.argmin(energy, 1), K.floatx())  # cast for tf-version `K.rnn`
            return argmin_table, [min_energy, i + 1]

    def recursion(self, input_energy, mask=None, go_backwards=False, return_sequences=True, return_logZ=True, input_length=None):
        """Forward (alpha) or backward (beta) recursion

        If `return_logZ = True`, compute the logZ, the normalization constance:

        \[ Z = \sum_{y1, y2, y3} exp(-E) # energy
          = \sum_{y1, y2, y3} exp(-(u1' y1 + y1' W y2 + u2' y2 + y2' W y3 + u3' y3))
          = sum_{y2, y3} (exp(-(u2' y2 + y2' W y3 + u3' y3)) sum_{y1} exp(-(u1' y1' + y1' W y2))) \]

        Denote:
            \[ S(y2) := sum_{y1} exp(-(u1' y1 + y1' W y2)), \]
            \[ Z = sum_{y2, y3} exp(log S(y2) - (u2' y2 + y2' W y3 + u3' y3)) \]
            \[ logS(y2) = log S(y2) = log_sum_exp(-(u1' y1' + y1' W y2)) \]
        Note that:
              yi's are one-hot vectors
              u1, u3: boundary energies have been merged

        If `return_logZ = False`, compute the Viterbi's best path lookup table.
        """
        chain_energy = self.chain_kernel
        chain_energy = K.expand_dims(chain_energy, 0)  # shape=(1, F, F): F=num of output features. 1st F is for t-1, 2nd F for t
        prev_target_val = K.zeros_like(input_energy[:, 0, :])  # shape=(B, F), dtype=float32

        if go_backwards:
            input_energy = K.reverse(input_energy, 1)
            if mask is not None:
                mask = K.reverse(mask, 1)

        initial_states = [prev_target_val, K.zeros_like(prev_target_val[:, :1])]
        constants = [chain_energy]

        if mask is not None:
            mask2 = K.cast(K.concatenate([mask, K.zeros_like(mask[:, :1])], axis=1), K.floatx())
            constants.append(mask2)

        def _step(input_energy_i, states):
            return self.step(input_energy_i, states, return_logZ)

        target_val_last, target_val_seq, _ = K.rnn(_step, input_energy, initial_states, constants=constants,
                                                   input_length=input_length, unroll=self.unroll)

        if return_sequences:
            if go_backwards:
                target_val_seq = K.reverse(target_val_seq, 1)
            return target_val_seq
        else:
            return target_val_last

    def forward_recursion(self, input_energy, **kwargs):
        return self.recursion(input_energy, **kwargs)

    def backward_recursion(self, input_energy, **kwargs):
        return self.recursion(input_energy, go_backwards=True, **kwargs)

    def get_marginal_prob(self, X, mask=None):
        input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
        if self.use_boundary:
            input_energy = self.add_boundary_energy(input_energy, mask, self.left_boundary, self.right_boundary)
        input_length = K.int_shape(X)[1]
        alpha = self.forward_recursion(input_energy, mask=mask, input_length=input_length)
        beta = self.backward_recursion(input_energy, mask=mask, input_length=input_length)
        if mask is not None:
            input_energy = input_energy * K.expand_dims(K.cast(mask, K.floatx()))
        margin = -(self.shift_right(alpha) + input_energy + self.shift_left(beta))
        return self.softmaxNd(margin)

    def viterbi_decoding(self, X, mask=None):
        input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
        if self.use_boundary:
            input_energy = self.add_boundary_energy(input_energy, mask, self.left_boundary, self.right_boundary)

        argmin_tables = self.recursion(input_energy, mask, return_logZ=False)
        argmin_tables = K.cast(argmin_tables, 'int32')

        # backward to find best path, `initial_best_idx` can be any, as all elements in the last argmin_table are the same
        argmin_tables = K.reverse(argmin_tables, 1)
        initial_best_idx = [K.expand_dims(argmin_tables[:, 0, 0])]  # matrix instead of vector is required by tf `K.rnn`
        if K.backend() == 'theano':
            initial_best_idx = [K.T.unbroadcast(initial_best_idx[0], 1)]

        def gather_each_row(params, indices):
            n = K.shape(indices)[0]
            if K.backend() == 'theano':
                return params[K.T.arange(n), indices]
            else:
                indices = K.transpose(K.stack([K.tf.range(n), indices]))
                return K.tf.gather_nd(params, indices)

        def find_path(argmin_table, best_idx):
            next_best_idx = gather_each_row(argmin_table, best_idx[0][:, 0])
            next_best_idx = K.expand_dims(next_best_idx)
            if K.backend() == 'theano':
                next_best_idx = K.T.unbroadcast(next_best_idx, 1)
            return next_best_idx, [next_best_idx]

        _, best_paths, _ = K.rnn(find_path, argmin_tables, initial_best_idx, input_length=K.int_shape(X)[1], unroll=self.unroll)
        best_paths = K.reverse(best_paths, 1)
        best_paths = K.squeeze(best_paths, 2)

        return K.one_hot(best_paths, self.units)

class CLSOut(Layer):

    def __init__(self, **kwargs):
        super(CLSOut, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)
    def call(self, inputs, mask=None):
        x=tf.squeeze(inputs[:,0:1,:],axis=1)
        return x


class InteractiveAttention(Layer):
    """
    Interactive attention between context and aspect text.
    Supporting Masking.
    Follows the work of Dehong et al. [https://www.ijcai.org/proceedings/2017/0568.pdf]
    "Interactive Attention Networks for Aspect-Level Sentiment Classification"
    """

    def __init__(self, return_attend_weight=False, initializer='orthogonal', regularizer=None,
                 constraint=None, **kwargs):
        self.return_attend_weight = return_attend_weight

        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.supports_masking = True
        super(InteractiveAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, asp_text_shape = input_shape

        self.context_w = self.add_weight(shape=(context_shape[-1], asp_text_shape[-1]), initializer=self.initializer,
                                         regularizer=self.regularizer, constraint=self.constraint,
                                         name='{}_context_w'.format(self.name))
        self.context_b = self.add_weight(shape=(context_shape[1],), initializer='zero', regularizer=self.regularizer,
                                         constraint=self.constraint, name='{}_context_b'.format(self.name))
        self.aspect_w = self.add_weight(shape=(asp_text_shape[-1], context_shape[-1]), initializer=self.initializer,
                                        regularizer=self.regularizer, constraint=self.constraint,
                                        name='{}_aspect_w'.format(self.name))
        self.aspect_b = self.add_weight(shape=(asp_text_shape[1],), initializer='zero', regularizer=self.regularizer,
                                        constraint=self.constraint, name='{}_aspect_b'.format(self.name))

        super(InteractiveAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        if mask is not None:
            context_mask, asp_text_mask = mask
        else:
            context_mask = None
            asp_text_mask = None

        context, asp_text = inputs

        context_avg = K.mean(context, axis=1)
        asp_text_avg = K.mean(asp_text, axis=1)

        # attention over context with aspect_text
        a_c = K.tanh(K.batch_dot(asp_text_avg, K.dot(context, self.context_w), axes=[1, 2]) + self.context_b)
        a_c = K.exp(a_c)
        if context_mask is not None:
            a_c *= K.cast(context_mask, K.floatx())
        a_c /= K.cast(K.sum(a_c, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attend_context = K.sum(context * K.expand_dims(a_c), axis=1)

        # attention over aspect text with context
        a_t = K.tanh(K.batch_dot(context_avg, K.dot(asp_text, self.aspect_w), axes=[1, 2]) + self.aspect_b)
        a_t = K.exp(a_t)
        if context_mask is not None:
            a_t *= K.cast(asp_text_mask, K.floatx())
        a_t = K.cast(K.sum(a_t, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attend_asp_text = K.sum(asp_text * K.expand_dims(a_t), axis=1)

        attend_concat = K.concatenate([attend_context, attend_asp_text], axis=-1)

        if self.return_attend_weight:
            return [attend_concat, a_c, a_t]
        else:
            return attend_concat

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, asp_text_shape = input_shape
        if self.return_attend_weight:
            return [(context_shape[0], context_shape[-1]+asp_text_shape[-1]), (context_shape[0], context_shape[1]),
                    (asp_text_shape[0], asp_text_shape[1])]
        else:
            return context_shape[0], context_shape[-1]+asp_text_shape[-1]


class Attention(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
 e: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None, W_constraint=None,
                 u_constraint=None, b_constraint=None, use_W=True, use_bias=False, return_self_attend=False,
                 return_attend_weight=True, **kwargs):
        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.use_W = use_W
        self.use_bias = use_bias
        self.return_self_attend = return_self_attend  # whether perform self attention and return it
        self.return_attend_weight = return_attend_weight  # whether return attention weight
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        if self.use_W:
            self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],), initializer=self.init,
                                     name='{}_W'.format(self.name), regularizer=self.W_regularizer,
                                     constraint=self.W_constraint)
        if self.use_bias:
            self.b = self.add_weight(shape=(input_shape[1],), initializer='zero', name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer, constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],), initializer=self.init, name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer, constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if self.use_W:
            x = K.tanh(K.dot(x, self.W))

        ait = Attention.dot_product(x, self.u)
        if self.use_bias:
            ait += self.b

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        if self.return_self_attend:
            attend_output = K.sum(x * K.expand_dims(a), axis=1)
            if self.return_attend_weight:
                return [attend_output, a]
            else:
                return attend_output
        else:
            return a

    def compute_output_shape(self, input_shape):
        if self.return_self_attend:
            if self.return_attend_weight:
                return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
            else:
                return input_shape[0], input_shape[-1]
        else:
            return input_shape[0], input_shape[1]

    @staticmethod
    def dot_product(x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
            x (): input
            kernel (): weights
        Returns:
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)

class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率
    """

    def __init__(self, layer, lamb, is_ada=False):
        self.layer = layer
        self.lamb = lamb # 学习率比例
        self.is_ada = is_ada # 是否自适应学习率优化器

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma', 'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb**0.5 # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb) # 更改初始化
                setattr(self.layer, key, weight * lamb) # 按比例替换
        return self.layer(inputs)

def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def _softmax(x, dim):

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.nn.softmax(x, dim)
    elif K.backend() is 'cntk':
        import cntk
        return cntk.softmax(x, dim)
    elif K.backend() == 'theano':
        # Theano cannot softmax along an arbitrary dim.
        # So, we will shuffle `dim` to -1 and un-shuffle after softmax.
        perm = np.arange(K.ndim(x))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_perm = K.permute_dimensions(x, perm)
        output = K.softmax(x_perm)

        # Permute back
        perm[dim], perm[-1] = perm[-1], perm[dim]
        output = K.permute_dimensions(x, output)
        return output
    else:
        raise ValueError("Backend '{}' not supported".format(K.backend()))
class DropConnect(Wrapper):
    """
    Keras Wrapper that implements a DropConnect Layer.
    When training with Dropout, a randomly selected subset of activations are
    set to zero within each layer. DropConnect instead sets a randomly
    selected subset of weights within the network to zero.
    Each unit thus receives input from a random subset of units in the
    previous layer.
    Reference: https://cs.nyu.edu/~wanli/dropc/
    Implementation: https://github.com/andry9454/KerasDropconnect
    """
    def __init__(self, layer, prob, **kwargs):
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        if 0. < self.prob < 1.:
            self.layer.kernel = K.in_train_phase(
                K.dropout(self.layer.kernel, self.prob),
                self.layer.kernel)
            self.layer.bias = K.in_train_phase(
                K.dropout(self.layer.bias, self.prob),
                self.layer.bias)
        return self.layer.call(x)


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)
def get_mask(dim1, dim2, start, end):
    mask = np.zeros(shape=[dim1, dim2])
    one = np.ones(shape=[dim2])
    for i in range(end - start):
        mask[start + i] = one
    return mask
class Chunk_MaxPooling(Layer):

    def __init__(self, k=3, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def build(self, input_shape):
        self.dim1=input_shape[1]
        self.dim2 = input_shape[2]
    def call(self, inputs):
        x=inputs
        pool_outs = []
        start = 0
        for i in range(self.k):
            if not i == self.k - 1:
                end = int(self.dim1 / self.k)
            else:
                end = self.dim1 - start
            mask = get_mask(self.dim1, self.dim2, start, end)
            mask = tf.constant(mask, dtype=tf.float32)
            start += int(self.dim1 / self.k)
            x_mask = x * mask
            pool = K.max(x_mask, axis=1)
            pool_outs.append(pool)
        x = concatenate(pool_outs, axis=-1)
        return x
class MaskedGlobalMeanPool1D(Layer):

    def __init__(self, **kwargs):
        super(MaskedGlobalMeanPool1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs -= K.expand_dims((1.0 - mask) * 1e6, axis=-1)
        return K.mean(inputs, axis=-2)

class MaskedGlobalMaxPool1D(Layer):

    def __init__(self, **kwargs):
        super(MaskedGlobalMaxPool1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs -= K.expand_dims((1.0 - mask) * 1e6, axis=-1)
        return K.max(inputs, axis=-2)
class M_Nadam(Optimizer):
    # lr_dict={'embedding_word':0.1,}
    #model.compile(optimizer=M_Nadam(lr,multipliers=lr_dict))

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999,
                 epsilon=None, schedule_decay=0.004, multipliers=None,**kwargs):
        super(M_Nadam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay

        if multipliers is None:
            multipliers = {}
        self.lr_multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (
            1. - 0.5 * (K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (
            1. - 0.5 * (K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))
        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs
        lr = self.lr
        for p, g, m, v in zip(params, grads, ms, vs):
            matched_layer = [x for x in self.lr_multipliers.keys() if x in p.name]
            if matched_layer:
                new_lr = lr * self.lr_multipliers[matched_layer[0]]
            else:
                new_lr = lr
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime
            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            p_t = p - new_lr * m_t_bar / (K.sqrt(v_t_prime) + self.epsilon)
            new_p = p_t
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon,
                  'schedule_decay': self.schedule_decay}
        base_config = super(M_Nadam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class M_Adam(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, multipliers=None,**kwargs):
        super(M_Adam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

        if multipliers is None:
            multipliers = {}
        self.lr_multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            matched_layer = [x for x in self.lr_multipliers.keys() if x in p.name]
            if matched_layer:
                new_lr = lr_t * self.lr_multipliers[matched_layer[0]]
            else:
                new_lr = lr_t


            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - new_lr * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - new_lr * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(M_Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class M_RMSprop(Optimizer):
    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0.,multipliers=None,
                 **kwargs):
        super(M_RMSprop, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.rho = K.variable(rho, name='rho')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

        if multipliers is None:
            multipliers = {}
        self.lr_multipliers = multipliers
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):

            matched_layer = [x for x in self.lr_multipliers.keys() if x in p.name]
            if matched_layer:
                new_lr = lr * self.lr_multipliers[matched_layer[0]]
            else:
                new_lr = lr

            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            new_p = p - new_lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(M_RMSprop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def polynomial_decay(learning_rate,global_step,decay_steps,end_learning_rate=0.0001,power=1.0,cycle=True):
    if cycle:
        decay_steps = decay_steps * math.ceil(global_step / decay_steps)
    else:
        decay_steps=min(global_step, decay_steps)
    lr=(learning_rate - end_learning_rate) *pow((1 - global_step / decay_steps) ,(power)) +end_learning_rate
    return lr
def exponential_decay(learning_rate,global_step,decay_steps,decay_rate,staircase=False):
    if staircase:
        lr = learning_rate * pow(decay_rate,  math.floor(global_step / decay_steps))
    else:
        lr = learning_rate * pow(decay_rate, (global_step / decay_steps))
    return lr

def natural_exp_decay(learning_rate,global_step,decay_steps,decay_rate,staircase=False):
    if staircase:
        lr = learning_rate * math.exp(-decay_rate * math.floor(global_step / decay_steps))
    else:
        lr=learning_rate * math.exp(-decay_rate * global_step /decay_steps)
    return lr

def inverse_time_decay(learning_rate,global_step,decay_steps,decay_rate,staircase=False):
    if staircase:
        lr= learning_rate / (1 + decay_rate * math.floor(global_step /decay_steps))
    else:
        lr = learning_rate / (1 + decay_rate * global_step /decay_steps)
    return lr

class LearningRateDecay(Callback):

    def __init__(self, name='polynomial_decay', learning_rate=0.001,decay_steps=1000,decay_rate=0.95,min_delta=1e-4,verbose=0,staircase=True,power=0.5,end_learning_rate=5e-6, to_min=4,cycle=True):
        super(LearningRateDecay, self).__init__()
        self.name = name
        self.learning_rate=learning_rate
        self.decay_rate=decay_rate
        self.min_delta = min_delta
        self.decay_steps=decay_steps
        self.verbose = verbose
        self.staircase=staircase
        self.power=power
        self.cycle=cycle
        self.end_learning_rate=end_learning_rate
        self.global_step=0
        self.epoch=1
        self.to_min=to_min
    def on_batch_begin(self, batch, logs=None):
        decay_function = eval('%s' % self.name)
        self.global_step += 1
        function1=['exponential_decay','natural_exp_decay','inverse_time_decay']
        if self.name in function1:
            lr = decay_function(self.learning_rate,self.global_step, self.decay_steps,self.decay_rate)
        else:
            lr=decay_function(self.learning_rate,self.global_step, self.decay_steps,self.end_learning_rate,self.power,self.cycle)

        if lr>self.min_delta:
            K.set_value(self.model.optimizer.lr, lr)
        else:
            K.set_value(self.model.optimizer.lr, self.min_delta)
        if self.epoch>=self.to_min:
            K.set_value(self.model.optimizer.lr, self.min_delta)
        # if self.verbose > 0:
            # print('\nEpoch %05d: LearningRateScheduler setting learning '
            #       'rate to %s.' % (batch + 1, lr))

    def on_epoch_begin(self, epoch, logs=None):

        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

        return epoch
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        self.epoch += 1



class ConcatMentionEntitiy(Layer):
    def __init__(self, merge_mode='concat',**kwargs):
        if merge_mode not in ['sum', 'mul', 'ave', 'concat']:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"sum", "mul", "ave", "concat"}')
        self.merge_mode = merge_mode
        super(ConcatMentionEntitiy, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][-1]+input_shape[1][-1])

    def compute_mask(self, inputs, mask=None):
        if not mask==None:
            return mask[0]
        else:
            return None

    def call(self, inputs,mask=None):
        dis_entity,men_state,dis_entity_mark=inputs
        dis_entity_mark=tf.cast(dis_entity_mark,tf.int32)
        men_state = tf.batch_gather(men_state, dis_entity_mark)
        out=concatenate([men_state, dis_entity])
        if mask is not None:
            mask=K.expand_dims(mask[0], 2)
            out *= K.cast(mask[0], K.floatx())
        return out

class StateMix(Layer):
    def __init__(self, merge_mode='concat',**kwargs):
        if merge_mode not in ['sum', 'mul', 'ave', 'concat']:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"sum", "mul", "ave", "concat"}')
        self.merge_mode = merge_mode
        super(StateMix, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        if self.merge_mode=='concat':
            return (input_shape[0][0], input_shape[0][1],input_shape[2][2]*2)
        else:
            return (input_shape[0][0], input_shape[0][1], input_shape[2][2])

    def compute_mask(self, inputs, mask=None):
        # mask=Lambda(lambda x: K.cast(K.greater(x, -1), 'float32'))(inputs[0])
        # if not mask==None:
        #     return mask[0]
        # else:
        return None
    def call(self, inputs,mask=None):
        begin,end,forward,backward = inputs
        mask = Lambda(lambda x: K.cast(K.greater(x, -1), 'float32'))(begin)
        begin = tf.cast(begin,tf.int32)
        end = tf.cast(end, tf.int32)
        begin_state = tf.batch_gather(backward, begin)
        end_state = tf.batch_gather(forward, end)
        if self.merge_mode == 'concat':
            out = concatenate([begin_state,end_state])
        elif self.merge_mode == 'sum':
            out = begin_state+begin_state
        elif self.merge_mode == 'ave':
            out = (begin_state+begin_state) / 2
        elif self.merge_mode == 'mul':
            out = begin_state*begin_state
        else:
            raise ValueError('Unrecognized value for argument '
                         'merge_mode: %s' % (self.merge_mode))
        # if mask is not None:
        mask=K.expand_dims(mask, 2)
        out *= K.cast(mask, K.floatx())
        return out

class StateMixOne(Layer):
    def __init__(self, merge_mode='concat',**kwargs):
        if merge_mode not in ['sum', 'mul', 'ave', 'concat']:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"sum", "mul", "ave", "concat"}')
        self.merge_mode = merge_mode
        super(StateMixOne, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        if self.merge_mode=='concat':
            return (input_shape[0][0],input_shape[2][2]*2)
        else:
            return (input_shape[0][0],input_shape[2][2])

    def compute_mask(self, inputs, mask=None):

        return None
    def call(self, inputs,mask=None):
        begin,end,forward,backward=inputs
        mask=Lambda(lambda x: K.cast(K.greater(x, -1), 'float32'))(begin)
        begin=tf.cast(begin,tf.int32)
        end = tf.cast(end, tf.int32)
        begin_state = tf.batch_gather(backward, begin)
        end_state = tf.batch_gather(forward, end)
        if self.merge_mode == 'concat':
            out = concatenate([begin_state,end_state])
        elif self.merge_mode == 'sum':
            out = begin_state+begin_state
        elif self.merge_mode == 'ave':
            out = (begin_state+begin_state) / 2
        elif self.merge_mode == 'mul':
            out = begin_state*begin_state
        else:
            out = concatenate([begin_state, end_state])
            raise ValueError('Unrecognized value for argument '
                         'merge_mode: %s' % (self.merge_mode))
        # if mask is not None:

        # mask=K.expand_dims(mask, 2)
        # out *= K.cast(mask, K.floatx())
        out=K.squeeze(out,1)
        return out







    def __init__(self, merge_mode='concat',**kwargs):
        if merge_mode not in ['sum', 'mul', 'ave', 'concat']:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"sum", "mul", "ave", "concat"}')
        self.merge_mode=merge_mode
        super(Mix_word, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        if self.merge_mode=='concat':
            return (input_shape[0][0], input_shape[0][1],input_shape[0][2]+input_shape[1][2])
        else:
            return (input_shape[0][0], input_shape[0][1], input_shape[0][2])

    def compute_mask(self, inputs, mask=None):
        if not mask==None:
            return mask[1]
        else:
            return None
    def build(self, input_shape):
        self.word_size = input_shape[0][2]
        self.char_size=input_shape[1][2]

    def call(self, inputs,mask=None):
        word,char,pos_index=inputs
        pos_index=tf.cast(pos_index,tf.int32)
        word = tf.batch_gather(word, pos_index)
        if self.merge_mode in ['mul','sum','ave']:
            if not self.word_size==self.char_size:
                raise ValueError('word_size != char_size,Merge mode must be concat')
        if self.merge_mode == 'concat':
            out = concatenate([char,word])
        elif self.merge_mode == 'sum':
            out = char+word
        elif self.merge_mode == 'ave':
            out = (char+word) / 2
        elif self.merge_mode == 'mul':
            out = char*word
        else:
            raise ValueError('Unrecognized value for argument '
                         'merge_mode: %s' % (self.merge_mode))
        if mask is not None:
            mask=K.expand_dims(mask[1], 2)
            out *= K.cast(mask[1], K.floatx())
        return out



    def __init__(self, merge_mode='add',**kwargs):
        if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"sum", "mul", "ave", "concat", None}')
        self.merge_mode=merge_mode
        super(Mix_char, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        if self.merge_mode=='concat':
            return (input_shape[0][0], input_shape[0][1],input_shape[0][2]*2)
        else:
            return (input_shape[0][0], input_shape[0][1], input_shape[0][2])
    def build(self, input_shape):
        self.x1_length = input_shape[0][1]
        self.x1_hidden_size = input_shape[0][2]
        self.x2_length=input_shape[1][1]
        self.x2_hidden_size=input_shape[1][2]

    def call(self, inputs,mask=None):
        word,char,begin,end=inputs
        char_b = tf.batch_gather(char, begin)
        char_e = tf.batch_gather(char, end)
        char=concatenate([char_b,char_e],axis=-1)
        # out=concatenate([x_b,x_e],axis=-1)
        out=tf.add(char,word)

        return out


class AdaBound(Optimizer):
    """AdaBound optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        final_lr: float >= 0. Final learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        gamma: float >= 0. Convergence speed of the bound function.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: Weight decay weight.
        amsbound: boolean. Whether to apply the AMSBound variant of this
            algorithm.
    # References
        - [Adaptive Gradient Methods with Dynamic Bound of Learning Rate]
          (https://openreview.net/forum?id=Bkg3g2R9FX)
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, final_lr=0.1, beta_1=0.9, beta_2=0.999, gamma=1e-3,
                 epsilon=None, decay=0., amsbound=False, weight_decay=0.0, **kwargs):
        super(AdaBound, self).__init__(**kwargs)

        if not 0. <= gamma <= 1.:
            raise ValueError("Invalid `gamma` parameter. Must lie in [0, 1] range.")

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

        self.final_lr = final_lr
        self.gamma = gamma

        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsbound = amsbound

        self.weight_decay = float(weight_decay)
        self.base_lr = float(lr)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        # Applies bounds on actual learning rate
        step_size = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                          (1. - K.pow(self.beta_1, t)))

        final_lr = self.final_lr * lr / self.base_lr
        lower_bound = final_lr * (1. - 1. / (self.gamma * t + 1.))
        upper_bound = final_lr * (1. + 1. / (self.gamma * t))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsbound:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # apply weight decay
            if self.weight_decay != 0.:
                g += self.weight_decay * K.stop_gradient(p)

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            if self.amsbound:
                vhat_t = K.maximum(vhat, v_t)
                denom = (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                denom = (K.sqrt(v_t) + self.epsilon)

            # Compute the bounds
            step_size_p = step_size * K.ones_like(denom)
            step_size_p_bound = step_size_p / denom
            bounded_lr_t = m_t * K.minimum(K.maximum(step_size_p_bound,
                                                     lower_bound), upper_bound)

            p_t = p - bounded_lr_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'final_lr': float(self.final_lr),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'gamma': float(self.gamma),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'weight_decay': self.weight_decay,
                  'amsbound': self.amsbound}
        base_config = super(AdaBound, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AdamWarmup(keras.optimizers.Optimizer):
    """Adam optimizer with warmup.

    Default parameters follow those provided in the original paper.

    # Arguments
        decay_steps: Learning rate will decay linearly to zero in decay steps.
        warmup_steps: Learning rate will increase linearly to lr in first warmup steps.
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        kernel_weight_decay: float >= 0. Weight decay.
        bias_weight_decay: float >= 0. Weight decay.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, decay_steps, warmup_steps, min_lr=0.0,
                 lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, kernel_weight_decay=0., bias_weight_decay=0.,
                 amsgrad=False, **kwargs):
        super(AdamWarmup, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.decay_steps = K.variable(decay_steps, name='decay_steps')
            self.warmup_steps = K.variable(warmup_steps, name='warmup_steps')
            self.min_lr = K.variable(min_lr, name='min_lr')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.kernel_weight_decay = K.variable(kernel_weight_decay, name='kernel_weight_decay')
            self.bias_weight_decay = K.variable(bias_weight_decay, name='bias_weight_decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_kernel_weight_decay = kernel_weight_decay
        self.initial_bias_weight_decay = bias_weight_decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        lr = K.switch(
            t <= self.warmup_steps,
            self.lr * (t / self.warmup_steps),
            self.lr * (1.0 - K.minimum(t, self.decay_steps) / self.decay_steps),
        )

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = m_t / (K.sqrt(v_t) + self.epsilon)

            if 'bias' in p.name:
                if self.initial_bias_weight_decay > 0.0:
                    p_t += self.bias_weight_decay * p
            else:
                if self.initial_kernel_weight_decay > 0.0:
                    p_t += self.kernel_weight_decay * p
            p_t = p - lr_t * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'decay_steps': float(K.get_value(self.decay_steps)),
            'warmup_steps': float(K.get_value(self.warmup_steps)),
            'min_lr': float(K.get_value(self.min_lr)),
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'epsilon': self.epsilon,
            'kernel_weight_decay': float(K.get_value(self.kernel_weight_decay)),
            'bias_weight_decay': float(K.get_value(self.bias_weight_decay)),
            'amsgrad': self.amsgrad,
        }
        base_config = super(AdamWarmup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

