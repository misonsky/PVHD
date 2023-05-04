#coding=utf-8
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import _caching_device
from tensorflow.python.keras.layers.recurrent import _config_for_enable_caching_device
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell
from tensorflow.python.keras.layers import StackedRNNCells
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
version = tf.__version__
if version=="2.3.0":
    from tensorflow.python.ops import control_flow_util
else:
    from tensorflow.python.keras.utils import control_flow_util
RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')

class FFN(keras.layers.Layer):
    def __init__(self):
        super(FFN, self).__init__()
        # self.dense1 = keras.layers.Dense(hidden)
        # self.dense2 = keras.layers.Dense(hidden)
    def build(self, input_shape):
        self.dense1 = keras.layers.Dense(input_shape[-1])
        self.dense2 = keras.layers.Dense(input_shape[-1])
    def call(self,x,activation=None):
        x = self.dense1(x)
        if activation is not None:
            x = activation(x)
        x = self.dense2(x)
        if activation is not None:
            x = activation(x)
        return x
class VariableLayer(keras.layers.Layer):
    def __init__(self):
        super(VariableLayer, self).__init__()
        self.noraml_initializer = keras.initializers.random_normal(mean=0., stddev=1.)
    def build(self, input_shape):
        self.c_hidden = input_shape[-1]
        self.prior_h = FFN()
        self.decoder = FFN()
        self.prior_mu = keras.layers.Dense(self.c_hidden)
        self.prior_var = keras.layers.Dense(self.c_hidden)
    def prior(self, context_outputs):
        # context_outputs: [batch, context_hidden]
        h_prior = self.prior_h(context_outputs,activation = tf.nn.tanh)
        mu_prior = self.prior_mu(h_prior)
        var_prior = tf.nn.softplus(self.prior_var(h_prior))
        return mu_prior, var_prior
    def aedecoder(self,input_x):
        decoder_vector = self.decoder(input_x,activation = tf.nn.relu)
        return decoder_vector
    def kl_div(self, mu1, var1, mu2, var2):
        one = tf.constant([1.0])
        kl_div = tf.reduce_sum(0.5 * (tf.math.log(var2)-tf.math.log(var1) + (var1 + tf.pow(mu1 - mu2,2.0)) / var2 -one),axis=1)
        return kl_div
    def KL_distance(self,mu1,var1,mu2,var2):
        kl_div = self.kl_div(mu1, var1, mu2, var2)
        kl_div = tf.reduce_sum(kl_div)
        return kl_div
    def call(self, inputs):
        # pre_c: [batch, c_hidden]
        # curr_c:[batch, c_hidden]
        # Return: z_sent [batch, c_hidden]
        eps = self.noraml_initializer(shape=(inputs.shape[0], self.c_hidden))
        mu_prior, var_prior = self.prior(inputs)
        z_sent = mu_prior + tf.math.sqrt(var_prior) * eps
        return mu_prior,var_prior,z_sent
@keras_export(v1=['keras.layers.VLSTMCell'])
class VLSTMCell(DropoutRNNCellMixin, Layer):
    def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
        if ops.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(VLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
    
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias
    
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
    
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
    
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        implementation = kwargs.pop('implementation', 1)
        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
            self.implementation = 1
        else:
            self.implementation = implementation
        # tuple(_ListWrapper) was silently dropping list content in at least 2.7.10,
        # and fixed after 2.7.16. Converting the state_size to wrapper around
        # NoDependency(), so that the base_layer.__setattr__ will not convert it to
        # ListWrapper. Down the stream, self.states will be a list since it is
        # generated from nest.map_structure with list, and tuple(list) will work
        # properly.
        self.state_size = data_structures.NoDependency([self.units, self.units,self.units])
        self.output_size = self.units
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)
        
        self.recurrent_latent = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_latent',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.get('ones')((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device)
        else:
            self.bias = None
        self.built = True
    def _compute_carry_and_output(self, x, h_tm1, c_tm1,z_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o= x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        z_tm1_i, z_tm1_f, z_tm1_c, z_tm1_o = z_tm1
        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) + K.dot(z_tm1_i,self.recurrent_latent[:, :self.units]))
        f = self.recurrent_activation(x_f + K.dot(
            h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]) + K.dot(z_tm1_f,self.recurrent_latent[:, self.units:self.units * 2]))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]) + K.dot(z_tm1_c,self.recurrent_latent[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]) + K.dot(z_tm1_o,self.recurrent_latent[:, self.units * 3:]))
        return c, o
    def _compute_carry_and_output_fused(self, x, c_tm1):
        """Computes carry and output using fused kernels."""
        x0, x1, x2, x3 = x
        i = self.recurrent_activation(x0)
        f = self.recurrent_activation(x1)
        c = f * c_tm1 + i * self.activation(x2)
        o = self.recurrent_activation(x3)
        return c, o
    def call(self, inputs, states,training=None):
        assert len(states) == 3
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        z_tm1 = states[2]  # previous latent state
        previous_h = h_tm1
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=4)
        rec_v_mask = self.get_recurrent_dropout_mask_for_cell(z_tm1, training, count=4)
        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            k_i, k_f, k_c, k_o = array_ops.split(self.kernel, num_or_size_splits=4, axis=1)
            x_i = K.dot(inputs_i, k_i)
            x_f = K.dot(inputs_f, k_f)
            x_c = K.dot(inputs_c, k_c)
            x_o = K.dot(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = array_ops.split(
                    self.bias, num_or_size_splits=4, axis=0)
                x_i = K.bias_add(x_i, b_i)
                x_f = K.bias_add(x_f, b_f)
                x_c = K.bias_add(x_c, b_c)
                x_o = K.bias_add(x_o, b_o)
            if 0 < self.recurrent_dropout < 1.:
                z_tm1_i = z_tm1 * rec_v_mask[0]
                z_tm1_f = z_tm1 * rec_v_mask[1]
                z_tm1_c = z_tm1 * rec_v_mask[2]
                z_tm1_o = z_tm1 * rec_v_mask[3]
            else:
                z_tm1_i = z_tm1
                z_tm1_f = z_tm1
                z_tm1_c = z_tm1
                z_tm1_o = z_tm1
            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            z_tm1 = (z_tm1_i, z_tm1_f, z_tm1_c, z_tm1_o)
            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1,z_tm1)
        else:
            if 0. < self.dropout < 1.:
                inputs = inputs * dp_mask[0]
            x = K.dot(inputs, self.kernel)
            x += K.dot(h_tm1, self.recurrent_kernel)
            x += K.dot(z_tm1,self.recurrent_latent)
            if self.use_bias:
                x = K.bias_add(x, self.bias)
            x = array_ops.split(x, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(x, c_tm1)
        h = o * self.activation(c)
        z = self.var(previous_h,h)
        return h, [h,c,z]
    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'unit_forget_bias':
                self.unit_forget_bias,
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout,
            'implementation':
                self.implementation
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(VLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)) 

@keras_export('keras.layers.VALSTMCell', v1=[])
class VALSTMCell(VLSTMCell):
    def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
        super(VALSTMCell, self).__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=kwargs.pop('implementation', 2),
            **kwargs)

@keras_export(v1=['keras.layers.VGRUCell'])
class VGRUCell(DropoutRNNCellMixin, Layer):
    def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               reset_after=False,
               **kwargs):
        # By default use cached variable under v2 mode, see b/143699808.
        if ops.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(VGRUCell, self).__init__(**kwargs)
        self.units = units
        self.var = VariableLayer()
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        
        implementation = kwargs.pop('implementation', 1)
        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
            self.implementation = 1
        else:
            self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = data_structures.NoDependency([self.units, self.units])
        self.output_size = self.units
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        default_caching_device = _caching_device(self)
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)
        self.recurrent_latent = self.add_weight(
            shape=(self.units, self.units*4),#*3
            name='recurrent_latent',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)
        if self.use_bias:
            if not self.reset_after:
                bias_shape = (4 * self.units,)#3
            else:
                bias_shape = (2, 4 * self.units)#3
            self.bias = self.add_weight(shape=bias_shape,
                                  name='bias',
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint,
                                  caching_device=default_caching_device)
        else:
            self.bias = None
        self.built = True
    def call(self, inputs, states, training=None):
        assert nest.is_nested(states)
        h_tm1 = states[0] # previous memory
        v_tm1 = states[-1]
#         h_tm1 = states[0] if nest.is_nested(states) else states  # previous memory
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=3)
        rec_var_mask = self.get_recurrent_dropout_mask_for_cell(v_tm1, training, count=3)
        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = array_ops.unstack(self.bias)
        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs
            x_z = K.dot(inputs_z, self.kernel[:, :self.units])
            x_r = K.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
            x_h = K.dot(inputs_h, self.kernel[:, self.units * 2:])
            if self.use_bias:
                x_z = K.bias_add(x_z, input_bias[:self.units])
                x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
                x_h = K.bias_add(x_h, input_bias[self.units * 2:])
            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1
            recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
            recurrent_r = K.dot(h_tm1_r,self.recurrent_kernel[:, self.units:self.units * 2])
            if 0. < self.recurrent_dropout < 1.:
#                 z_tm1_z = z_tm1 * rec_var_mask[0]
#                 z_tm1_r = z_tm1 * rec_var_mask[1]
                z_tm1_h = v_tm1 * rec_var_mask[2]
            else:
#                 z_tm1_z = z_tm1
#                 z_tm1_r = z_tm1
                z_tm1_h = v_tm1
#             recurrent_z += K.dot(z_tm1_z, self.recurrent_latent[:, :self.units])
#             recurrent_r += K.dot(z_tm1_r, self.recurrent_latent[:, self.units:self.units * 2])
            if self.reset_after and self.use_bias:
                recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
                recurrent_r = K.bias_add(recurrent_r,recurrent_bias[self.units:self.units * 2])
            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)
            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2:]) + K.dot(z_tm1_h, self.recurrent_latent)#[:, self.units * 2:]
                if self.use_bias:
                    recurrent_h = K.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(r * h_tm1_h,self.recurrent_kernel[:, self.units * 2:]) + K.dot(z_tm1_h, self.recurrent_latent)#[:, self.units * 2:]
            hh = self.activation(x_h + recurrent_h)
        else:
            if 0. < self.dropout < 1.:
                inputs = inputs * dp_mask[0]
            matrix_x = K.dot(inputs, self.kernel) # x * keral
            if self.use_bias:
                matrix_x = K.bias_add(matrix_x, input_bias) # + bias
            x_z, x_r, x_h,x_v = array_ops.split(matrix_x, 4, axis=-1)#3
            if self.reset_after:
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel) + K.dot(v_tm1, self.recurrent_latent)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
            else:
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel[:, :3 * self.units]) + K.dot(v_tm1, self.recurrent_latent[:, :3 * self.units])
            recurrent_z, recurrent_r,recurrent_v,recurrent_h= array_ops.split(matrix_inner, [self.units, self.units,self.units,-1], axis=-1)
            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)
            v = self.recurrent_activation(x_v + recurrent_v)
            if self.reset_after:
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(r * h_tm1,self.recurrent_kernel[:, 3 * self.units:]) + K.dot(v_tm1, self.recurrent_latent[:, self.units * 3:])
            hh = self.activation(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        ###ver_1
        # input_m,input_v,_=self.var(x_v)
        # p_m,p_v,_=self.var(h-h_tm1)
        # step_kl = self.var.KL_distance(input_m, input_v, p_m, p_v)
        # _,_,var_v=self.var(h)
        # var_v = v * v_tm1 + (1-v) * var_v
        ###ver_2
        # input_m,input_v,_=self.var(x_v)
        # p_m,p_v,var_v=self.var(h-h_tm1)
        # step_kl = self.var.KL_distance(input_m, input_v, p_m, p_v)
        # var_v = v * v_tm1 + (1-v) * var_v
        ### ver_3
        input_m,input_v,_=self.var(x_v)
        p_m,p_v,var_v=self.var(h-h_tm1)
        step_kl = self.var.KL_distance(input_m, input_v, p_m, p_v)
        var_v = v * v_tm1 + (1-v) * var_v
        ae_v = self.var.aedecoder(var_v)
        step_kl += keras.losses.huber(h,ae_v)
        return h, [h,var_v],step_kl
    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'implementation': self.implementation,
            'reset_after': self.reset_after
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(VGRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)
@keras_export('keras.layers.VGRUCell', v1=[])
class VAGRUCell(VGRUCell):
    def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               reset_after=True,
               **kwargs):
        super(VAGRUCell, self).__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=kwargs.pop('implementation', 2),
            reset_after=reset_after,
            **kwargs)
        

@keras_export('keras.layers.Bidirectional')
class MyBidirectional(Bidirectional):
    def __init__(self,
               layer,
               merge_mode='concat',
               weights=None,
               backward_layer=None,
               **kwargs):
        super(MyBidirectional, self).__init__(layer,
                                              merge_mode,
                                              weights,
                                              backward_layer,
                                              **kwargs)
    def call(self,
           inputs,
           training=None,
           mask=None,
           initial_state=None,
           constants=None):
        kwargs = {}
        if generic_utils.has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        if generic_utils.has_arg(self.layer.call, 'mask'):
            kwargs['mask'] = mask
        if generic_utils.has_arg(self.layer.call, 'constants'):
            kwargs['constants'] = constants
        if generic_utils.has_arg(self.layer.call, 'initial_state'):
            if isinstance(inputs, list) and len(inputs) > 1:
                forward_inputs = [inputs[0]]
                backward_inputs = [inputs[0]]
                pivot = (len(inputs) - self._num_constants) // 2 + 1
                # add forward initial state
                forward_inputs += inputs[1:pivot]
                if not self._num_constants:
                    # add backward initial state
                    backward_inputs += inputs[pivot:]
                else:
                    # add backward initial state
                    backward_inputs += inputs[pivot:-self._num_constants]
                    # add constants for forward and backward layers
                    forward_inputs += inputs[-self._num_constants:]
                    backward_inputs += inputs[-self._num_constants:]
                forward_state, backward_state = None, None
                if 'constants' in kwargs:
                    kwargs['constants'] = None
            elif initial_state is not None:
                forward_inputs, backward_inputs = inputs, inputs
                half = len(initial_state) // 2
                forward_state = initial_state[:half]
                backward_state = initial_state[half:]
            else:
                forward_inputs, backward_inputs = inputs, inputs
                forward_state, backward_state = None, None
            y = self.forward_layer(forward_inputs,initial_state=forward_state, **kwargs)
            y_rev = self.backward_layer(backward_inputs,initial_state=backward_state, **kwargs)
        else:
            y = self.forward_layer(inputs, **kwargs)
            y_rev = self.backward_layer(inputs, **kwargs)
        forward_kl = y[-1]
        backward_kl = y_rev[-1]
        y = y[0]
        y_rev = y_rev[0]
        if self.return_state:
            states = y[1:] + y_rev[1:]
            y = y[0]
            y_rev = y_rev[0]
        if self.return_sequences:
            time_dim = 0 if getattr(self.forward_layer, 'time_major', False) else 1
            y_rev = reverse(y_rev, time_dim)
        if self.merge_mode == 'concat':
            output = K.concatenate([y, y_rev])
        elif self.merge_mode == 'sum':
            output = y + y_rev
        elif self.merge_mode == 'ave':
            output = (y + y_rev) / 2
        elif self.merge_mode == 'mul':
            output = y * y_rev
        elif self.merge_mode is None:
            output = [y, y_rev]
        else:
            raise ValueError('Unrecognized value for `merge_mode`: %s' % (self.merge_mode))
        if self.return_state:
            if self.merge_mode is None:
                return output + states
            return [output] + states,K.concatenate([forward_kl, backward_kl],axis=-1)
        return output,K.concatenate([forward_kl, backward_kl],axis=-1)
            
@keras_export('keras.layers.MyStackedRNNCells')
class MyStackedRNNCells(StackedRNNCells):
    def __init__(self, cells, **kwargs):
        super(MyStackedRNNCells, self).__init__(cells,**kwargs)
    def call(self, inputs, states, constants=None, training=None, **kwargs):
        state_size = (self.state_size[::-1] if self.reverse_state_order else self.state_size)
        nested_states = nest.pack_sequence_as(state_size, nest.flatten(states))
        new_nested_states = []
        kl_score = []
        for cell, states in zip(self.cells, nested_states):
            states = states if nest.is_nested(states) else [states]
            # TF cell does not wrap the state into list when there is only one state.
            is_tf_rnn_cell = getattr(cell, '_is_tf_rnn_cell', None) is not None
            states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
            if generic_utils.has_arg(cell.call, 'training'):
                kwargs['training'] = training
            else:
                kwargs.pop('training', None)
            cell_call_fn = cell.__call__ if callable(cell) else cell.call
            if generic_utils.has_arg(cell.call, 'constants'):
                inputs, states,kl = cell_call_fn(inputs, states,constants=constants, **kwargs)
            else:
                inputs, states,kl = cell_call_fn(inputs, states,**kwargs)
            kl_score.append(kl)
            new_nested_states.append(states)
        return inputs, nest.pack_sequence_as(state_size,nest.flatten(new_nested_states)),tf.math.reduce_mean(tf.stack(kl_score,axis=0))

@keras_export('keras.layers.MyRNN')
class MyRNN(keras.layers.RNN):
    def __init__(self,
               cell,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               time_major=False,
               **kwargs):
        super(MyRNN, self).__init__(cell,**kwargs)
        if isinstance(cell, (list, tuple)):
            cell = MyStackedRNNCells(cell)
        if not 'call' in dir(cell):
            raise ValueError('`cell` should have a `call` method. '
                       'The RNN was passed:', cell)
        if not 'state_size' in dir(cell):
            raise ValueError('The RNN cell should have '
                       'an attribute `state_size` '
                       '(tuple of integers, '
                       'one integer per RNN state).')
        self.zero_output_for_mask = kwargs.pop('zero_output_for_mask', False)
        if 'input_shape' not in kwargs and ('input_dim' in kwargs or 'input_length' in kwargs):
            input_shape = (kwargs.pop('input_length', None),kwargs.pop('input_dim', None))
            kwargs['input_shape'] = input_shape
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.time_major = time_major
        self.supports_masking = True
        self.input_spec = None
        self.state_spec = None
        self._states = None
        self.constants_spec = None
        self._num_constants = 0
        if stateful:
            if ds_context.has_strategy():
                raise ValueError('RNNs with stateful=True not yet supported with tf.distribute.Strategy.')



    def call(self, 
        inputs, 
        mask=None, 
        training=None, 
        initial_state=None, 
        constants=None):
        inputs, row_lengths = K.convert_inputs_if_ragged(inputs)
        is_ragged_input = (row_lengths is not None)
        self._validate_args_if_ragged(is_ragged_input, mask)
        inputs, initial_state, constants = self._process_inputs(inputs, initial_state, constants)
        self._maybe_reset_cell_dropout_mask(self.cell)
        if isinstance(self.cell, MyStackedRNNCells):
            for cell in self.cell.cells:
                self._maybe_reset_cell_dropout_mask(cell)
        if mask is not None:
            mask = nest.flatten(mask)[0]
        if nest.is_nested(inputs):
            # In the case of nested input, use the first element for shape check.
            input_shape = K.int_shape(nest.flatten(inputs)[0])
        else:
            input_shape = K.int_shape(inputs)
        timesteps = input_shape[0] if self.time_major else input_shape[1]
        if self.unroll and timesteps is None:
            raise ValueError('Cannot unroll a RNN if the '
                       'time dimension is undefined. \n'
                       '- If using a Sequential model, '
                       'specify the time dimension by passing '
                       'an `input_shape` or `batch_input_shape` '
                       'argument to your first layer. If your '
                       'first layer is an Embedding, you can '
                       'also use the `input_length` argument.\n'
                       '- If using the functional API, specify '
                       'the time dimension by passing a `shape` '
                       'or `batch_shape` argument to your Input layer.')
        kwargs = {}
        if generic_utils.has_arg(self.cell.call, 'training'):
            kwargs['training'] = training
        is_tf_rnn_cell = getattr(self.cell, '_is_tf_rnn_cell', None) is not None
        cell_call_fn = self.cell.__call__ if callable(self.cell) else self.cell.call
        if constants:
            if not generic_utils.has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')
            def step(inputs, states):
                constants = states[-self._num_constants:]  # pylint: disable=invalid-unary-operand-type
                states = states[:-self._num_constants]  # pylint: disable=invalid-unary-operand-type

                states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
                output, new_states,KL = cell_call_fn(
                    inputs, states, constants=constants, **kwargs)
                if not nest.is_nested(new_states):
                    new_states = [new_states]
                return output, new_states,KL
        else:
            def step(inputs, states):
                states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
                output, new_states,KL= cell_call_fn(inputs, states, **kwargs)
                if not nest.is_nested(new_states):
                    new_states = [new_states]
                return output, new_states,KL
        last_output, outputs, states ,kl_loss= rnn(
            step,
            inputs,
            initial_state,
            constants=constants,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=row_lengths if row_lengths is not None else timesteps,
            time_major=self.time_major,
            zero_output_for_mask=self.zero_output_for_mask)
        if self.stateful:
            updates = [
              state_ops.assign(self_state, state) for self_state, state in zip(
                  nest.flatten(self.states), nest.flatten(states))]
            self.add_update(updates)
        if self.return_sequences:
            output = K.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)
        else:
            output = last_output
        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return generic_utils.to_list(output) + states,kl_loss
        else:
            return output,kl_loss

@keras_export('keras.backend.reverse')
@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def reverse(x, axes):
  """Reverse a tensor along the specified axes.

  Arguments:
      x: Tensor to reverse.
      axes: Integer or iterable of integers.
          Axes to reverse.

  Returns:
      A tensor.
  """
  if isinstance(axes, int):
    axes = [axes]
  return array_ops.reverse(x, axes)


@keras_export('keras.backend.zeros_like')
@doc_controls.do_not_generate_docs
def zeros_like(x, dtype=None, name=None):
  """Instantiates an all-zeros variable of the same shape as another tensor.

  Arguments:
      x: Keras variable or Keras tensor.
      dtype: dtype of returned Keras variable.
             `None` uses the dtype of `x`.
      name: name for the variable to create.

  Returns:
      A Keras variable with the shape of `x` filled with zeros.

  Example:


  from tensorflow.keras import backend as K
  kvar = K.variable(np.random.random((2,3)))
  kvar_zeros = K.zeros_like(kvar)
  K.eval(kvar_zeros)
  # array([[ 0.,  0.,  0.], [ 0.,  0.,  0.]], dtype=float32)


  """
  return array_ops.zeros_like(x, dtype=dtype, name=name)

@keras_export('keras.backend.expand_dims')
@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def expand_dims(x, axis=-1):
  """Adds a 1-sized dimension at index "axis".

  Arguments:
      x: A tensor or variable.
      axis: Position where to add a new axis.

  Returns:
      A tensor with expanded dimensions.
  """
  return array_ops.expand_dims(x, axis)

@keras_export('keras.backend.rnn')
@dispatch.add_dispatch_support
def rnn(step_function,
        inputs,
        initial_states,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False,
        input_length=None,
        time_major=False,
        zero_output_for_mask=False):
  """Iterates over the time dimension of a tensor.

  Arguments:
      step_function: RNN step function.
          Args;
              input; Tensor with shape `(samples, ...)` (no time dimension),
                  representing input for the batch of samples at a certain
                  time step.
              states; List of tensors.
          Returns;
              output; Tensor with shape `(samples, output_dim)`
                  (no time dimension).
              new_states; List of tensors, same length and shapes
                  as 'states'. The first state in the list must be the
                  output tensor at the previous timestep.
      inputs: Tensor of temporal data of shape `(samples, time, ...)`
          (at least 3D), or nested tensors, and each of which has shape
          `(samples, time, ...)`.
      initial_states: Tensor with shape `(samples, state_size)`
          (no time dimension), containing the initial values for the states used
          in the step function. In the case that state_size is in a nested
          shape, the shape of initial_states will also follow the nested
          structure.
      go_backwards: Boolean. If True, do the iteration over the time
          dimension in reverse order and return the reversed sequence.
      mask: Binary tensor with shape `(samples, time, 1)`,
          with a zero for every element that is masked.
      constants: List of constant values passed at each step.
      unroll: Whether to unroll the RNN or to use a symbolic `while_loop`.
      input_length: An integer or a 1-D Tensor, depending on whether
          the time dimension is fixed-length or not. In case of variable length
          input, it is used for masking in case there's no mask specified.
      time_major: Boolean. If true, the inputs and outputs will be in shape
          `(timesteps, batch, ...)`, whereas in the False case, it will be
          `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
          efficient because it avoids transposes at the beginning and end of the
          RNN calculation. However, most TensorFlow data is batch-major, so by
          default this function accepts input and emits output in batch-major
          form.
      zero_output_for_mask: Boolean. If True, the output for masked timestep
          will be zeros, whereas in the False case, output from previous
          timestep is returned.

  Returns:
      A tuple, `(last_output, outputs, new_states)`.
          last_output: the latest output of the rnn, of shape `(samples, ...)`
          outputs: tensor with shape `(samples, time, ...)` where each
              entry `outputs[s, t]` is the output of the step function
              at time `t` for sample `s`.
          new_states: list of tensors, latest states returned by
              the step function, of shape `(samples, ...)`.

  Raises:
      ValueError: if input dimension is less than 3.
      ValueError: if `unroll` is `True` but input timestep is not a fixed
      number.
      ValueError: if `mask` is provided (not `None`) but states is not provided
          (`len(states)` == 0).
  """
  def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return array_ops.transpose(input_t, axes)

  if not time_major:
    inputs = nest.map_structure(swap_batch_timestep, inputs)

  flatted_inputs = nest.flatten(inputs)
  time_steps = flatted_inputs[0].shape[0]
  batch = flatted_inputs[0].shape[1]
  time_steps_t = array_ops.shape(flatted_inputs[0])[0]
  for input_ in flatted_inputs:
    input_.shape.with_rank_at_least(3)

  if mask is not None:
    if mask.dtype != dtypes_module.bool:
      mask = math_ops.cast(mask, dtypes_module.bool)
    if len(mask.shape) == 2:
      mask = expand_dims(mask)
    if not time_major:
      mask = swap_batch_timestep(mask)

  if constants is None:
    constants = []

  # tf.where needs its condition tensor to be the same shape as its two
  # result tensors, but in our case the condition (mask) tensor is
  # (nsamples, 1), and inputs are (nsamples, ndimensions) or even more.
  # So we need to broadcast the mask to match the shape of inputs.
  # That's what the tile call does, it just repeats the mask along its
  # second dimension n times.
  def _expand_mask(mask_t, input_t, fixed_dim=1):
    if nest.is_nested(mask_t):
      raise ValueError('mask_t is expected to be tensor, but got %s' % mask_t)
    if nest.is_nested(input_t):
      raise ValueError('input_t is expected to be tensor, but got %s' % input_t)
    rank_diff = len(input_t.shape) - len(mask_t.shape)
    for _ in range(rank_diff):
      mask_t = array_ops.expand_dims(mask_t, -1)
    multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]
    return array_ops.tile(mask_t, multiples)
  if unroll:
    if not time_steps:
      raise ValueError('Unrolling requires a fixed number of timesteps.')
    states = tuple(initial_states)
    successive_states = []
    successive_outputs = []
    successive_kl = []
    # Process the input tensors. The input tensor need to be split on the
    # time_step dim, and reverse if go_backwards is True. In the case of nested
    # input, the input is flattened and then transformed individually.
    # The result of this will be a tuple of lists, each of the item in tuple is
    # list of the tensor with shape (batch, feature)
    def _process_single_input_t(input_t):
      input_t = array_ops.unstack(input_t)  # unstack for time_step dim
      if go_backwards:
        input_t.reverse()
      return input_t

    if nest.is_nested(inputs):
      processed_input = nest.map_structure(_process_single_input_t, inputs)
    else:
      processed_input = (_process_single_input_t(inputs),)

    def _get_input_tensor(time):
      inp = [t_[time] for t_ in processed_input]
      return nest.pack_sequence_as(inputs, inp)

    if mask is not None:
      mask_list = array_ops.unstack(mask)
      if go_backwards:
        mask_list.reverse()

      for i in range(time_steps):
        inp = _get_input_tensor(i)
        mask_t = mask_list[i]
        output, new_states,step_loss = step_function(inp,
                                           tuple(states) + tuple(constants))
        tiled_mask_t = _expand_mask(mask_t, output)
        if not successive_outputs:
          prev_output = zeros_like(output)
        else:
          prev_output = successive_outputs[-1]

        output = array_ops.where_v2(tiled_mask_t, output, prev_output)

        flat_states = nest.flatten(states)
        flat_new_states = nest.flatten(new_states)
        tiled_mask_t = tuple(_expand_mask(mask_t, s) for s in flat_states)
        flat_final_states = tuple(
            array_ops.where_v2(m, s, ps)
            for m, s, ps in zip(tiled_mask_t, flat_new_states, flat_states))
        states = nest.pack_sequence_as(states, flat_final_states)

        successive_outputs.append(output)
        successive_states.append(states)
        if isinstance(step_loss,list):
            successive_kl.extend(step_loss)
        else:
            successive_kl.append(step_loss)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

      if zero_output_for_mask:
        last_output = array_ops.where_v2(
            _expand_mask(mask_list[-1], last_output), last_output,
            zeros_like(last_output))
        outputs = array_ops.where_v2(
            _expand_mask(mask, outputs, fixed_dim=2), outputs,
            zeros_like(outputs))

    else:  # mask is None
      for i in range(time_steps):
        inp = _get_input_tensor(i)
        output, states,step_loss = step_function(inp, tuple(states) + tuple(constants))
        successive_outputs.append(output)
        successive_states.append(states)
        # if isinstance(step_loss,list):
        #     kl_loss.extend(step_loss)
        # else:
        #     kl_loss.append(step_loss)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

  else:  # Unroll == False
    states = tuple(initial_states)

    # Create input tensor array, if the inputs is nested tensors, then it will
    # be flattened first, and tensor array will be created one per flattened
    # tensor.
    input_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=inp.dtype,
            size=time_steps_t,
            tensor_array_name='input_ta_%s' % i)
        for i, inp in enumerate(flatted_inputs))
    input_ta = tuple(
        ta.unstack(input_) if not go_backwards else ta
        .unstack(reverse(input_, 0))
        for ta, input_ in zip(input_ta, flatted_inputs))
    kl_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=inp.dtype,
            size=time_steps_t,
            tensor_array_name='kl_ta_%s' % i)
        for i, inp in enumerate(flatted_inputs))
    # Get the time(0) input and compute the output for that, the output will be
    # used to determine the dtype of output tensor array. Don't read from
    # input_ta due to TensorArray clear_after_read default to True.    kl_shape = tf.zeros(flatted_inputs[0].shape[0])
    input_time_zero = nest.pack_sequence_as(inputs,
                                            [inp[0] for inp in flatted_inputs])
    # output_time_zero is used to determine the cell output shape and its dtype.
    # the value is discarded.
    output_time_zero, _,_= step_function(
        input_time_zero, tuple(initial_states) + tuple(constants))
    output_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=out.dtype,
            size=time_steps_t,
            element_shape=out.shape,
            tensor_array_name='output_ta_%s' % i)
        for i, out in enumerate(nest.flatten(output_time_zero)))
    time = constant_op.constant(0, dtype='int32', name='time')

    # We only specify the 'maximum_iterations' when building for XLA since that
    # causes slowdowns on GPU in TF.
    if (not context.executing_eagerly() and
        control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph())):
      max_iterations = math_ops.reduce_max(input_length)
    else:
      max_iterations = None
    while_loop_kwargs = {
        'cond': lambda time, *_: time < time_steps_t,
        'maximum_iterations': max_iterations,
        'parallel_iterations': 32,
        'swap_memory': True,
    }
    if mask is not None:
      if go_backwards:
        mask = reverse(mask, 0)

      mask_ta = tensor_array_ops.TensorArray(
          dtype=dtypes_module.bool,
          size=time_steps_t,
          tensor_array_name='mask_ta')
      mask_ta = mask_ta.unstack(mask)

      def masking_fn(time):
        return mask_ta.read(time)

      def compute_masked_output(mask_t, flat_out, flat_mask):
        tiled_mask_t = tuple(
            _expand_mask(mask_t, o, fixed_dim=len(mask_t.shape))
            for o in flat_out)
        return tuple(
            array_ops.where_v2(m, o, fm)
            for m, o, fm in zip(tiled_mask_t, flat_out, flat_mask))
    elif isinstance(input_length, ops.Tensor):
      if go_backwards:
        max_len = math_ops.reduce_max(input_length, axis=0)
        rev_input_length = math_ops.subtract(max_len - 1, input_length)

        def masking_fn(time):
          return math_ops.less(rev_input_length, time)
      else:

        def masking_fn(time):
          return math_ops.greater(input_length, time)

      def compute_masked_output(mask_t, flat_out, flat_mask):
        return tuple(
            array_ops.where(mask_t, o, zo)
            for (o, zo) in zip(flat_out, flat_mask))
    else:
      masking_fn = None

    if masking_fn is not None:
      # Mask for the T output will be base on the output of T - 1. In the case
      # T = 0, a zero filled tensor will be used.
      flat_zero_output = tuple(array_ops.zeros_like(o)
                               for o in nest.flatten(output_time_zero))
      def _step(time,kl_ta_t,output_ta_t, prev_output, *states):
        """RNN step function.

        Arguments:
            time: Current timestep value.
            output_ta_t: TensorArray.
            prev_output: tuple of outputs from time - 1.
            *states: List of states.

        Returns:
            Tuple: `(time + 1, output_ta_t, output) + tuple(new_states)`
        """
        current_input = tuple(ta.read(time) for ta in input_ta)
        # maybe set shape.
        current_input = nest.pack_sequence_as(inputs, current_input)
        mask_t = masking_fn(time)
        output, new_states,step_loss= step_function(current_input,tuple(states) + tuple(constants))
        # mask output
        flat_output = nest.flatten(output)
        flat_mask_output = (flat_zero_output if zero_output_for_mask else nest.flatten(prev_output))
        flat_new_output = compute_masked_output(mask_t, flat_output,flat_mask_output)
        #mask loss
        flat_loss = nest.flatten(step_loss)
        # mask states
        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
            if isinstance(new_state, ops.Tensor):
                new_state.set_shape(state.shape)
        flat_final_state = compute_masked_output(mask_t, flat_new_state,flat_state)
        new_states = nest.pack_sequence_as(new_states, flat_final_state)
        kl_ta_t = tuple(ta.write(time, out) for ta, out in zip(kl_ta_t, flat_loss))
        output_ta_t = tuple(ta.write(time, out) for ta, out in zip(output_ta_t, flat_new_output))
        return (time + 1,kl_ta_t,output_ta_t,tuple(flat_new_output)) + tuple(new_states)
      final_outputs = control_flow_ops.while_loop(
          body=_step,
          loop_vars=(time,kl_ta,output_ta, flat_zero_output) + states,
          **while_loop_kwargs)
      # Skip final_outputs[2] which is the output for final timestep.
      new_states = final_outputs[4:]
    else:
      def _step(time, kl_ta_t,output_ta_t, *states):
        """RNN step function.

        Arguments:
            time: Current timestep value.
            output_ta_t: TensorArray.
            *states: List of states.

        Returns:
            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
        """
        current_input = tuple(ta.read(time) for ta in input_ta)
        current_input = nest.pack_sequence_as(inputs, current_input)
        output, new_states,step_loss= step_function(current_input,tuple(states) + tuple(constants))
        #output
        flat_output = nest.flatten(output)
        #loss
        flat_loss = nest.flatten(step_loss)
        #state
        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
            if isinstance(new_state, ops.Tensor):
                new_state.set_shape(state.shape)
        new_states = nest.pack_sequence_as(initial_states, flat_new_state)
        output_ta_t = tuple(ta.write(time, out) for ta, out in zip(output_ta_t, flat_output))
        kl_ta_t = tuple(ta.write(time, out) for ta, out in zip(kl_ta_t, flat_loss))
        return (time + 1, kl_ta_t,output_ta_t) + tuple(new_states)

      final_outputs = control_flow_ops.while_loop(
          body=_step,
          loop_vars=(time, kl_ta,output_ta) + states,
          **while_loop_kwargs)
      new_states = final_outputs[3:]
    output_kl = final_outputs[1]
    output_ta = final_outputs[2]
    outputs = tuple(o.stack() for o in output_ta)
    kl_outputs = tuple(o.stack() for o in output_kl)
    last_output = tuple(o[-1] for o in outputs)
    outputs = nest.pack_sequence_as(output_time_zero, outputs)
    kl_outputs = nest.pack_sequence_as(time_steps, kl_outputs)
    last_output = nest.pack_sequence_as(output_time_zero, last_output)

  # static shape inference
  def set_shape(output_):
    if isinstance(output_, ops.Tensor):
      shape = output_.shape.as_list()
      shape[0] = time_steps
      shape[1] = batch
      output_.set_shape(shape)
    return output_
  
  outputs = nest.map_structure(set_shape, outputs)
  if not time_major:
    outputs = nest.map_structure(swap_batch_timestep, outputs)
  kl_outputs.set_shape(time_steps)
  return last_output, outputs, new_states,kl_outputs

        
            
        
    
    





























    