#coding=utf-8
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import _caching_device
from tensorflow.python.keras.layers.recurrent import _config_for_enable_caching_device
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell
from tensorflow.python.keras.layers import StackedRNNCells
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.wrappers import Bidirectional
RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')

class FFN(keras.layers.Layer):
    def __init__(self):
        super(FFN, self).__init__()
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
        self.posterior_h = FFN()
        self.posterior_mu = keras.layers.Dense(self.c_hidden)
        self.posterior_var = keras.layers.Dense(self.c_hidden)
    def posterior(self, context_outputs, encoder_hidden):
        # context_outputs: [batch, context_hidden]
        # encoder_hidden: [batch, encoder_hidden]
        h_posterior = tf.concat([context_outputs, encoder_hidden],axis=1)
        h_posterior = self.posterior_h(h_posterior,activation = tf.nn.tanh)
        mu_posterior = self.posterior_mu(h_posterior)
        var_posterior = tf.nn.softplus(self.posterior_var(h_posterior))
        return mu_posterior, var_posterior
    def call(self, pre_c, curr_c):
        # pre_c: [batch, c_hidden]
        # curr_c:[batch, c_hidden]
        # Return: z_sent [batch, c_hidden]
        eps = self.noraml_initializer(shape=(pre_c.shape[0], self.c_hidden))
        mu_posterior, var_posterior = self.posterior(pre_c, curr_c)
        z_sent = mu_posterior + tf.math.sqrt(var_posterior) * eps
        return z_sent
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
        self.var = VariableLayer()
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
            shape=(input_dim, self.units * 3),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)
        self.recurrent_latent = self.add_weight(
            shape=(self.units, self.units*3),#*3
            name='recurrent_latent',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)
        if self.use_bias:
            if not self.reset_after:
                bias_shape = (3 * self.units,)#3
            else:
                bias_shape = (2, 3 * self.units)#3
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
        z_tm1 = states[-1]
#         h_tm1 = states[0] if nest.is_nested(states) else states  # previous memory
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=3)
        rec_var_mask = self.get_recurrent_dropout_mask_for_cell(z_tm1, training, count=3)
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
                z_tm1_h = z_tm1 * rec_var_mask[2]
            else:
#                 z_tm1_z = z_tm1
#                 z_tm1_r = z_tm1
                z_tm1_h = z_tm1
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
            x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)#3
            if self.reset_after:
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel) + K.dot(z_tm1, self.recurrent_latent)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
            else:
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel[:, :2 * self.units]) + K.dot(z_tm1, self.recurrent_latent[:, :2 * self.units])
            recurrent_z, recurrent_r,recurrent_h = array_ops.split(matrix_inner, [self.units, self.units,-1], axis=-1)
            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)
            if self.reset_after:
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(r * h_tm1,self.recurrent_kernel[:, 2 * self.units:]) + K.dot(z_tm1_h, self.recurrent_latent[:, self.units * 2:])
            hh = self.activation(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        z_var = self.var(h_tm1,h)
        new_state = [h,z_var] if nest.is_nested(states) else [h,z_var]
        return h, new_state 
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
        if self.return_state:
            states = y[1:] + y_rev[1:]
            y = y[0]
            y_rev = y_rev[0]
        if self.return_sequences:
            time_dim = 0 if getattr(self.forward_layer, 'time_major', False) else 1
            y_rev = K.reverse(y_rev, time_dim)
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
            return [output] + states   
        return output    
            
@keras_export('keras.layers.MyStackedRNNCells')
class MyStackedRNNCells(StackedRNNCells):
    def __init__(self, cells, **kwargs):
        super(MyStackedRNNCells, self).__init__(cells,**kwargs)
    def call(self, inputs, states, constants=None, training=None, **kwargs):
        state_size = (self.state_size[::-1] if self.reverse_state_order else self.state_size)
        nested_states = nest.pack_sequence_as(state_size, nest.flatten(states))
        new_nested_states = []
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
                inputs, states = cell_call_fn(inputs, states,constants=constants, **kwargs)
            else:
                inputs, states = cell_call_fn(inputs, states,**kwargs)
            new_nested_states.append(states)
        return inputs, nest.pack_sequence_as(state_size,nest.flatten(new_nested_states)) 

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
                output, new_states = cell_call_fn(
                    inputs, states, constants=constants, **kwargs)
                if not nest.is_nested(new_states):
                    new_states = [new_states]
                return output, new_states
        else:
            def step(inputs, states):
                states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
                output, new_states= cell_call_fn(inputs, states, **kwargs)
                if not nest.is_nested(new_states):
                    new_states = [new_states]
                return output, new_states
        last_output, outputs, states = K.rnn(
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
            return generic_utils.to_list(output) + states
        else:
            return output

        
            
        
    
    





























    