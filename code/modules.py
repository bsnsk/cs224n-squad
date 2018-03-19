# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class BiDAFAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)
          keys_mask: Tensor shape (batch_size, num_keys)
            similar to values_mask

        Outputs:
          key2value: key to value attention, used for self-attention
            Shape (batch_size, num_keys, value_vec_size * 3)
          output: Tensor shape (batch_size, num_keys, hidden_size*6).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BiDAFAttn"):

            N, M = keys.shape[1], values.shape[1]
            WSim1 = tf.get_variable("W_sim_1", shape=[values.shape[2]], initializer=tf.contrib.layers.xavier_initializer())
            WSim2 = tf.get_variable("W_sim_2", shape=[values.shape[2]], initializer=tf.contrib.layers.xavier_initializer())
            WSim3 = tf.get_variable("W_sim_3", shape=[values.shape[2]], initializer=tf.contrib.layers.xavier_initializer())

            keys_part = tf.matmul(keys, tf.tile(tf.reshape(WSim1, [1, WSim1.shape[0], 1]), [tf.shape(keys)[0], 1, 1]))  # batch_size, num_keys, 1

            values_part = tf.matmul(values, tf.tile(tf.reshape(WSim2, [1, WSim2.shape[0], 1]), [tf.shape(keys)[0], 1, 1]))  # batch_size, num_values, 1

            keys_t = tf.expand_dims(keys, 2)  # batch_size, num_keys, 1, value_vec_size
            values_t = tf.expand_dims(values, 1)  # batch_size, 1, num_values, value_vec_size
            prod_part = tf.tensordot(
                keys_t * values_t,
                WSim3,
                [[3], [0]],
            )  # batch_size, num_keys, num_values

            sim = keys_part + tf.transpose(values_part, perm=[0, 2, 1]) + prod_part  # batch_size, num_keys, num_values
            assert(len(sim.shape) == 3 and sim.shape[1] == N and sim.shape[2] == M)

            # Key2Value Attention
            mask = tf.expand_dims(values_mask, 1)  # batch_size, 1, num_values
            _, alpha = masked_softmax(sim, mask, 2)  # batch_size, num_keys, num_values
            a = tf.matmul(alpha, values)  # batch_size, num_keys, value_vec_size

            # Value2Key Attention
            m = tf.reduce_max(sim, axis=2)  # batch_size, num_keys
            _, beta_raw = masked_softmax(m, keys_mask, 1)  # batch_size, num_keys
            beta = tf.expand_dims(beta_raw, 1)  # batch_size, 1, num_keys
            c_prime = tf.squeeze(tf.matmul(beta, keys), [1])  # batch_size, value_vec_size

            output = tf.concat([a, keys * a, keys * tf.expand_dims(c_prime, 1)], 2)  # batch_size, num_keys, value_vec_size * 3

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return tf.concat([keys, a, keys * a], 2), output, (alpha, beta)


class SelfAttn(object):
    """Module for self attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, hidden_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.hidden_vec_size = hidden_size
        self.hidden_size = hidden_size
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          keys: Tensor shape (batch_size, num_keys, key_vec_size)
          key_mask: Tensor shape (batch_size, num_keys)
            1s where there's real input, 0s where there's padding

        Outputs:
          output: Tensor shape (batch_size, num_keys, self_attn_hidden_size).
            This is the attention output.
        """
        with vs.variable_scope("SelfAttn"):

            N = keys.shape[1]
            v = tf.get_variable("v", shape=[self.hidden_vec_size], initializer=tf.contrib.layers.xavier_initializer())
            W1 = tf.get_variable("W1", shape=[self.key_vec_size, self.hidden_vec_size], initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable("W2", shape=[self.key_vec_size, self.hidden_vec_size], initializer=tf.contrib.layers.xavier_initializer())

            part_j = tf.expand_dims(keys, 1)  # batch_size, 1, num_keys, key_vec_size
            part_i = tf.expand_dims(keys, 2)  # batch_size, num_keys, 1, key_vec_size
            t = tf.tanh(tf.tensordot(part_j, W1, [[3], [0]]) + tf.tensordot(part_i, W2, [[3], [0]]))  # batch_size, num_keys, num_keys, self.hidden_vec_size
            e = tf.tensordot(t, v, [[3], [0]])  # batch_size, num_keys, num_keys
            e_mask = tf.expand_dims(keys_mask, 1)  # batch_size, 1, num_keys
            _, attn_dist = masked_softmax(e, e_mask, 2)  # batch_size, num_keys, num_keys
            a = tf.matmul(attn_dist, keys)  # batch_size, num_keys, key_vec_size

            key_lens = tf.reduce_sum(keys_mask, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, num_keys, self_attn_hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, a, key_lens, dtype=tf.float32)

            output = tf.concat([fw_out, bw_out], 2)  # batch_size, num_keys, self_attn_hidden_size

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return output, attn_dist

class CoAttn2(object):

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

        self.lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(key_vec_size, forget_bias = 1.0)
        self.lstm_fw_cell = DropoutWrapper(self.lstm_fw_cell, input_keep_prob=self.keep_prob)
        self.lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(key_vec_size, forget_bias = 1.0)
        self.lstm_bw_cell = DropoutWrapper(self.lstm_bw_cell, input_keep_prob=self.keep_prob)

        self.key_sentinel = tf.get_variable("key_sentinel", shape = (self.key_vec_size, ), initializer=tf.contrib.layers.xavier_initializer())
        self.value_sentinel = tf.get_variable("val_sentinel", shape = (self.value_vec_size, ), initializer=tf.contrib.layers.xavier_initializer())

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """

        with vs.variable_scope("CoAttn2"):
            num_keys = tf.shape(keys)[1]
            num_values= tf.shape(values)[1]

            #num_values = tf.shape(values)
            phi_mask = tf.tile(tf.convert_to_tensor([[1]]), [tf.shape(keys)[0], 1])

            keys_mask = tf.concat([keys_mask,phi_mask], 1)
            values_mask = tf.concat([values_mask, phi_mask], 1) #(batch_size, num_values+1)

            #key: context, value: question
            key_phi = tf.tile(tf.expand_dims(tf.expand_dims(self.key_sentinel, 0), 0), [tf.shape(keys)[0],1,1])
            value_phi = tf.tile(tf.expand_dims(tf.expand_dims(self.value_sentinel, 0), 0), [tf.shape(keys)[0],1,1])
            keys_sent = tf.concat([keys, key_phi], 1)
            values_sent = tf.concat([values, value_phi], 1)

            values_p = tf.contrib.layers.fully_connected(values_sent, num_outputs=self.value_vec_size, activation_fn=tf.nn.tanh) # (batch_size, question_len, value_vec_size)
            values_pt = tf.transpose(values_p, perm=[0, 2, 1])

            L = tf.matmul(keys_sent, values_pt)# shape (batch_size, num_keys+1, num_values+1)

            #C2Q (a)
            C2Q_attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values+1)
            _, C2Q_attn_dist = masked_softmax(L, C2Q_attn_logits_mask, 2) #shape(batch_size, nums_keys+1, num_values+1)
            C2Q_output = tf.matmul(C2Q_attn_dist, values_p) # shape (batch_size, num_keys + 1, value_vec_size)
            #print(C2Q_output.get_shape())

            #Q2C (b)
            Q2C_attn_logits_mask = tf.expand_dims(keys_mask, 1) # shape (batch_size, 1, nums_keys + 1)
            _, Q2C_attn_dist = masked_softmax(tf.transpose(L, perm=[0, 2, 1]), Q2C_attn_logits_mask, 2) # shape (batch_size, num_values+1, num_keys+1)
            Q2C_output = tf.matmul(Q2C_attn_dist, keys_sent) # shape (batch_size, num_values + 1, key_vec_size)
            #print(Q2C_output.get_shape())

            # second_level_attention(CQ*AC)
            second_level_attn = tf.matmul(C2Q_attn_dist, Q2C_output) # shape (batch_size, num_keys + 1, key_vec_size)
            CD = tf.concat([C2Q_output, second_level_attn], 2)# shape (batch_size, num_keys + 1, 2*key_vec_size)

            input_lens = tf.reduce_sum(keys_mask, reduction_indices=1) - 1 # shape (batch_size)
            inputs = tf.concat([keys_sent, CD], 2)
            inputs = inputs[:, :-1, :]

            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, self.lstm_bw_cell, inputs, input_lens, dtype=tf.float32)
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out, (C2Q_attn_dist, Q2C_attn_dist)

class ModelingLayer(object):
    """Module for modeling layer.

        Similar to BiDAF, we use BiLSTM here.

    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.num_layers = 2
        self.rnn_cell_fw = rnn_cell.MultiRNNCell([
            DropoutWrapper(rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_prob)
            for _ in range(self.num_layers)
        ])
        self.rnn_cell_bw = rnn_cell.MultiRNNCell([
            DropoutWrapper(rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_prob)
            for _ in range(self.num_layers)
        ])

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, context_len, hidden_size*4)
          masks: Tensor shape (batch_size, context_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, context_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("ModelingLayer"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, context_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
