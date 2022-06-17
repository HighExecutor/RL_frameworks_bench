from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer, SimpleRNN
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import Huber
from tensorflow.keras import Model, Input
import tensorflow as tf
from tensorflow.keras import backend as K


class DuelQNetwork(Model):
    def __init__(self, state_space, action_space, layers):
        super(DuelQNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.layers_arr = layers
        self.ls = len(layers)

        self.input_layer = InputLayer(input_shape=(state_space,))
        if self.ls > 0:
            self.l1 = Dense(layers[0], activation='relu')
        if self.ls > 1:
            self.l2 = Dense(layers[1], activation='relu')
        if self.ls > 2:
            self.l3 = Dense(layers[2], activation='relu')
        self.v = Dense(1, activation='linear')
        self.a = Dense(action_space, activation='linear')

    def __call__(self, state, **kwargs):
        x = self.input_layer(state)
        if self.ls > 0:
            x = self.l1(x)
        if self.ls > 1:
            x = self.l2(x)
        if self.ls > 2:
            x = self.l3(x)
        v = self.v(x)
        a = self.a(x)
        q = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
        return q


def build_dual(state_space, action_space, layers):
    model = DuelQNetwork(state_space, action_space, layers)
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    return model


def build_dense(state_space, action_space, layers):
    initializer = RandomNormal(mean=0.01, stddev=0.1)
    model = Sequential()
    model.add(InputLayer(input_shape=(state_space,)))
    for i in range(0, len(layers)):
        model.add(Dense(layers[i], activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    return model


def build_rnn(state_space, action_space, obs, layers):
    model = Sequential()
    model.add(SimpleRNN(layers[0], input_shape=(obs, state_space), return_sequences=False, activation='relu'))
    for i in range(1, len(layers)):
        model.add(Dense(layers[i], activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(optimizer=Adam(lr=0.001), loss=Huber())
    return model


def build_pg_network(state_space, action_space, layers):
    input = Input(shape=(state_space,))
    advantages = Input(shape=[1])
    ls = len(layers)
    dense1 = None
    dense2 = None
    dense3 = None
    probs = None
    if ls == 1:
        dense1 = Dense(layers[0], activation='relu')(input)
        probs = Dense(action_space, activation='softmax')(dense1)
    if ls== 2:
        dense1 = Dense(layers[0], activation='relu')(input)
        dense2 = Dense(layers[1], activation='relu')(dense1)
        probs = Dense(action_space, activation='softmax')(dense2)
    if ls == 3:
        dense1 = Dense(layers[0], activation='relu')(input)
        dense2 = Dense(layers[1], activation='relu')(dense1)
        dense3 = Dense(layers[2], activation='relu')(dense2)
        probs = Dense(action_space, activation='softmax')(dense3)

    def custom_loss(y_true, y_pred):
        out = K.clip(y_pred, 1e-8, 1 - 1e-8)
        log_lik = y_true * K.log(out)
        return K.sum(-log_lik * advantages)

    policy = Model(inputs=[input, advantages], outputs=[probs])
    policy.compile(optimizer=Adam(lr=0.001), loss=custom_loss)
    predict = Model(inputs=[input], outputs=[probs])
    return policy, predict
