"""
Those function are better illustrated in ONN_training.ipynb.
This file allows to import ONN in another file/notebook.
ONN class is scikit-learn style class of a basic Optical Neural Network.
"""

import jax
from jax import numpy as np
import numpy as npo
from typing import *

random_seed=1
MZI_STRAT="MZI_norm" # "MZI", "MZI_norm", "MZI_noisy"

def get_key():
    """ random number generator """
    global random_seed
    random_seed+=1
    return jax.random.PRNGKey(random_seed)

def no_back(f):
    """ Decorator to avoid backpropagation of the decorated function.
    For example it is useful for "round(x)" """
    def decorated_f(x, *args):
        # Create an exactly-zero expression with Sterbenz lemma that has
        # an exactly-one gradient.
        # URL : https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html
        zero = x - jax.lax.stop_gradient(x)
        return zero + jax.lax.stop_gradient(f(x, *args))
    return decorated_f

# Previous function implement noise AnalogVNN: https://arxiv.org/pdf/2210.10048.pdf
def _rounding_with_thresh(g, r):
    g_abs = np.abs(g)
    g_floor = np.floor(g_abs)
    g_ceil = np.ceil(g_abs)
    prob_floor = 1. - np.abs(g_floor - g)
    do_floor = np.array( r <= prob_floor, dtype=np.float32)
    do_ceil = np.array( r > prob_floor, dtype=np.float32)
    return do_floor * g_floor + do_ceil * g_ceil

@no_back
def precion_reduction(x, p):
    """warning precision=4 means 5 potential value:  {0,0.25,0.5,0.75,1}
    substracting by 1 before calling it is maybe always required"""
    r=0.5
    g = x * p
    f = np.sign(g) * _rounding_with_thresh(g, r) * (1. / p)
    return f

@no_back
def stochastic_reduce_precision(x, p):
    g = x * p
    r=jax.random.uniform(shape=x.shape, key=get_key(), dtype=np.float32)
    f = np.sign(g) * _rounding_with_thresh(g, r) * (1. / p)
    return f

def normalization(x):
    return np.clip(x,-1.,+1.)

@no_back
def additive_noise(x, std):
    noise=jax.random.normal(shape=x.shape, key=get_key(),dtype=np.float32) * std
    return x+noise

def MZI(X, teta):
    R = np.array([
        [np.cos(teta), -np.sin(teta)],
        [np.sin(teta), np.cos(teta)]
    ])
    out_vector = np.dot(R, X)
    return out_vector

def MZI_norm(X, teta):
    X=normalization(X)
    teta=normalization(teta)
    R = np.array([
        [np.cos(teta), -np.sin(teta)],
        [np.sin(teta), np.cos(teta)]
    ])
    out_vector = np.dot(R, X)
    out_vector=normalization(out_vector)
    return out_vector

def MZI_noisy(X, teta):
    p_signal = 2. ** 8
    p_weights = 2. ** 8
    noise_signal = 1e-4
    noise_weights = 1e-4

    X = additive_noise(precion_reduction(normalization(X), p_signal), noise_signal)
    teta = additive_noise(stochastic_reduce_precision(normalization(teta), p_weights), noise_weights)

    y = MZI(X, teta)

    y = precion_reduction(normalization(additive_noise(y, noise_signal)), p_signal)

    return y


def MZI_col0(X, nb_mzi, W):
    strats={"MZI":MZI, "MZI_norm":MZI_norm, "MZI_noisy":MZI_noisy}
    mzi_func = strats[MZI_STRAT]

    # Column type: odd or even ?
    nb_pins = nb_mzi * 2
    if nb_pins == len(X):
        start_pin_id = 0
    elif nb_pins + 2 == len(X):
        start_pin_id = 1
    else:
        raise ValueError("This mesh pattern is not compatible with this input size and #MZIs")

    # pin them
    layer_outputs = []
    if start_pin_id == 1:
        layer_outputs.append(np.array([X[0]]))

    for ID in range(0, nb_mzi):
        # take input vector
        first_pin_pos = 2 * ID + start_pin_id
        second_pin_pos = first_pin_pos + 1
        local_inp = X[first_pin_pos:second_pin_pos + 1]

        # compute the output vector
        local_out = mzi_func(local_inp, W[ID])
        layer_outputs.append(local_out)

    if start_pin_id == 1:
        layer_outputs.append(np.array([X[-1]]))

    Y = np.concatenate(layer_outputs)
    return Y

def split(input_size, centred_window_size)->Tuple[int, int]:
    if centred_window_size>input_size:
        raise ValueError("Unexpected condition: should be true: centred_window_size*2<=input_size")
    part_A_size=int((input_size - centred_window_size) / 2)

    if (part_A_size*2+centred_window_size) == input_size:
        part_B_size = part_A_size
    else:
        part_B_size = part_A_size - 1 # not expected
    if centred_window_size+part_A_size+part_B_size!=input_size:
        raise ValueError("Unexpeced condition: should be true: centred_window_size+part_A_size+part_B_size==input_size")
    return part_A_size, part_B_size
def MZI_col(X, nb_mzi, W):
    strats={"MZI":MZI, "MZI_norm":MZI_norm, "MZI_noisy":MZI_noisy}
    mzi_func = strats[MZI_STRAT]



    pin_identity_part_A,pin_identity_part_B = split(len(X), nb_mzi*2)

    # pin them
    layer_outputs = []
    for i in range(pin_identity_part_A):
        layer_outputs.append(np.array([X[i]]))

    first_pin_pos=0
    second_pin_pos=0
    for ID in range(0, nb_mzi):
        # take input vector
        first_pin_pos = 2 * ID + pin_identity_part_A
        second_pin_pos = first_pin_pos + 1
        local_inp = X[first_pin_pos:second_pin_pos + 1]

        # compute the output vector
        local_out = mzi_func(local_inp, W[ID])
        layer_outputs.append(local_out)

    for i in range(pin_identity_part_A):
        layer_outputs.append(np.array([X[i+second_pin_pos+1]]))

    Y = np.concatenate(layer_outputs)
    return Y

def column_size_for_square_mzi_mesh(matrix_rank)->List[int]: # Example: 6->3,2,3,2,3,2 , 4->2,1,2,1
    cols = matrix_rank
    mzi_per_col = matrix_rank // 2 # when `nb_unis` is "6" `mzi_per_col` is 3
    nb_mzis = []
    for i in range(cols):
        nb_mzi_this_col=mzi_per_col - i % 2
        if nb_mzi_this_col>0:
            nb_mzis.append(nb_mzi_this_col)
    return nb_mzis

def mesh(X, nb_mzis, weights):
    nb_mzi_col = len(weights)

    def recusive_column_builder(id_column=0):
        if id_column == nb_mzi_col - 1:  # last layer. No dependency
            y = MZI_col(X, nb_mzis[id_column], weights[id_column])
        else:
            y = recusive_column_builder(id_column + 1)
            y = MZI_col(y, nb_mzis[id_column], weights[id_column])
        return y

    Y = recusive_column_builder()
    return Y

def glorot_init(nb_mzi):
    weights = jax.random.normal(shape=(nb_mzi,), key=get_key(), dtype=np.float32) * np.sqrt(0.5)
    return weights

def MSE(y_expected, y_pred):
    A,B=split(len(y_pred), len(y_expected))
    try:
        return np.mean((y_expected - y_pred[A:A+len(y_expected)]) ** 2)
    except:
        print("ok")
def accuracy(Y, y_preds):
    nb_correct = 0
    for y_pred, y in zip(y_preds, Y):
        nb_correct += np.argmax(y_pred) == np.argmax(y)
    return float(nb_correct) / len(Y)

def identity(x):
    return x
def relu(x):
    return np.maximum(x,0.)

class ONN:
    def __init__(self, hp):
        assert("layers" in hp)
        assert("lr" in hp)
        assert("lr_decay" in hp)
        assert("epochs" in hp)

        self.hp = hp
        self.W = None
        self.nb_mzis=[]
        self.compiled_forward=None #f(X,W)->pred
        self.compiled_forward_with_loss=None #f2(X,Y,W)->loss
        self.compiled_backward_with_loss=None #b(X,Y,W)->dW

        self.loss=MSE
        self.metrics=MSE

    def initialize(self):
        # building mzis
        self.nb_mzis=[]
        for units in self.hp["layers"]:
            layer_mzis=column_size_for_square_mzi_mesh(units)
            self.nb_mzis.append(layer_mzis)

        # initializing weights
        self.W=[] # 2D array, layer x mzicolumn
        for mzi_layer in self.nb_mzis:
            layer=[]
            for nb_mzi_col in mzi_layer:
                layer.append(glorot_init(nb_mzi_col))
            self.W.append(layer)

        # compilation of the forward
        def forward(X, W):
            y=X
            for i in range(len(W)):
                y=mesh(y, self.nb_mzis[i], W[i])
                if i<len(W)-1: # not the last
                    y=np.maximum(y,0)
            return y
        self.compiled_forward = jax.jit(forward)  # JIT func. is ~3300 times faster!

        # compilation of the backward
        def forward_with_loss(*args):
            y_pred = forward(*(args[0], args[2]))  # 0:X, 2:W
            y_expected = args[1]  # 1:Y
            loss = self.loss(y_expected, y_pred)
            return loss

        self.compiled_forward_with_loss = jax.jit(forward_with_loss)

        backward_with_loss = jax.grad(forward_with_loss, argnums=(-1,))
        self.compiled_backward_with_loss = jax.jit(backward_with_loss)  # JIT circuit is ~3300 times faster!

    def check_initialized(self):
        return not (self.W is None)

    def fit(self, X, Y, X_test=None, Y_test=None):
        # Init
        if not self.check_initialized():
            self.initialize()

        cur_lr = self.hp["lr"]
        ids = npo.array(range(len(X)))
        for e in range(self.hp["epochs"]):  # for each epoch

            # Data shuffling
            npo.random.shuffle(ids)
            X = X[ids]
            Y = Y[ids]

            # Training
            for x_train, y_train in zip(X, Y):  # for each data sample
                # backward phase
                nn_dW = self.compiled_backward_with_loss(x_train, y_train, self.W)[0]

                # Update using the gradient information
                for layer_id, layer_dW in enumerate(nn_dW):
                    for col_id, col_wD in enumerate(layer_dW):
                        self.W[layer_id][col_id] = self.W[layer_id][col_id] - cur_lr * col_wD #error with col_Wd

            # Evaluate
            if X_test is not None and Y_test is not None:
                print(self.evaluate(X_test, Y_test))

            cur_lr/=self.hp["lr_decay"]

    def evaluate(self, X, Y):
        y_preds=self.predict(X)

        m=0
        for yi, ypi in zip(Y, y_preds):
           m += self.metrics(yi, ypi)/len(Y)

        return m

    def predict(self, X):
        preds=[]
        for x in X:
            y=self.compiled_forward(x,self.W)
            preds.append(y)
        return np.array(preds)


if __name__=="__main__":
    hp = {"lr": 0.01, "lr_decay": 1., "layers": [8, 4], "epochs": 8}
    model = ONN(hp)
    model.initialize()

    X = np.array([[0.1, 0.9, 0, 0, 0.1, 0.9, 0, 0], [0.9, 0.1, 0, 0, 0.1, 0.9, 0, 0]])
    Y = np.array([[0.9, 0.1, 0, 0], [0.1, 0.9, 0, 0]])

    model.fit(X, Y, X, Y)

"""

# 0.64163846
# 0.64162725
# 0.64161617
# 0.64160514
# 0.6415936
# 0.641582
# 0.6415713
# 0.64156055
"""
