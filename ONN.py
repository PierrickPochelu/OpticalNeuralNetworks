import jax
from jax import numpy as np
import numpy as npo
from typing import *
import time

random_seed = 1


def get_key():
    """ random number generator """
    global random_seed
    random_seed += 1
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
    do_floor = np.array(r <= prob_floor, dtype=np.float32)
    do_ceil = np.array(r > prob_floor, dtype=np.float32)
    return do_floor * g_floor + do_ceil * g_ceil


@no_back
def precion_reduction(x, p):
    """warning precision=4 means 5 potential value:  {0,0.25,0.5,0.75,1}
    substracting by 1 before calling it is maybe always required"""
    r = 0.5
    g = x * p
    f = np.sign(g) * _rounding_with_thresh(g, r) * (1. / p)
    return f


@no_back
def stochastic_reduce_precision(x, p):
    g = x * p
    r = jax.random.uniform(shape=x.shape, key=get_key(), dtype=np.float32)
    f = np.sign(g) * _rounding_with_thresh(g, r) * (1. / p)
    return f


def normalization(x):
    return np.clip(x, -1., +1.)


@no_back
def additive_noise(x, std):
    noise = jax.random.normal(shape=x.shape, key=get_key(), dtype=np.float32) * std
    return x + noise


def MZI(X, c_w, s_w):
    """ shape input (2,1) shape output (2,1) """
    R = np.array([
        [c_w, -s_w],
        [s_w, c_w]
    ])
    out_vector = np.dot(R, X)
    return out_vector


def split(input_size, centred_window_size) -> Tuple[int, int]:
    if centred_window_size > input_size:
        raise ValueError("Unexpected condition: should be true: centred_window_size*2<=input_size")
    part_A_size = int((input_size - centred_window_size) / 2)

    if (part_A_size * 2 + centred_window_size) == input_size:
        part_B_size = part_A_size
    else:
        part_B_size = part_A_size - 1  # not expected
    if centred_window_size + part_A_size + part_B_size != input_size:
        raise ValueError("Unexpeced condition: should be true: centred_window_size+part_A_size+part_B_size==input_size")
    return part_A_size, part_B_size


def MZI_col(X, nb_mzi, W):
    assert (len(X) % 2 == 0)
    pin_identity_part_A, pin_identity_part_B = split(len(X), nb_mzi * 2)

    cos_W = np.cos(W)
    sin_W = np.sin(W)
    pinnable_X = X.reshape((len(X) // 2, 2, 1))

    # pin them
    layer_outputs = []
    for i in range(pin_identity_part_A):
        layer_outputs.append(np.array([X[i]]))

    def pinning(ID):
        first_pin_pos = 2 * ID + pin_identity_part_A
        second_pin_pos = first_pin_pos + 1
        local_inp = np.array([[X[first_pin_pos]], [X[second_pin_pos]]], dtype=npo.float32)
        local_out = MZI(local_inp, cos_W[ID], sin_W[ID])
        return local_out

    window_out = jax.vmap(pinning)(npo.arange(nb_mzi))
    window_out = np.concatenate(window_out)
    layer_outputs.extend(window_out)

    """
    # for loop version is slower
    for ID in range(nb_mzi):
        layer_outputs.extend(pinning(ID))
    """

    for i in range(pin_identity_part_B):
        ID = pin_identity_part_A + 2 * nb_mzi + i
        layer_outputs.append(np.array([X[ID]]))

    Y = np.concatenate(layer_outputs)

    return Y


def column_size_for_square_mzi_mesh(matrix_rank, col_layer_limit=1000000, pattern="rectangle") -> List[int]:  # Example: 6->3,2,3,2,3,2 , 4->2,1,2,1
    assert(pattern in {"rectangle", "triangle"})
    cols = min(matrix_rank, col_layer_limit)
    mzi_per_col = matrix_rank // 2  # when `nb_unis` is "6" `mzi_per_col` is 3
    nb_mzis = []

    print(matrix_rank)
    if pattern=="rectangle":
        for i in range(cols):
            nb_mzi_this_col = mzi_per_col - i % 2
            if nb_mzi_this_col > 0:
                nb_mzis.append(nb_mzi_this_col)
    elif pattern=="triangle":
        for i in range(cols):
            nb_mzi_this_col = mzi_per_col - i
            if nb_mzi_this_col > 0:
                nb_mzis.append(nb_mzi_this_col)
    else:
        raise ValueError("Unexpected pattern")
    return nb_mzis


def mesh0(X, nb_mzis, weights, noise: Dict = {}):
    nb_mzi_col = len(weights)

    if "sp" in noise and "sn" in noise:
        X = additive_noise(precion_reduction(normalization(X), noise["sp"]), noise["sn"])

    if "wp" in noise and "wn" in noise:
        weights = [additive_noise(stochastic_reduce_precision(normalization(w), noise["wp"]), noise["wn"]) for w in
                   weights]

    def recusive_column_builder(id_column=0):
        if id_column == nb_mzi_col - 1:  # last layer. No dependency
            y = MZI_col(X, nb_mzis[id_column], weights[id_column])
        else:
            y = recusive_column_builder(id_column + 1)
            y = MZI_col(y, nb_mzis[id_column], weights[id_column])
        return y

    Y = recusive_column_builder()

    if "sp" in noise and "sn" in noise:
        Y = additive_noise(precion_reduction(normalization(Y), noise["sp"]), noise["sn"])

    return Y


def mesh(X, nb_mzis, weights, noise: Dict = {}):
    nb_mzi_col = len(weights)

    if "sp" in noise and "sn" in noise:
        X = additive_noise(precion_reduction(normalization(X), noise["sp"]), noise["sn"])

    if "wp" in noise and "wn" in noise:
        weights = [additive_noise(stochastic_reduce_precision(normalization(w), noise["wp"]), noise["wn"]) for w in
                   weights]

    for id_column in range(len(nb_mzis)):
        if id_column == 0:  # last layer. No dependency
            y = MZI_col(X, nb_mzis[id_column], weights[id_column])
        else:
            y = MZI_col(y, nb_mzis[id_column], weights[id_column])

    if "sp" in noise and "sn" in noise:
        y = additive_noise(precion_reduction(normalization(y), noise["sp"]), noise["sn"])

    return y


def glorot_init(nb_mzi):
    weights = jax.random.normal(shape=(nb_mzi,), key=get_key(), dtype=np.float32) * np.sqrt(0.5)
    return weights

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def MSE(y_expected, y_pred):
    """ the lost takes 1 data """
    A, B = split(len(y_pred), len(y_expected))
    return np.mean((y_expected - softmax(y_pred[A:A + len(y_expected)])) ** 2)

def clipped_MSE(y_expected, y_pred):
    """ the lost takes 1 data """
    A, B = split(len(y_pred), len(y_expected))
    y_expected=np.clip(y_expected, -1., +1.)
    return np.mean((y_expected - y_pred[A:A + len(y_expected)]) ** 2)


def accuracy(Y, y_preds):
    """ the metrics takes the overall database labels/predictions"""
    nb_correct = 0
    for y_pred, y in zip(y_preds, Y):
        A, B = split(len(y_pred), len(y))
        y_pred = y_pred[A:A + len(y)]
        nb_correct += np.argmax(y_pred) == np.argmax(y)
    return np.float32(nb_correct) / len(Y)


def identity(x):
    return x


def relu(x):
    return np.maximum(x, 0.)


class ONN:
    def __init__(self, hp, noise: Dict = {}, optim: Dict = {}, jit=True):
        assert ("layers" in hp)
        assert ("lr" in hp)
        assert ("lr_decay" in hp)
        self.epochs = optim.get("epochs", 10)
        self.loss = optim.get("loss", MSE)
        self.metrics = optim.get("metrics", accuracy)

        # Architecture
        self.layers=hp["layers"]
        self.col_layer_limit = hp["col_layer_limit"]
        self.pattern = hp["pattern"]
        assert( len(self.layers) == len(self.col_layer_limit) == len(self.pattern) )

        self.hp = hp

        self.noise = noise
        self.jit = jit
        self.W = None
        self.nb_mzis = []
        self.compiled_forward = None  # f(X,W)->pred
        self.compiled_forward_with_loss = None  # f2(X,Y,W)->loss
        self.compiled_backward_with_loss = None  # b(X,Y,W)->dW

    def initialize(self):
        # building mzis
        self.nb_mzis = []
        for units, col_limit, mesh_pattern in zip(self.layers, self.col_layer_limit, self.pattern):
            layer_mzis = column_size_for_square_mzi_mesh(units, col_limit, mesh_pattern)
            self.nb_mzis.append(layer_mzis)

        # initializing weights
        self.W = []  # 2D array, layer x mzicolumn
        for mzi_layer in self.nb_mzis:
            layer = []
            for nb_mzi_col in mzi_layer:
                layer.append(glorot_init(nb_mzi_col))
            self.W.append(layer)

        # compilation of the forward
        def forward(X, W):
            y = X
            for i in range(len(W)):
                y = mesh(y, self.nb_mzis[i], W[i], self.noise)
                if i < len(W) - 1:  # not the last
                    y = relu(y)
            return y
            #return softmax(y)

        # compilation of the backward
        def forward_with_loss(*args):
            y_pred = forward(*(args[0], args[2]))  # 0:X, 2:W
            y_expected = args[1]  # 1:Y
            loss = self.loss(y_expected, y_pred)
            return loss

        backward_with_loss = jax.grad(forward_with_loss, argnums=(-1,))

        self.compiled_forward = jax.jit(forward) if self.jit else forward  # JIT func. is ~3300 times faster!
        # self.compiled_forward_with_loss = jax.jit(forward_with_loss) if self.jit else forward_with_loss # finalement on s'en sert pas
        self.compiled_backward_with_loss = jax.jit(
            backward_with_loss) if self.jit else backward_with_loss  # JIT circuit is ~3300 times faster!

    def check_initialized(self):
        return not (self.W is None)

    def fit(self, X, Y, X_test=None, Y_test=None):
        # Init
        if not self.check_initialized():
            self.initialize()

        cur_lr = self.hp["lr"]
        ids = npo.array(range(len(X)))
        for e in range(self.epochs):  # for each epoch

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
                        self.W[layer_id][col_id] = self.W[layer_id][col_id] - cur_lr * col_wD  # error with col_Wd

            # Evaluate
            if X_test is not None and Y_test is not None:
                print("Test: ", self.evaluate(X_test, Y_test))

            cur_lr /= self.hp["lr_decay"]

    def evaluate(self, X, Y):
        y_preds = self.predict(X)
        return self.metrics(Y, y_preds)

    def predict(self, X):
        preds = []
        for x in X:
            y = self.compiled_forward(x, self.W)
            preds.append(y)
        return np.array(preds)

    def save(self, path):
        npo.save(path, self.W)

    def restore(self, path):
        self.W = npo.load(path + ".npy")


if __name__ == "__main__":
    from ANN import get_db

    n = 8
    n_features = n * n
    DB = "MNIST"

    (train_X, train_y2), (test_X, test_y2) = get_db(DB, interpol_out=n, shuffle=True)

    # conversion into jax.numpy array
    train_X = np.array(train_X)
    train_y2 = np.array(train_y2)
    test_X = np.array(test_X)
    test_y2 = np.array(test_y2)

    hp = {"lr": 0.1,
          "lr_decay": 2.,
          "layers": [n_features, 10],
          "col_layer_limit": [8, 8],
          "pattern":["rectangle", "rectangle"]}
    # noise={"sn": 0.01, "wn": 0.01, "sp":2.**6, "wp":2.**6}
    noise = {}
    optim = {"loss": MSE, "metrics": accuracy, "epochs": 10}

    model = ONN(hp, noise, optim, jit=True)

    model.fit(train_X, train_y2, test_X, test_y2)
    model.save(f"./tmp/ONN_{DB}")
