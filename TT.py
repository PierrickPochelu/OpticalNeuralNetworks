import numpy as np
def ranks(tt_cores):
    ranks = []
    for i in range(len(tt_cores)):
        s = tt_cores[i].shape[0]
        ranks.append(s)
    s = tt_cores[-1].shape[-1]
    ranks.append(s)
    return np.stack(ranks, axis=0)


class TensorTrain:
    def __init__(self, tt_cores, tt_shapes, tt_ranks):
        self.tt_cores = tt_cores
        self.tt_shapes = tt_shapes
        self.tt_ranks = tt_ranks


def from_tt_to_arr(tt: TensorTrain, original_shape: tuple) -> np.ndarray:
    """Converts a TensorTrain into a regular tensor or matrix."""
    tt_ranks = tt.tt_ranks
    res = tt.tt_cores[0]
    for i in range(1, len(tt.tt_cores)):
        res = np.reshape(res, (-1, tt_ranks[i]))
        curr_core = np.reshape(tt.tt_cores[i], (tt_ranks[i], -1))
        res = np.matmul(res, curr_core)
    return np.reshape(res, original_shape)


def _from_nd_arr_to_tt(arr: np.ndarray, max_tt_rank: int = 10) -> TensorTrain:
    """Converts a given Numpy array to a TT-tensor of the same shape."""
    static_shape = list(arr.shape)
    dynamic_shape = arr.shape
    d = static_shape.__len__()
    max_tt_rank = np.array(max_tt_rank).astype(np.int32)
    if max_tt_rank.size == 1:
        max_tt_rank = (max_tt_rank * np.ones(d + 1)).astype(np.int32)
    ranks = [1] * (d + 1)
    tt_cores = []
    are_tt_ranks_defined = True
    for core_idx in range(d - 1):
        curr_mode = static_shape[core_idx]
        if curr_mode is None:
            curr_mode = dynamic_shape[core_idx]
        rows = ranks[core_idx] * curr_mode
        arr = np.reshape(arr, [rows, -1])
        columns = arr.shape[1]
        if columns is None:
            columns = np.shape(arr)[1]

        u, s, vT = np.linalg.svd(arr, full_matrices=False)
        v = vT.T.T.T  # anti-transpose

        # arr == u @ diag(s) @ vT
        if max_tt_rank[core_idx + 1] == 1:
            ranks[core_idx + 1] = 1
        else:
            ranks[core_idx + 1] = min(max_tt_rank[core_idx + 1], rows, columns)
        u = u[:, 0:ranks[core_idx + 1]]
        s = s[0:ranks[core_idx + 1]]
        v = v[:, 0:ranks[core_idx + 1]]
        core_shape = (ranks[core_idx], curr_mode, ranks[core_idx + 1])
        tt_cores.append(np.reshape(u, core_shape))
        arr = np.matmul(np.diag(s), np.transpose(v))
    last_mode = static_shape[-1]
    if last_mode is None:
        last_mode = dynamic_shape[-1]
    core_shape = (ranks[d - 1], last_mode, ranks[d])
    tt_cores.append(np.reshape(arr, core_shape))
    if not are_tt_ranks_defined:
        ranks = None
    return TensorTrain(tt_cores, static_shape, ranks)


def from_arr_to_tt(mat: np.ndarray, shape: tuple, max_tt_rank: int = 10) -> TensorTrain:
    """Converts a given matrix or vector to a TT-matrix."""

    # transpose
    shape = np.array(shape)
    tens = np.reshape(mat, shape.flatten())  # Warning there:
    d = len(shape[0])
    transpose_idx = np.arange(2 * d).reshape(2, d).T.flatten()
    transpose_idx = list(transpose_idx.astype(int))
    while len(transpose_idx) < len(tens.shape):
        transpose_idx.append(len(transpose_idx))
    tens = np.transpose(tens, transpose_idx)

    new_shape = np.prod(shape, axis=0)
    tens = np.reshape(tens, new_shape)
    tt_tens = _from_nd_arr_to_tt(tens, max_tt_rank)

    tt_cores = []
    static_tt_ranks = list(tt_tens.tt_ranks)
    dynamic_tt_ranks = ranks(tt_tens.tt_cores)
    for core_idx in range(d):
        curr_core = tt_tens.tt_cores[core_idx]
        curr_rank = static_tt_ranks[core_idx]
        if curr_rank is None:
            curr_rank = dynamic_tt_ranks[core_idx]
        next_rank = static_tt_ranks[core_idx + 1]
        if next_rank is None:
            next_rank = dynamic_tt_ranks[core_idx + 1]
        curr_core_new_shape = [curr_rank, shape[0, core_idx], shape[1, core_idx], next_rank]

        # patch:
        # if max_tt_rank==2:
        # while np.prod(curr_core_new_shape) < np.prod(curr_core.shape):
        #  curr_core_new_shape.insert(1, 2)
        try:
            curr_core = np.reshape(curr_core, curr_core_new_shape)
        except:
            print("Error")

        tt_cores.append(curr_core)
    return TensorTrain(tt_cores, shape, tt_tens.tt_ranks)


def tt_dot(a: TensorTrain, b: TensorTrain) -> TensorTrain:
    """Multiplies two TT-matrices and returns the TT-matrix of the result."""
    ndims = len(a.tt_cores)
    einsum_str = 'aijb,cjkd->acikbd'
    result_cores = []
    for core_idx in range(ndims):
        a_core = a.tt_cores[core_idx]
        b_core = b.tt_cores[core_idx]

        curr_res_core = np.einsum(einsum_str, a_core, b_core)  # <------------ 2x2 multiplication

        res_left_rank = a.tt_ranks[core_idx] * b.tt_ranks[core_idx]
        res_right_rank = a.tt_ranks[core_idx + 1] * b.tt_ranks[core_idx + 1]
        left_mode = a.tt_shapes[0][core_idx]
        right_mode = b.tt_shapes[1][core_idx]

        core_shape = [res_left_rank, left_mode, right_mode, res_right_rank]
        # while np.prod(core_shape) < np.prod(curr_res_core.shape):
        #  core_shape.insert(1, 2)
        curr_res_core = np.reshape(curr_res_core, core_shape)

        result_cores.append(curr_res_core)
    res_shape = (a.tt_shapes[0], b.tt_shapes[1])
    out_ranks = [a_r * b_r for a_r, b_r in zip(a.tt_ranks, b.tt_ranks)]
    return TensorTrain(result_cores, res_shape, out_ranks)

def from_mat_to_tt_with_SVD(weights, X, Y, rank=8):
    # SVD decomposition
    u, s, vT = np.linalg.svd(weights, full_matrices=False)
    V = vT.T.T.T  # anti-transpose
    s_diag = np.diag(s)
    reconstructed_weights = np.dot(np.dot(u, s_diag), vT)
    print(f"Weigths reconstruction after SVD MSE: {np.mean((weights - reconstructed_weights) ** 2)}")

    # from array to rank2 tt format
    N=weights.shape[0]
    tt_core_dim=tuple([2 for i in range(int(np.log2(N)))])
    tt_input_dim=tuple([1 for i in range(int(np.log2(N)))])

    X_tt = from_arr_to_tt(X, (tt_core_dim, tt_input_dim), max_tt_rank=rank)
    vt_tt = from_arr_to_tt(vT, (tt_core_dim, tt_core_dim), max_tt_rank=rank)
    s_tt = from_arr_to_tt(s_diag, (tt_core_dim, tt_core_dim), max_tt_rank=rank)
    u_tt = from_arr_to_tt(u, (tt_core_dim, tt_core_dim), max_tt_rank=rank)


    # compute TT weights
    w_tt = tt_dot(tt_dot(u_tt, s_tt), vt_tt)
    # w_tt=from_arr_to_tt(weights, ((2, 2), (2, 2)), max_tt_rank=2)

    # Only for checking
    weights_reconstructed = from_tt_to_arr(w_tt, weights.shape)

    for i in range(len(w_tt.tt_cores)):
        print("tt core core.shapes:" , w_tt.tt_cores[i].shape)
        print("tt core ranks:" , w_tt.tt_ranks[i])
    print("tt core shapes:" , w_tt.tt_shapes)
    print(w_tt.tt_cores)

    print(f"Weigths reconstruction after 2x2 TT decomp. MSE: {np.mean((weights - weights_reconstructed) ** 2)}")

    Y_tt = tt_dot(w_tt, X_tt)
    Y_reconstructed = from_tt_to_arr(Y_tt, Y.shape)

    print(f"Expected Y:", Y)
    print(f"Reconstructed Y:", Y_reconstructed)
    print(f"Prediction MSE: {np.mean((Y - Y_reconstructed) ** 2)}")

if __name__=="__main__":
    import numpy as np

    np.random.seed(0)
    N = 512
    weights = np.random.uniform(-1., +1., (N, N))
    X = np.random.uniform(-1., +1., (N, 1))

    Y = np.dot(weights, X)
    print("X")
    print(X)
    print("weights")
    print(weights)
    print("Y")
    print(Y)

    from_mat_to_tt_with_SVD(weights, X, Y)