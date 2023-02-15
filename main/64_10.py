import numpy as np
import ONN
from ANN import get_db

if __name__ == "__main__":
    N = 8
    DB = "MNIST"

    (train_X, train_y2), (test_X, test_y2) = get_db(DB, cropping_out=24, interpol_out=N, shuffle=True)

    # conversion into jax.numpy array
    train_X = np.array(train_X)
    train_y2 = np.array(train_y2)
    test_X = np.array(test_X)
    test_y2 = np.array(test_y2)

    hp = {"lr": 0.1,
          "lr_decay": 1.,
          "layers": [N*N, 10],
          "col_layer_limit": [N*N, N*N-10],
          "pattern":["rectangle","triangle"]}

    # noise={"sn": 0.01, "wn": 0.01, "sp":2.**6, "wp":2.**6}
    noise = {}
    optim = {"loss": ONN.MSE, "metrics": ONN.accuracy, "epochs": 10}

    model = ONN.ONN(hp, noise, optim, jit=True)

    model.fit(train_X, train_y2, test_X, test_y2)
    model.save(f"ONN_{DB}")

"""



    hp = {"lr": 0.1,
          "lr_decay": 1.1,
          "layers": [N*N, 10],
          "col_layer_limit": [N*N, 10],
          "pattern":["triangle","rectangle"]}
          
Test:  0.5609
Test:  0.6799
Test:  0.6783
Test:  0.6631
Test:  0.6989
Test:  0.7154
Test:  0.6815
Test:  0.6901
Test:  0.6905



    hp = {"lr": 0.1,
          "lr_decay": 1.,
          "layers": [N*N, 10],
          "col_layer_limit": [N*N, N*N-10],
          "pattern":["rectangle","triangle"]}
Test:  0.6704
Test:  0.6911
Test:  0.7128
Test:  0.7022
Test:  0.7177
Test:  0.7178
"""