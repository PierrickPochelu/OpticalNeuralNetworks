import numpy as np
import ONN

if __name__ == "__main__":
    from ANN import get_db

    N = 10
    DB = "MNIST"

    (train_X, train_y2), (test_X, test_y2) = get_db(DB, cropping_out=24, projection=N, shuffle=True)

    # conversion into jax.numpy array
    train_X = np.array(train_X)
    train_y2 = np.array(train_y2)
    test_X = np.array(test_X)
    test_y2 = np.array(test_y2)

    hp = {"lr": 0.1,
          "lr_decay": 2.,
          "layers": [N],
          "col_layer_limit": [10],
          "pattern":["rectangle"]}

    # noise={"sn": 0.01, "wn": 0.01, "sp":2.**6, "wp":2.**6}
    noise = {}
    optim = {"loss": ONN.clipped_MSE, "metrics": ONN.accuracy, "epochs": 10}

    model = ONN.ONN(hp, noise, optim, jit=True)

    model.fit(train_X, train_y2, test_X, test_y2)
    model.save(f"./tmp/ONN_{DB}")