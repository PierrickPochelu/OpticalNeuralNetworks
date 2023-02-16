import numpy as npo
import tensorflow as tf

from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, regularizers


def get_db(db_name, interpol_out=20, cropping_out=24, projection=-1, shuffle=False):
    if db_name == "MNIST":
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
    elif db_name == "CIFAR10":
        (train_X, train_y), (test_X, test_y) = cifar10.load_data()
        train_X = npo.mean(train_X, axis=3)
        test_X = npo.mean(test_X, axis=3)
    else:
        raise ValueError("DB unknown")

    # Cropping
    if cropping_out!=-1:
        s=train_X.shape[1]
        gap=s-cropping_out
        start_slice=gap//2
        end_slice=s-gap//2
        train_X = train_X[:, start_slice:end_slice, start_slice:end_slice]
        test_X = test_X[:, start_slice:end_slice, start_slice:end_slice]

    # Interpolating
    from scipy.ndimage import zoom
    zr = interpol_out / float(train_X.shape[1])
    train_X = zoom(train_X, (1., zr, zr), order=3)  # order = 3 for cubic interpolation
    test_X = zoom(test_X, (1., zr, zr), order=3)

    # intensity scaling
    n_features = train_X.shape[1] * train_X.shape[2]
    train_X = train_X.reshape((len(train_X), n_features))
    test_X = test_X.reshape((len(test_X), n_features))

    # normalizing
    train_X = (train_X.astype(npo.float32) / (255. / 2.)) - 1.
    test_X = (test_X.astype(npo.float32) / (255. / 2.)) - 1.

    # projection
    if projection>-1:
        n_comp=projection # 10 -> variance explained is only 52%
        from sklearn.decomposition import PCA
        proj = PCA(n_components = n_comp)
        train_X = proj.fit_transform(train_X)
        test_X=proj.transform(test_X)
        print(f"PCA variance explained: {sum(proj.explained_variance_ratio_)}")


    # label processing into one-hot vector
    n_comp = 10
    train_y2 = npo.zeros((len(train_X), n_comp), dtype=npo.float32)
    test_y2 = npo.zeros((len(test_X), n_comp), dtype=npo.float32)
    for i, v in enumerate(train_y):
        train_y2[i][v] = 1.
    for i, v in enumerate(test_y):
        test_y2[i][v] = 1.

    if shuffle:
        ids = npo.array(range(len(train_X)))
        npo.random.shuffle(ids)
        train_X = train_X[ids]
        train_y2 = train_y2[ids]

    # Dimension check
    print(train_X.shape)
    print(train_y2.shape)
    print(test_X.shape)
    print(test_y2.shape)

    return (train_X, train_y2), (test_X, test_y2)


if __name__ == "__main__":
    for DB in ["MNIST", "CIFAR10"]:
        n = 8
        n_features = n * n

        (train_X, train_y2), (test_X, test_y2) = get_db(DB, n, cropping_out=24)

        for ep in range(1):
            model = Sequential()
            model.add(Flatten(input_shape=(n_features,)))
            keras_l1 = Dense(n_features, use_bias=False)
            model.add(keras_l1)
            model.add(Activation("relu"))
            keras_l2 = Dense(10, use_bias=False)
            model.add(keras_l2)
            model.add(Activation('sigmoid'))

            optimizer = optimizers.Adam(lr=0.01)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(train_X, train_y2, epochs=10, batch_size=64, validation_data=(test_X, test_y2))


            w2 = keras_l2.get_weights()[0]
            w1 = keras_l1.get_weights()[0]
            npo.save(f"tmp/{DB}/l1_{ep}", w1)
            npo.save(f"tmp/{DB}/l2_{ep}", w2)

            model.save(f"tmp/{DB}/model_{ep}")

            ypred = model.predict(train_X)
            npo.save(f"tmp/{DB}/y_pred_train_{ep}", ypred)

            ypred = model.predict(test_X)
            npo.save(f"tmp/{DB}/y_pred_train_{ep}", ypred)

            """
            # pour loader:
            # reconstructed_model = keras.models.load_model("model")
            # npo.load(f"tmp/{DB}/l1_{ep}.npy")
            """