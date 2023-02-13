from typing import *
from keras import backend as K
import numpy as np
from ONN import MSE, accuracy
import ONN


class TeacherStudent:
    def __init__(self, teacher, student, info: Dict = {}):
        assert (hasattr(teacher, "predict"))

        assert (hasattr(student, "predict"))
        assert (hasattr(student, "fit"))
        self.teacher = teacher
        self.student = student

        self.lr = info.get("lr", 0.1)
        self.lr_decay = info.get("lr_decay", 10)
        self.lr_min = info.get("lr_min", 1e-5)
        self.epsilon = info.get("epsilon", 1e-5)
        self.patience = info.get("patience", 1)

        self.fraction_valid = info.get("fraction_valid", 0.1)
        self.fraction_test = info.get("fraction_test", 0.1)

    def evaluate_student(self, X, Y):
        student_preds = self.student.predict(X)
        return MSE(Y, student_preds)

    def fit(self, X):
        nb_data = len(X)

        nb_valid = int(nb_data * self.fraction_valid)
        nb_test = int(nb_data * self.fraction_test)
        nb_train = int(nb_data - (nb_valid + nb_test))

        X_train = X[:nb_train]
        Y_train = self.teacher.predict(X_train)
        X_valid = X[nb_train:nb_valid + nb_train]
        Y_valid = self.teacher.predict(X_valid)
        X_test = X[nb_valid + nb_train:]
        Y_test = self.teacher.predict(X_test)

        cur_lr = self.lr

        while cur_lr > self.lr_min:
            cont = True
            last_mse = np.inf
            while cont:
                # Training
                for i in range(self.patience):
                    self.student.fit(X_train, Y_train)

                # Validation
                valid_mse = self.evaluate_student(X_valid, Y_valid)
                if valid_mse + self.epsilon > last_mse:
                    cont = False
                last_mse = valid_mse

            # lr update
            cur_lr /= self.lr_decay
            print("New LR: ", cur_lr)

        print("Stopping")
        print("Training score: ", self.evaluate_student(X_train, Y_train))
        print("Validation score: ", self.evaluate_student(X_valid, Y_valid))
        test = self.evaluate_student(X_test, Y_test)
        print("Testing score: ", test)
        return test


class TeacherMatrix:
    def __init__(self, W):
        self.W = W

    def predict(self, X):
        """ X shape is (nb_data, nb_features)"""
        X_formated_for_mat_mult = X.reshape((len(X), X.shape[1], 1))
        teacher_pred = []
        for x in X_formated_for_mat_mult:
            pred_Y = np.dot(self.W.T, x)
            teacher_pred.append(pred_Y)

        return np.array(teacher_pred).squeeze()  # ouput shape is (nb_data, nb_preds)


class TeacherStudent_SVD:
    def __init__(self, w, student_u, student_v, teacher_student_config={}):
        assert (hasattr(student_u, "predict"))
        assert (hasattr(student_u, "fit"))
        assert (hasattr(student_v, "predict"))
        assert (hasattr(student_v, "fit"))

        u, s, vT = np.linalg.svd(w, full_matrices=False)
        self.w = w
        self.s = s
        self.s_diag = np.diag(s)
        self.u = u
        self.vT = vT

        self.teacher_u = TeacherMatrix(u)
        self.teacher_v = TeacherMatrix(vT)
        self.student_u = student_u
        self.student_v = student_v

        self.teacher_student_u = TeacherStudent(self.teacher_u, student_u, teacher_student_config)
        self.teacher_student_v = TeacherStudent(self.teacher_v, student_v, teacher_student_config)

    def fit(self, simulated_X):
        self.teacher_student_u.fit(simulated_X)
        self.teacher_student_v.fit(simulated_X)

        # Compute ground truth
        offset = int(1. - (self.teacher_student_u.fraction_test + self.teacher_student_u.fraction_valid))
        test_X = simulated_X[offset:]
        Y = []
        for x in test_X:
            yp = np.dot(self.w, x)
            Y.append(yp)
        Y = np.array(Y)

        # prediction
        Y_preds = self.predict(test_X)

        score = MSE(Y, Y_preds)
        return score

    def predict(self, X):
        Y_pred = []
        N=X.shape[1] # shape: nb_data, nb_features
        for xi in X:
            # compute v.T*x
            vTx = self.student_v.predict([xi])

            svTx = np.dot(self.s_diag, vTx.reshape((N, 1)))

            # compute u*v.T*x
            usvTx = self.student_u.predict([svTx.squeeze()])

            Y_pred.append(usvTx)

        return np.array(Y_pred, dtype=np.float32)

    def predict0(self, X):
        Y_pred = []

        for xi in X:
            # compute v.T*x
            vTx = self.student_v.predict([xi])

            # compute u*v.T*x
            uvTx = self.student_u.predict(vTx)

            # compute diag_s*u*v.T*x
            yp = np.dot(self.s_diag, uvTx.reshape((uvTx.shape[0], uvTx.shape[1], 1)))

            Y_pred.append(yp)

        return np.array(Y_pred, dtype=np.float32)

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        loss = MSE(Y, Y_pred)
        return loss

    def get_usv(self):
        return self.student_u.W, self.s, self.student_v.W


def from_matrix_to_usv_photonic(calibration_data, matrix) -> Tuple[float, TeacherStudent_SVD]:
    N = matrix.shape[0] # shape: dim_input, dim_output
    hp_config = {"lr": 0.1, "lr_decay": 10., "layers": [N], "col_layer_limit": [8], "pattern":["rectangle"]}
    training_config = {"epochs": 10, "loss": ONN.MSE, "metrics": ONN.MSE}
    noise_config = {}
    student_u = ONN.ONN(hp_config, noise_config, training_config)
    student_v = ONN.ONN(hp_config, noise_config, training_config)
    tsv = TeacherStudent_SVD(matrix, student_u, student_v, {})

    score = tsv.fit(calibration_data)

    return score, tsv


def from_matrix_to_photonic(calibration_data, matrix) -> Tuple[float, TeacherStudent_SVD]:
    N = matrix.shape[0]
    hp_config = {"lr": 0.1, "lr_decay": 10., "layers": [N], "col_layer_limit": [8], "pattern":["rectangle"]}
    training_config = {"epochs": 10, "loss": ONN.MSE, "metrics": ONN.MSE}
    noise_config={}

    student = ONN.ONN(hp_config, noise_config, training_config)

    teacher = TeacherMatrix(matrix)
    tsv = TeacherStudent(teacher, student, training_config)

    score = tsv.fit(calibration_data)

    return score, tsv


"""
This is a variant of Knowledge Distilling paper (2014) where the goal is not compress a neural network but converts 
a standard ann to a photonic one (ONN below). I tried other simpler technics but they do not scale very well when matrix is larger than 4x4.

There the procedure:
For each tensor named W in the standard ANN:
1. We computed Y=W.X . Therefore we produce a couple (X,Y)
2. The ONN matrix O with rectangular mesh is trained to fit (X,Y)
3. add O to the list of photonic_layer

Reconstruct the ONN using photonic_layer by adding the ANN activation function.



teacher-student procedure where the teacher is a standard NN implemented with Tensorflow, 
the student is an Optical Neural Network. Calibration data given allows the teacher "to teach" .  We vary this data to 
do conclusion, we observe random number works well like realistic data.

"""
import keras


def from_keras_to_photonic(calibration_data, test_X, test_Y, keras_path, prior_usv=True) -> Tuple[
    float, List[TeacherStudent_SVD]]:
    # extract weights
    ann = keras.models.load_model(keras_path)


    # Conversion tensor by tensor and flowing calibration data
    onn_usv = []
    current_data = calibration_data.copy()
    for layer in ann.layers:
        if isinstance(layer, keras.layers.Dense):
            ann_matrix = layer.get_weights()[0]
            # Train model
            if prior_usv:
                score, tsv = from_matrix_to_usv_photonic(current_data, ann_matrix)
            else:
                score, tsv = from_matrix_to_photonic(current_data, ann_matrix)

            print("Local loss: ", score)
            onn_usv.append(tsv)
        elif isinstance(layer, keras.layers.Activation):
            current_data = np.maximum(current_data, 0)
        else:
            print("Ignored layer: ", str(layer))

    # get global loss
    fraction_valid = onn_usv[0].student_u.fraction_valid
    fraction_test = onn_usv[0].student_u.fraction_test
    nb_test_data = int(1. - (fraction_valid + fraction_test))
    global_test_X = calibration_data[nb_test_data:]

    ann_pred_test = ann.predict(global_test_X)

    # remove the model
    K.clear_session()
    del ann

    # prediction
    def reconstructed_ONN(onn_usv, input_x):
        x = onn_usv[0].predict(input_x)
        x = np.maximum(x, 0)
        x = onn_usv[1].predict(x)

    global_test_Y = reconstructed_ONN(onn_usv, global_test_X)

    # global score
    loss = MSE(ann_pred_test, global_test_Y)
    print("Global loss: ", loss)

    # global metric
    pred = reconstructed_ONN(onn_usv, test_X)
    metric_score = accuracy(test_Y, pred)
    print("Final metric: ", metric_score)

    return metric_score


"""
if __name__ == "__main__":
    import numpy as np

    N = 8
    nb_data = 100

    from ANN import get_db
    (train_X, train_Y), (test_X, test_Y) = get_db("MNIST", N, shuffle=True)


    keras_path = "tmp/MNIST/model_0/"

    score = from_keras_to_photonic(train_X[:nb_data], test_X[:nb_data], test_Y[:nb_data], keras_path, prior_usv=False)
"""
if __name__ == "__main__":
    import numpy as np

    N = 16
    nb_data = 100

    from ANN import get_db
    (train_X, train_Y), (test_X, test_Y) = get_db("MNIST", N, shuffle=True)
    calib_data=train_X[:nb_data]


    # Free memory
    del train_X, train_Y, test_X, test_Y

    keras_path = "tmp/MNIST/model_0/"
    model=keras.models.load_model(keras_path)
    matrix=model.layers[1].get_weights()[0]
    del model

    from_matrix_to_photonic(calib_data, matrix)
