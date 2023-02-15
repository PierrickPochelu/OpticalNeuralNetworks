from ANN import get_db
from TeacherStudent import TeacherMatrix, TeacherStudent, TeacherStudent_SVD
import ONN
import keras

if __name__=="__main__":
    import numpy as np

    # synthetic data
    # calib_data=np.random.uniform(-1., +1., (100, 256))

    # realistic data
    N = 8
    nb_data = 1000

    (train_X, train_Y), (test_X, test_Y) = get_db("MNIST", N, shuffle=True)
    calib_data = train_X[:nb_data]

    print(calib_data.shape)
    print(np.argmax(train_Y[:nb_data], axis=1))

    keras_path = "../tmp/MNIST/model_0/"
    ann = keras.models.load_model(keras_path)

    W2 = ann.layers[3].get_weights()[0]

    # free memory
    keras.backend.clear_session()
    del ann

    hp = {"lr": 0.1, "lr_decay": 10., "layers": [W2.shape[0]], "pattern": ["triangle"], "col_layer_limit": [W2.shape[0]-10]}
    student_W2 = ONN.ONN(hp, {}, {"epochs": 5, "loss": ONN.clipped_MSE, "metrics": ONN.MSE})


    teacher_W2=TeacherMatrix(W2)

    teacher_student_W2 = TeacherStudent(teacher_W2, student_W2)

    A=np.random.uniform(0, 1, (100, 64)).astype(np.float32)

    score=teacher_student_W2.fit(A)

    print(score)
