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
    nb_data = 100

    (train_X, train_Y), (test_X, test_Y) = get_db("MNIST", N, shuffle=True)
    calib_data = train_X[:nb_data]

    print(calib_data.shape)
    print(np.argmax(train_Y[:nb_data], axis=1))

    keras_path = "../tmp/MNIST/model_0/"
    ann = keras.models.load_model(keras_path)

    W1 = ann.layers[1].get_weights()[0]
    W2 = ann.layers[3].get_weights()[0]

    # free memory
    keras.backend.clear_session()
    del ann


    hp = {"lr": 0.1, "lr_decay": 10., "layers": [W1.shape[0]], "pattern": ["rectangle"], "col_layer_limit": [4]}
    student_v_W1 = ONN.ONN(hp, {}, {"epochs": 5, "loss": ONN.clipped_MSE, "metrics": ONN.MSE})
    student_u_W1 = ONN.ONN(hp, {}, {"epochs": 5, "loss": ONN.clipped_MSE, "metrics": ONN.MSE})
    teacher_student_W1 = TeacherStudent_SVD(W1, student_u_W1, student_v_W1 )

    A=calib_data

    score=teacher_student_W1.fit(A)
    print("W1 MSE : ", score)
    hp_v = {"lr": 0.1, "lr_decay": 10., "layers": [W2.shape[0]], "pattern": ["triangle"], "col_layer_limit": [4]}
    hp_u = {"lr": 0.1, "lr_decay": 10., "layers": [W2.shape[1]], "pattern": ["triangle"], "col_layer_limit": [4]}

    student_v_W2 = ONN.ONN(hp_v, {}, {"epochs": 1, "loss": ONN.clipped_MSE, "metrics": ONN.MSE})
    student_u_W2 = ONN.ONN(hp_u, {}, {"epochs": 1, "loss": ONN.clipped_MSE, "metrics": ONN.MSE})
    teacher_student_W2 = TeacherStudent_SVD(W2, student_u_W2, student_v_W2 )


    A=teacher_student_W1.predict(calib_data)
    A=np.maximum(A,0)
    A=np.random.uniform(0, 1, (100, 64)).astype(np.float32)
    score=teacher_student_W2.fit(A)

    print("W2 MSE : ", score)



    # Photonic prediction
    A=teacher_student_W1.predict(A)
    A=np.maximum(A,0)
    Y2=teacher_student_W2.predict(A)

    # accuracy
    score=np.mean(   np.argmax(Y2,axis=1) == np.argmax(train_Y[:nb_data],axis=1)   )
    print("ACCURACY : ", score)
