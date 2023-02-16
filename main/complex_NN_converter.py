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

    ############
    # TRAINING #
    ############
    LR=0.1
    LR_DECAY=10.
    EPOCHS=5
    DEBUG=True

    M,N=W1.shape
    col_M=min(M,4) if DEBUG else M
    col_N=min(N,4) if DEBUG else N
    hp_v = {"lr": LR, "lr_decay": LR_DECAY, "layers": [M], "pattern": ["rectangle"], "col_layer_limit": [col_M]}
    hp_u = {"lr": LR, "lr_decay": LR_DECAY, "layers": [N], "pattern": ["rectangle"],
            "col_layer_limit": [col_N]}
    student_u_W1 = ONN.ONN(hp_v, {}, {"epochs": EPOCHS, "loss": ONN.clipped_MSE, "metrics": ONN.MSE})
    student_v_W1 = ONN.ONN(hp_u, {}, {"epochs": EPOCHS, "loss": ONN.clipped_MSE, "metrics": ONN.MSE})
    teacher_student_W1 = TeacherStudent_SVD(W1, student_u_W1, student_v_W1 )


    M,N=W2.shape
    col_M=min(M,4) if DEBUG else M
    col_N=min(N-10,4) if DEBUG else N-10
    hp_v = {"lr": LR, "lr_decay": LR_DECAY, "layers": [M], "pattern": ["triangle"], "col_layer_limit": [col_M]}
    hp_u = {"lr": LR, "lr_decay": LR_DECAY, "layers": [N], "pattern": ["triangle"], "col_layer_limit": [col_N]}

    student_u_W2 = ONN.ONN(hp_u, {}, {"epochs": EPOCHS, "loss": ONN.clipped_MSE, "metrics": ONN.MSE})
    student_v_W2 = ONN.ONN(hp_v, {}, {"epochs": EPOCHS, "loss": ONN.clipped_MSE, "metrics": ONN.MSE})
    teacher_student_W2 = TeacherStudent_SVD(W2, student_u_W2, student_v_W2 )


    # creating data for W2
    A=teacher_student_W1.predict(calib_data)
    A=np.maximum(A,0)

    # Training for W2
    score=teacher_student_W2.fit(A)

    print("W2 MSE : ", score)

    ###########
    # TESTING #
    ###########

    # Photonic prediction
    A=teacher_student_W1.predict(test_X)
    A=np.maximum(A,0)
    Y2=teacher_student_W2.predict(A)

    # accuracy
    score=np.mean(   np.argmax(Y2,axis=1) == np.argmax(test_Y,axis=1)   )
    print("ACCURACY : ", score)


    ###########
    # STORING #
    ###########
    np.save("W1_ONN_U", teacher_student_W1.student_u.W)
    np.save("W1_ONN_V", teacher_student_W1.student_v.W)
    np.save("W1_ONN_S", teacher_student_W1.s)
    # TODO