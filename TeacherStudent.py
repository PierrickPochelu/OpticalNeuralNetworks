import numpy as np
def MSE(y,y_pred):
    return np.mean( (y-y_pred)**2 )

class TeacherStudent:
    def __init__(self, teacher, student, info={}):
        assert(hasattr(teacher, "predict"))

        assert(hasattr(student, "predict"))
        assert(hasattr(student, "fit"))
        self.teacher=teacher
        self.student=student

        self.lr=info.get("lr", 0.1)
        self.lr_decay=info.get("lr_decay", 10)
        self.lr_min=info.get("lr_min", 1e-4)
        self.epsilon=info.get("epsilon", 1e-5)
        self.patience=info.get("patience", 1)

        self.fraction_valid=info.get("fraction_valid", 0.1)
        self.fraction_test=info.get("fraction_test", 0.1)

    def evaluate_student(self, X, Y):
        student_preds=self.student.predict(X)
        return MSE(Y, student_preds)

    def fit(self, X):
        nb_data=len(X)

        nb_valid=int(nb_data*self.fraction_valid)
        nb_test=int(nb_data*self.fraction_test)
        nb_train=int(nb_data-(nb_valid+nb_test))

        X_train=X[:nb_train]
        Y_train=self.teacher.predict(X_train)
        X_valid=X[nb_train:nb_valid+nb_train]
        Y_valid=self.teacher.predict(X_valid)
        X_test=X[nb_valid+nb_train:]
        Y_test=self.teacher.predict(X_test)

        del X

        cur_lr=self.lr

        while cur_lr > self.lr_min:
            cont=True
            last_mse=np.inf
            while cont:
                # Training
                for i in range(self.patience):
                    self.student.fit(X_train, Y_train)

                # Validation
                valid_mse=self.evaluate_student(X_valid, Y_valid)
                print("Student: ", valid_mse)
                if valid_mse+self.epsilon>last_mse:
                    cont=False
                last_mse=valid_mse

            # lr update
            cur_lr/=self.lr_decay
            print("New LR: ", cur_lr)

        print("Stopping")
        print("Training score: ", self.evaluate_student(X_train, Y_train))
        print("Validation score: ", self.evaluate_student(X_valid, Y_valid))
        print("Testing score: ", self.evaluate_student(X_test, Y_test))


class TeacherMatrix:
    def __init__(self, W):
        self.W = W

    def predict(self, X):
        """ X shape is (nb_data, nb_features)"""
        X_formated_for_mat_mult = X.reshape(( len(X), X.shape[1], 1))
        teacher_pred = []
        for x in X_formated_for_mat_mult:
            pred_Y = np.dot(self.W, x)
            teacher_pred.append(pred_Y)

        return np.array(teacher_pred).squeeze()  # ouput shape is (nb_data, nb_preds)

class TeacherStudent_SVD:
    def __init__(self, w, student_u, student_v, teacher_student_config={}):
        assert(hasattr(student_u, "predict"))
        assert(hasattr(student_u, "fit"))
        assert(hasattr(student_v, "predict"))
        assert(hasattr(student_v, "fit"))

        u, s, vT = np.linalg.svd(w, full_matrices=False)
        self.s=s
        self.s_diag = np.diag(s)
        self.u=u
        self.vT=vT

        self.teacher_u=TeacherMatrix(u)
        self.teacher_v=TeacherMatrix(vT)
        self.student_u=student_u
        self.student_v=student_v

        self.teacher_student_u = TeacherStudent(self.teacher_u, student_u, teacher_student_config)
        self.teacher_student_v = TeacherStudent(self.teacher_v, student_v, teacher_student_config)


    def fit(self, simulated_X):
        self.teacher_student_u.fit(simulated_X)
        self.teacher_student_v.fit(simulated_X)

    def predict(self, X):
        Y_pred=[]

        for xi in X:
            # compute v.T*x
            vTx = self.student_v.predict([xi])

            svTx=np.dot(self.s_diag, vTx.reshape((16,1)))

            # compute u*v.T*x
            usvTx = self.student_u.predict( [svTx.squeeze()] )


            Y_pred.append(usvTx)

        return np.array(Y_pred, dtype=np.float32)

    def predict0(self, X):
        Y_pred=[]

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
        Y_pred=self.predict(X)
        loss=MSE(Y, Y_pred)
        return loss


if __name__=="__main__":
    import numpy as np
    N = 16  # <----- Small value like "4" is easier to follow, bigger values are more realistic like "16"
    weights = np.random.normal(0, np.sqrt(2. / (2 * N)), (N, N))  # normal glorot  initialization
    X = np.random.uniform(-1., 1., (N, 1))

    Y = np.dot(weights, X)
    print("X")
    print(X)
    print("weights")
    print(weights)
    print("Y")
    print(Y)

    nb_data = 1000
    simulated_X = np.random.uniform(-1., +1., (nb_data, weights.shape[1]))


    import ONN
    hp = {"lr": 0.1, "lr_decay": 1., "layers": [weights.shape[1]]}
    tr = {"epochs": 5, "loss": ONN.MSE, "metrics": ONN.MSE}
    student_u = ONN.ONN(hp, {}, tr)
    student_v = ONN.ONN(hp, {}, tr)
    tsv = TeacherStudent_SVD(weights, student_u, student_v, {})

    tsv.fit(simulated_X)

    Y_reconstructed = tsv.predict([X.squeeze()])
    Y_expected = Y.squeeze()
    print(f"Y expected: {Y_expected}")
    print(f"Y reconstructed: {Y_reconstructed}")
    score=np.mean((Y_expected - Y_reconstructed) ** 2)
    print(f"Prediction MSE: {score}")

    """
    Y_reconstructed = tsv.predict0([X.squeeze()])
    Y_expected = Y.squeeze()
    print(f"Y expected: {Y_expected}")
    print(f"Y reconstructed: {Y_reconstructed}")
    score=np.mean((Y_expected - Y_reconstructed) ** 2)
    print(f"Prediction MSE: {score}")
    """