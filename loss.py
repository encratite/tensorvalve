from keras.backend import square, mean, sqrt

def root_mean_squared_error(y_true, y_pred):
    return sqrt(mean(square(y_pred - y_true), axis=-1))