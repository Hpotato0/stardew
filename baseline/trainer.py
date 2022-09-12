from model import build_model, call_back_set
from tensorflow import keras

import tensorflow as tf

## Model 훈련 함수
class Trainer():
    def __init__(self, model, x_train, y_train, x_val, y_val, batch_size, name, verbose = 1):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.model = model.model
        self.batch_size = batch_size
        self.name = name
        self.verbose = verbose

    def train(self, epoch):
        # Train the model
        with tf.device('/device:GPU:0'):
            history1 = self.model.fit(
                self.x_train, self.y_train,
                epochs = epoch,
                steps_per_epoch=len(self.x_train) / self.batch_size,
                batch_size=self.batch_size,
                validation_data=(self.x_val, self.y_val),
                validation_steps=len(self.x_val) / self.batch_size,
                shuffle=True,
                callbacks=call_back_set(self.name, epoch, self.batch_size),
                verbose=self.verbose)