import os
import joblib
import numpy as np
import logging


class Perceptron:
    def __init__(self, eta=None, epochs=None):
        if (eta is not None) and (epochs is not None):
            self.eta=eta
            self.epochs = epochs

            self.weights = np.random.rand(3)*1e-4

            logging.info("Initial weights are \n")
            logging.info(self.weights)
        
    def _z_value(self, inputs, weights):
        return np.dot(inputs, weights)
        
    def _activation_func(self, z):
        return np.where(z>=0, 1, 0)
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        
        x_with_bias = np.c_[x, -np.ones((len(x), 1))]
        
        for epoch in range(1, self.epochs+1):
            logging.info("#"*50)
            logging.info(f"\n Epoch {epoch}/{self.epochs}\n")
            
            z = self._z_value(x_with_bias, self.weights)
            y_hat = self._activation_func(z)
            logging.info(f"Prediction is : is {y_hat}\n")
            
            error = y - y_hat
            logging.info(f"Error : is\n{error}\n")
            
            self.weights = self.weights + self.eta * np.dot(x_with_bias.T, error)
            logging.info(f"Updated weights are  : is\n{self.weights}\n")
        
        self.error = error
        # logging.info(f"Total loss is {self.error.sum()}")
        logging.info("model fitting complete")
    
    def predict(self, x):
        x.append(-1)
        z = np.dot(x, self.weights)
        y_hat = self._activation_func(z)
        return y_hat
    
    @property
    def total_loss(self):
        return self.error.sum()
    
    def save(self, filename, model_dir="model"):
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, filename)
        joblib.dump(self, file_path)
        logging.info(f"Model is save to {file_path}")
        
    def load(self, filename, model_dir="model"):
        """
        method for loading the model with the path of saved model
        """
        file_path = os.path.join(model_dir, filename)
        return joblib.load(file_path)
