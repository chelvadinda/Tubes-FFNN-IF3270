import numpy as np

class FFNN:
    def __init__(self, layer_sizes, activations, loss_function='mse', init_method='normal', seed=None, regularization=None, reg_lambda=0.01):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_function = loss_function
        self.init_method = init_method
        self.seed = seed
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.weights = []
        self.biases = []
        
        self.init_weights()

    def init_weights(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        for i in range(len(self.layer_sizes) - 1):
            if self.init_method == 'zero':
                w = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
                b = np.zeros((1, self.layer_sizes[i+1]))
            elif self.init_method == 'uniform':
                lower_bound = -0.1  
                upper_bound = 0.1
                w = np.random.uniform(lower_bound, upper_bound, (self.layer_sizes[i], self.layer_sizes[i+1]))
                b = np.random.uniform(lower_bound, upper_bound, (1, self.layer_sizes[i+1]))
            elif self.init_method == 'normal':
                w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1])
                b = np.random.randn(1, self.layer_sizes[i+1])
            else:
                raise Exception("Metode inisialisasi bobot tidak dikenali!")

            self.weights.append(w)
            self.biases.append(b)

    def activation(self, x, func):
        if func == 'linear':
            return x
        elif func == 'relu':
            return np.maximum(0, x)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func == 'tanh':
            return np.tanh(x)
        elif func == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise Exception("Fungsi aktivasi tidak dikenali!")

    def activation_derivative(self, x, func):
        if func == 'linear':
            return np.ones_like(x)
        elif func == 'relu':
            return (x > 0).astype(float)
        elif func == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif func == 'tanh':
            return 1 - np.tanh(x)**2
        elif func == 'softmax':
            return np.ones_like(x)
        else:
            raise Exception("Fungsi aktivasi tidak dikenali untuk turunan!")

    def forward(self, X):
        a = X
        self.a_s = [X]  
        self.z_s = []   

        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.activation(z, self.activations[i])
            self.z_s.append(z)
            self.a_s.append(a)
        
        return a

    def compute_loss(self, y_true, y_pred):
        if self.loss_function == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.loss_function == 'binary_crossentropy':
            epsilon = 1e-12
            y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss_function == 'categorical_crossentropy':
            epsilon = 1e-12
            y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            raise Exception("Loss function tidak dikenali!")

    def compute_loss_derivative(self, y_true, y_pred):
        if self.loss_function == 'mse':
            return 2 * (y_pred - y_true) / y_true.shape[0]
        elif self.loss_function == 'binary_crossentropy' or self.loss_function == 'categorical_crossentropy':
            epsilon = 1e-12
            y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
            return (y_pred - y_true) / y_true.shape[0]
        else:
            raise Exception("Loss function tidak dikenali!")

    def backward(self, X, y):
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        
        delta = self.compute_loss_derivative(y, self.a_s[-1])
        
        for i in reversed(range(len(self.weights))):
            delta *= self.activation_derivative(self.z_s[i], self.activations[i])
            
            grads_w[i] = np.dot(self.a_s[i].T, delta)
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            
            if i != 0:
                delta = np.dot(delta, self.weights[i].T)
        
        return grads_w, grads_b
    
    def save(self, path):
        np.savez(path, weights=self.weights, biases=self.biases)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])

