import numpy as np

# Load data
path = 'data_path/spam.data'
data = open(path)
data = np.loadtxt(data, delimiter = ' ', dtype = np.float32)

# Split
data = np.array(data)
X, y = data[:, :-1], data[:, -1]

"""
Logistic Regression model with l2 norm regulizer 

Code for Logistic Regression was adapted from the sourse below: 
Author: Koushikl0l (Github)
Date Accessed: October 17, 2024
Licence: N/A
Title: Implementing logistic regression from scratch in Python
Availability: https://github.com/Koushikl0l/Machine_learning_from_scratch/blob/main/LogisticRegression/my_logistic.ipynb

Code for l2 regularization was adapted from the source below:
Author: desertnaut (Stack Overflow)
Date: October 18, 2024
Licence: N/A
Title: Machine learning Logistic Regression L2 regularization
Availability: https://stackoverflow.com/questions/69934443/machine-learning-logistic-regression-l2-regularization
""" 

class LogisticRegression: 
    def __init__(self, lr=0.01, n_iters=1000, lamda=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.lamda = lamda
        self.weights = None
        self.bias =  None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # initialize parameters
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

        # minimize the loss with gradient descent
        for _ in range(self.n_iters):
            y_pred = self.feed_forward(X)
            dz = y_pred - y # difference between predictions and real values

            # compute gradients and apply regularization 
            dw = (1 / n_samples) * np.dot(X.T, dz)

            # L2 regularization gradient
            dw += (self.lamda / n_samples) * self.weights

            db = (1 / n_samples) * np.sum(dz)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # the sigmoid function maps the probability to be between 1 and 0 
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))     

    # loss function
    def loss_function(self, y_true, y_pred):
        # given this a binary classification problem, we use the cross entropy 
        loss = -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
        regularization = (self.lamda / (2 * len(y_true))) * np.sum(self.weights ** 2) # define l2 to the loss function 
        return loss + regularization 
    
    # feed forward 
    def feed_forward(self,X):
        z = np.dot(X, self.weights) + self.bias # summing the inputs, weights and bias
        return self._sigmoid(z) # applying sigmoid to the sum 
            
    def predict(self, X):
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(y_hat)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted] # setting the treshold with respect to sigmoid 

        return np.array(y_predicted_cls)

    def accuracy(self, y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracy

if __name__ == "__main__":

    # train model
    model = LogisticRegression(
        lr=0.1,
        n_iters=1000,
        lamda=0.01
    )

    model.fit(X, y)


    # predictions
    predictions = model.predict(X)


    # evaluate
    print(
        "Logistic Regression classification accuracy:",
        model.accuracy(y, predictions)
    )

    