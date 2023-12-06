import numpy as np

class LogisticRegression:
    def __init__(self, alpha, regLambda, epsilon, maxNumIters):
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None 
    def sigmoid(self, z):
        # Implement the sigmoid function
        return 1 / (1 + np.exp(-z))

    def computeCost(self, theta, X, y, regLambda):
        # Implement the logistic regression cost function
        h = self.sigmoid(np.dot(X, theta))
        cost = (-1 / len(y)) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        reg_term = (regLambda / (2 * len(y))) * np.sum(theta[1:]**2)
        return cost + reg_term

    def computeGradient(self, theta, X, y, regLambda):
        # Implement the gradient of the logistic regression cost function
        h = self.sigmoid(np.dot(X, theta))
        gradient = (1 / len(y)) * np.dot(X.T, (h - y))
        reg_term = (regLambda / len(y)) * np.concatenate(([0], theta[1:]))
        return gradient + reg_term

    def hasConverged(self, theta, oldTheta):
        # Implement the convergence test
        return np.linalg.norm(theta - oldTheta, 2) <= self.epsilon

    def fit(self, X, y):
        # Add intercept term to X
       
        #X = np.concatenate((np.ones((n, 1)), X), axis=1)
        n, d = X
        self.theta = np.zeros(d)
        #X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        X = np.concatenate((np.ones((n, 1)), X), axis=1)
        # Initialize theta to random values with mean 0
        n, d = X.shape
        self.theta = np.random.randn(X.shape[1])

        for i in range(self.maxNumIters):
            gradient = self.computeGradient(self.theta, X, y, self.regLambda)
            oldTheta = self.theta.copy()
            self.theta -= self.alpha * gradient
            #self.theta -= self.alpha * self.computeGradient(self.theta, X, y, self.regLambda)
            optimized_theta = self.theta - self.alpha * gradient
            if np.linalg.norm(optimized_theta - oldTheta) < self.epsilon:
                break

           # if np.linalg.norm(optimized_theta - self.theta) < self.epsilon:
               # break
            if self.hasConverged(self.theta, oldTheta):
                break
            self.theta = optimized_theta     
    def predict(self, X):
        # Add intercept term to X
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # Predict using the learned parameters
        predictions = self.sigmoid(np.dot(X, self.theta))
        return predictions >= 0.5
