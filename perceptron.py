"""
    Esta clase preceptron tendra 3 metodos:
    1. - Inicializar el perceptron Pesos iniciales aleatorios.

    2. - Calculo de la salido del perceptron

    3. - Entrenamiento
"""

import numpy as np
import matplotlib.pyplot as plt


# Funciones anonimas o funciones lambda

def sigmoid():
    return lambda x: 1 / (1 + np.e ** -x)


def tanh():
    return lambda x: np.tanh(x)


def relu():
    return lambda x: np.maximum(0, x)


def random_points(n=100):
    x = np.random.uniform(0.0, 1.0, n)
    y = np.random.uniform(0.0, 1.0, n)

    return np.array([x, y]).T


class Perceptron:

    def __init__(self, n_inputs, act_f):
        self._weights = np.random.rand(n_inputs, 1)
        self._bias = np.random.rand()
        self._act_f = act_f
        self._n_inputs = n_inputs

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weigths(self, weights):
        self._weights = weights

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def act_f(self):
        return self._act_f

    @act_f.setter
    def act_f(self, act_f):
        self.act_f = act_f

    @property
    def n_inputs(self):
        return self._n_inputs

    @n_inputs.setter
    def n_inputs(self, n_inputs):
        self._n_inputs = n_inputs

    # Funcion para caluclar el fitfowar

    def predict(self, x):
        return self._act_f(x @ self._weights + self._bias)

    # Metodo de aprendizaje no supervisado
    def fit(self, x, y, epochs=100):
        for i in range(epochs):
            for j in range(self._n_inputs):
                output = self.predict(x[j])
                error = y[j] - output
                self._weights = self._weights + (error * x[j][1])
                self._bias = self._bias + error


def main():
    points = random_points(10000)
    # plt.scatter(points[:, 0], points[:, 1], s=10)
    # plt.show()
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [1],
    ])
    p_or = Perceptron(2, sigmoid())

    yp = p_or.predict(points)
    p_or.fit(x=x, y=y, epochs=1000)
    plt.scatter(points[:, 0], points[:, 1], s=10, c=yp, cmap='GnBu')
    plt.show()


if __name__ == '__main__':
    main()
