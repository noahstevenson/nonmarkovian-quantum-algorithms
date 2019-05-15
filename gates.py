import numpy as np
import cirq


class ZZ(cirq.TwoQubitGate):

    def __init__(self, theta):
        self.theta = theta

    def _unitary_(self):
        return np.array([
            [np.exp(1j * self.theta / 2), 0, 0, 0],
            [0, np.exp(-1j * self.theta / 2), 0, 0],
            [0, 0, np.exp(-1j * self.theta / 2), 0],
            [0, 0, 0, np.exp(1j * self.theta / 2)]
        ])

    def _circuit_diagram_info_(self, args):
        return f'ZZ({str(self.theta)[:5]})', f'ZZ({str(self.theta)[:5]})'


class XX(cirq.TwoQubitGate):

    def __init__(self, theta):
        self.theta = theta

    def _unitary_(self):
        return (2 ** (-1 / 2)) * np.array([
            [1, 0, 0, np.exp(1j * (self.theta - np.pi / 2))],
            [0, 1, -1j, 0],
            [0, -1j, 1, 0],
            [np.exp(1j * (-self.theta - np.pi / 2)), 0, 0, 1]
        ])

    def _circuit_diagram_info_(self, args):
        return f'XX({str(self.theta)[:5]})', f'XX({str(self.theta)[:5]})'


class YY(cirq.TwoQubitGate):

    def __init__(self, theta):
        self.theta = theta

    def _unitary_(self):
        return np.array([
            [np.cos(self.theta), 0, 0, 1j * np.sin(self.theta)],
            [0, np.cos(self.theta), -1j * np.sin(self.theta), 0],
            [0, -1j * np.sin(self.theta), np.cos(self.theta), 0],
            [1j * np.sin(self.theta), 0, 0, np.cos(self.theta)]
        ])

    def _circuit_diagram_info_(self, args):
        return f'YY({str(self.theta)[:5]})', f'YY({str(self.theta)[:5]})'


class XX_YY(cirq.TwoQubitGate):

    def __init__(self, xx_theta, yy_theta):
        self.xx_theta = xx_theta
        self.yy_theta = yy_theta

    def _unitary_(self):
        _xx_yy_ = \
            np.array([
            [np.cos(self.yy_theta), 0, 0, 1j * np.sin(self.yy_theta)],
            [0, np.cos(self.yy_theta), -1j * np.sin(self.yy_theta), 0],
            [0, -1j * np.sin(self.yy_theta), np.cos(self.yy_theta), 0],
            [1j * np.sin(self.yy_theta), 0, 0, np.cos(self.yy_theta)]
        ])\
            + \
            (2 ** (-1 / 2)) * np.array([
            [1, 0, 0, np.exp(1j * (self.xx_theta - np.pi / 2))],
            [0, 1, -1j, 0],
            [0, -1j, 1, 0],
            [np.exp(1j * (-self.xx_theta - np.pi / 2)), 0, 0, 1]
        ])
        return 2*(_xx_yy_/np.linalg.norm(_xx_yy_))

    def _circuit_diagram_info_(self, args):
        return f'XX_YY({str(self.theta)[:5]})', f'XX_YY({str(self.theta)[:5]})'
