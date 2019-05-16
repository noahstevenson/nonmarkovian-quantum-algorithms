import cirq
import itertools
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import Counter

class XX(cirq.TwoQubitGate):

   def __init__(self, theta):
       self.theta = theta

   def _unitary_(self):
       return (2**(-1/2))*np.array([
           [1, 0, 0, np.exp(1j*(self.theta-np.pi/2))],
           [0, 1, -1j, 0],
           [0, -1j, 1, 0],
           [np.exp(1j*(-self.theta-np.pi/2)), 0, 0, 1]
       ])

   def _circuit_diagram_info_(self, args):
       return f'XX({str(self.theta)[:5]})', f'XX({str(self.theta)[:5]})'


class YY(cirq.TwoQubitGate):

   def __init__(self, theta):
       self.theta = theta

   def _unitary_(self):
       return np.array([
           [np.cos(self.theta), 0, 0, 1j*np.sin(self.theta)],
           [0, np.cos(self.theta), -1j*np.sin(self.theta), 0],
           [0, -1j*np.sin(self.theta), np.cos(self.theta), 0],
           [1j*np.sin(self.theta), 0, 0, np.cos(self.theta)]
       ])

   def _circuit_diagram_info_(self, args):
       return f'YY({str(self.theta)[:5]})', f'YY({str(self.theta)[:5]})'
class ZZ(cirq.TwoQubitGate):

   def __init__(self, theta):
       self.theta = theta

   def _unitary_(self):
       return np.array([
           [np.exp(1j*self.theta/2), 0, 0, 0],
           [0, np.exp(-1j*self.theta/2), 0, 0],
           [0, 0, np.exp(-1j*self.theta/2),0],
           [0, 0, 0, np.exp(1j*self.theta/2)]
       ])

   def _circuit_diagram_info_(self, args):
       return f'ZZ({str(self.theta)[:5]})', f'ZZ({str(self.theta)[:5]})'


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
        return f'XX_YY({str(self.xx_theta)[:5]})', f'XX_YY({str(self.yy_theta)[:5]})'


class XX_YY_ZZ(cirq.TwoQubitGate):

    def __init__(self, xx_theta, yy_theta, zz_theta):
        self.xx_theta = xx_theta
        self.yy_theta = yy_theta
        self.zz_theta = zz_theta

    def _unitary_(self):
        _xx_yy_zz= \
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
        ])\
            + \
            np.array([
            [np.exp(1j*self.zz_theta /2), 0, 0, 0],
            [0, np.exp(-1j*self.zz_theta /2), 0, 0],
            [0, 0, np.exp(-1j*self.zz_theta /2),0],
            [0, 0, 0, np.exp(1j*self.zz_theta /2)]
       ])
        return 2*(_xx_yy_zz/np.linalg.norm(_xx_yy_zz))

    def _circuit_diagram_info_(self, args):
        return f'XX_YY_ZZ({str(self.xx_theta)[:5]})', f'XX_YY_ZZ({str(self.xx_theta)[:5]})'



class QPE:

	def __init__(self):
		self.computation_qubits = None
		self.ancil_qubit = None
		self.bath_qubits = None
		self.n_computation_qubits = None
		self.n_bath_qubits = None
		self.n_simulations = None
		self.c = None
		self.coupling_dict = None
		self.bath_type = None
		self.unknown_gate = None
		self.prob_ground = None 


	def gate(self,phi):
		"""A unitary 1-qubit gate U with an eigen vector |0> and an eigen value exp(2*Pi*i*phi)"""
		gate = cirq.SingleQubitMatrixGate(matrix=np.array([[np.exp(2*np.pi*1.0j*phi), 0], [0, 1]]))
		return gate


	def qft_inv(self):
		qubits = list(self.computation_qubits)
		while len(qubits) > 0:
			q_head = qubits.pop(0)
			yield cirq.H(q_head)
			self.bath_append()
			self.markovianity()
			for i, qubit in enumerate(qubits):
				yield (cirq.CZ**(-1/2.0**(i+1)))(qubit, q_head)

	def generate_couplings_dict(self):
		# TODO: expand this to include individual system-bath coupling thetas
		self.coupling_dict = {
            'SB_ZZ': np.pi / 20,
            'SB_XX': np.pi / 20,
            'SB_YY': np.pi / 20,
            'BB_ZZ': np.pi / 2.5,
            'BB_XX': np.pi / 2.5,
            'BB_YY': np.pi / 2.5,
        }
	def set_io_qubits(self):
		""" Adds n=n_computation_qubits number of input qubits, one output qubit"""
		self.computation_qubits = [cirq.GridQubit(i, 0) for i in range(self.n_computation_qubits)]
		self.ancil_qubit = cirq.GridQubit(self.n_computation_qubits, 0)
		self.bath_qubits = [cirq.GridQubit(i, 0) for i in range(self.n_computation_qubits + 1, self.n_computation_qubits + self.n_bath_qubits + 1)]
		return self.computation_qubits, self.ancil_qubit

	

	def bath_append(self):
		""" Append system-bath and bath-bath interactions at a single moment"""

		system_qubits = self.computation_qubits + [self.ancil_qubit]

		# get coupling strength from coupling_dict dictionary
		self.generate_couplings_dict()
		system_bath_zz = ZZ(theta=self.coupling_dict['SB_ZZ'])
		bath_bath_xx_yy = XX_YY(xx_theta=self.coupling_dict['BB_XX'], yy_theta=self.coupling_dict['BB_YY'])

		for bq, bath_qubit in enumerate(self.bath_qubits):

			# system-bath coupling
			for system_qubit in system_qubits:
				self.c.append(system_bath_zz.on(bath_qubit, system_qubit))

			# bath-bath coupling
			for other_bath_qubit in np.delete(self.bath_qubits, bq):
				self.c.append(bath_bath_xx_yy.on(bath_qubit, other_bath_qubit))




	def partition_function(self, operation=None):
		""" Set the bath qubits according to a partition function """

		assert self.prob_ground <= 1, f"prob_ground must be <1 (it's a probability), not {self.prob_ground}"

		_rand_ = random.random()  # random number to determine if qubit will be flipped

		if operation == 'Append':
			num_excited = 0
			for bath_qubit in self.bath_qubits:
 				if _rand_ > self.prob_ground:
 					self.c.append(cirq.X.on(bath_qubit))
 					num_excited += 1

		elif operation == 'Insert':
			num_excited = 0
			for bath_qubit in self.bath_qubits:
				if _rand_ > self.prob_ground:
					self.c.insert(cirq.X.on(bath_qubit), strategy=cirq.InsertStrategy.NEW)
					num_excited += 1

		else:
			raise TypeError(
				"operation must be either \"Insert\" or \"Append\", not \"{0}\" ".format(operation))








	def markovianity(self):
		if self.bath_type == 'Markovian':

			# Dissipative dynamics
			self.c.append(cirq.AmplitudeDampingChannel(gamma=1).on_each(*self.bath_qubits), strategy=cirq.InsertStrategy.NEW)
			self.partition_function(operation="Append")

		elif self.bath_type == 'non-Markovian':
			pass
		else:
			raise TypeError(
				"bath_type must be either \"Markovian\" or \"non-Markovian\", not \"{0}\" ".format(self.bath_type))


	def make_qpe_circuit(self):
		# define empty circuit
		self.c = cirq.Circuit()
		
		# Thermally initialize bath qubits according to the partition function
		self.partition_function(operation='Append')


        # Apply Walsh-Hadamard transform to put Qubits in superposition
		self.c.append(cirq.H.on_each(*self.computation_qubits))
		self.bath_append()
		self.markovianity()
		

		#Apply unitary 1-qubit gate U gate to ancil_qubit if its corresponding control bit is |1⟩
		for i in range(self.n_computation_qubits):
			self.bath_append()
			self.markovianity()
			self.c.append(cirq.ControlledGate(self.unknown_gate**(2**i)).on(self.computation_qubits[self.n_computation_qubits-i-1], self.ancil_qubit))
			
		

		#Applies reverse fourier transform to 
		self.c.append(self.qft_inv())

		self.c.append(cirq.measure(*self.computation_qubits, key='result'))
		return self.c







	def main(self, n_computation_qubits=10, n_bath_qubits=0, n_simulations=100,
             print_circuit=False, bath_type='Markovian', unknown_gate_phase=.8*np.pi, prob_ground=0.821662):
		""" Runs QPE on (n_computation_qubits)-qubits """


		self.prob_ground = prob_ground
		self.n_computation_qubits = n_computation_qubits
		self.n_bath_qubits = n_bath_qubits
		self.n_simulations = n_simulations

		self.unknown_gate = self.gate(unknown_gate_phase/(2*np.pi))
		
		self.bath_type = bath_type
		self.computation_qubits, self.ancil_qubit = self.set_io_qubits()  # Set up input and output qubits.
		
		
		 # compile QPE

        
		

		simulator = cirq.DensityMatrixSimulator()  # initialize simulator
		# result = simulator.run(self.c, repetitions=n_simulations)  # run the simulations 1000 times
		solutions = []
		fold_func = lambda ms: ''.join(np.flip(ms, 0).astype(int).astype(str))
		for i in range(n_simulations):
			# compile a quantum circuit for QPE algorithm each time to set different qubits according to partition function
			self.c = self.make_qpe_circuit() 
			result = simulator.run(self.c, repetitions=1)  # run the simulation
			hist = result.histogram(key='result', fold_func=fold_func)
			solutions += [hist.most_common(1)[0][0]] # see result

		if print_circuit: print(self.c)  # plot quantum circuit

		count = Counter(solutions)

		estimate_bin = list(map(int, list(count.most_common()[0][0])))
		

		estimate = (sum([float(s)*0.5**(order+1) for order, s in enumerate(estimate_bin)]))* (2*np.pi) #Calculates phase estimate
		
		#returns the chosen gate phase followed by the algorithims estimate, difference between the two, and the most probable measurement
		return unknown_gate_phase,estimate,abs(estimate - unknown_gate_phase), count.most_common()[0][0]



