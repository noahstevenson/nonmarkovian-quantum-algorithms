"""

Realizes Grover's algorithm for any number of system and bath qubits;
Coupling style is as similar as possible to Keshav's Pyquil script;
Uniform coupling between system and bath;
Boltzmann distribution implemented for initialization of qubits via application of X gate;

author: Noah Stevenson
date: 2019-05-08

runs in Cirq 0.5.0
"""

# preamble
import cirq
import itertools
import numpy as np
import random
import time
import gates
import coupling_dict as cdict


class Grovers:

    def __init__(self):
        self.computation_qubits = None
        self.oracle_qubit = None
        self.bath_qubits = None
        self.target = None
        self.n_computation_qubits = None
        self.n_bath_qubits = None
        self.n_iterations = None
        self.n_simulations = None
        self.c = None
        self.amplitude_damping_constant = None
        self.coupling_dict = None
        self.bath_type = None
        self.prob_ground = None

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

    def grovers_step_oqs(self):
        """

        Adds one iteration of Grover's algorithm to the circuit.
        This is in the style of Keshav and Matt's circuit, as opposed to more frequent interactions.
        oqs stands for "open quantum system"

        """

        # oqs
        self.bath_append()

        # Boolean oracle from ArXiV[1703.10535]
        self.append_oracle()

        # oqs
        self.bath_append()

        # diffusion operator
        self.c.append(cirq.H.on_each(*self.computation_qubits))
        self.c.append(cirq.X.on_each(*self.computation_qubits))
        ncz = cirq.ControlledGate(cirq.Z,
                                  num_controls=self.n_computation_qubits - 1)  # phase flip gate; ncz => n-controlled Z
        self.c.append(ncz.on(*self.computation_qubits))
        self.c.append(cirq.X.on_each(*self.computation_qubits))
        self.c.append(cirq.H.on_each(*self.computation_qubits))

        # loss from the bath and reinitialization
        if self.bath_type == 'Markovian':

            # Dissipative dynamics
            self.c.append(cirq.AmplitudeDampingChannel(gamma=1).on_each(*self.bath_qubits), strategy=cirq.InsertStrategy.NEW)
            self.partition_function(operation="Append")

        elif self.bath_type == 'non-Markovian':
            pass
        else:
            raise TypeError(
                "bath_type must be either \"Markovian\" or \"non-Markovian\", not \"{0}\" ".format(self.bath_type))

    def make_grover_circuit_oqs(self):
        """ Assembles the circuit for Grover's algorithm """

        # define empty circuit
        self.c = cirq.Circuit()

        # setting the oracle qubit to 1 before applying H
        self.c.append(cirq.X.on(self.oracle_qubit))

        # Thermally initialize bath qubits according to the partition function
        self.partition_function(operation='Append')

        # Initial Walsh-Hadamard transform
        self.c.append(cirq.H.on_each(*self.computation_qubits, self.oracle_qubit))

        # Repeat n_iterations of Grover's
        for _ in itertools.repeat(None, self.n_iterations):
            self.grovers_step_oqs()

        # Hadamard the oracle at the end; this may be to uncompute it. From Box 6.1 of Nielsen and Chuang
        self.c.append(cirq.H.on(self.oracle_qubit))

        # Measure the result.
        self.c.append(cirq.measure(*self.computation_qubits, key='result'))

        return self.c

    def negate_zeros(self, binary_string):
        """ Negates the qubits (using an X gate) for which the binary representation of the target number is 0. """

        for pos, val in enumerate(list(binary_string)):
            if val == '0':  # checks if the binary value is 0
                self.c.append(cirq.X(self.computation_qubits[pos]))  # adds X gate to negate qubit

    def append_oracle(self):
        """ Generates binary number representation of the target of length(n_computation_qubits)"""

        _target_b_ = format(self.target, "b")

        # raise error if too many bits are needed to represent target
        if len(_target_b_) > self.n_computation_qubits:
            raise ValueError(
                f"Binary representation of target exceeds resource qubits available; \n\n Target: {self.target}; binary representation of target: {_target_b_} \
                ({len(_target_b_)} bits); available qubits: {self.n_computation_qubits}")

        # pad binary representation with 0's if there are unused qubits
        elif len(_target_b_) < self.n_computation_qubits:
            pad_length = int(self.n_computation_qubits) - int(len(_target_b_))
            target_b = '0' * pad_length + _target_b_

        # trivial assignment case
        elif len(_target_b_) == self.n_computation_qubits:
            target_b = _target_b_

        # Negate 0's on each side of the oracle gate so that the oracle runs correctly but doesn't disturb the qubits
        self.negate_zeros(target_b)
        ncx = cirq.ControlledGate(cirq.X, num_controls=self.n_computation_qubits)  # oracle gate: n-controlled NOT
        self.c.append(ncx(*self.computation_qubits, self.oracle_qubit))
        self.negate_zeros(target_b)

    def set_io_qubits(self):
        """ Adds n=n_computation_qubits number of input qubits, one output qubit"""

        self.computation_qubits = [cirq.GridQubit(i, 0) for i in range(self.n_computation_qubits)]
        self.oracle_qubit = cirq.GridQubit(self.n_computation_qubits, 0)
        self.bath_qubits = [cirq.GridQubit(i, 0) for i in range(self.n_computation_qubits + 1,
                                                                self.n_computation_qubits + self.n_bath_qubits + 1)]
        return self.computation_qubits, self.oracle_qubit

    def generate_couplings_dict(self):
        """ Generates dictionary of couplings between qubits """
        self.coupling_dict = cdict.coupling_dict

    def bath_append(self):
        """ Append system-bath and bath-bath interactions at a single moment"""

        system_qubits = self.computation_qubits + [self.oracle_qubit]

        # get coupling strength from coupling_dict dictionary
        self.generate_couplings_dict()
        system_bath_zz = gates.ZZ(theta=self.coupling_dict['SB_ZZ'])
        bath_bath_xx_yy = gates.XX_YY(xx_theta=self.coupling_dict['BB_XX'], yy_theta=self.coupling_dict['BB_YY'])

        for bq, bath_qubit in enumerate(self.bath_qubits):

            # system-bath coupling
            for system_qubit in system_qubits:
                self.c.append(system_bath_zz.on(bath_qubit, system_qubit))

            # bath-bath coupling
            for other_bath_qubit in np.delete(self.bath_qubits, bq):
                self.c.append(bath_bath_xx_yy.on(bath_qubit, other_bath_qubit))

    def main_oqs(self, target=3, n_computation_qubits=2, n_bath_qubits=3, n_iterations=1, n_simulations=1000,
                 amplitude_damping_constant=0, bath_type='Markovian', prob_ground=1, print_runtime=False,
                 print_prob_correct=False, print_circuit=False, run_simulation=True):
        """ Runs Grover's algorithm on (n_computation_qubits)-qubits """

        _start_ = time.time()  # for recording runtime

        self.target = target
        self.n_computation_qubits = n_computation_qubits
        self.n_bath_qubits = n_bath_qubits
        self.n_iterations = n_iterations
        self.n_simulations = n_simulations
        self.amplitude_damping_constant = amplitude_damping_constant
        self.bath_type = bath_type
        self.prob_ground = prob_ground

        counter = 0  # counts number of successful outcomes

        self.computation_qubits, self.oracle_qubit = self.set_io_qubits()  # Set up input and output qubits.
        _simulator_ = cirq.DensityMatrixSimulator()  # initialize simulator

        if print_circuit:
            self.c = self.make_grover_circuit_oqs()
            print(self.c)

        if run_simulation is False:
            exit()
            
        for i in range(n_simulations):

            # compile a quantum circuit for Grover's algorithm each time to set \
            # different qubits according to partition function
            self.c = self.make_grover_circuit_oqs()
            _result_ = _simulator_.run(self.c, repetitions=1)  # run the simulation
            _hist_ = _result_.histogram(key='result')
            counter += _hist_[self.target]  # check if the circuit results in the desired outcome

        prob_correct = counter/self.n_simulations  # covert raw outcomes to probability

        if print_prob_correct: print(prob_correct)
        _stop_ = time.time()

        if print_runtime: print(f'\n runtime: {str(_stop_-_start_)[:5]} seconds')

        return prob_correct
