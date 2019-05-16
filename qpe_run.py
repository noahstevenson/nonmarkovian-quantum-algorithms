import qpe_oqs
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import time


def sweep(max_n_computational_qubits=5,n_bath_qubits =3,phase = .4*2*np.pi):
	qpe = qpe_oqs.QPE()

	#Prograrm will run QPE on 1 to max_n_computational_qubits qubits comparing markovian, nonmarkovian, and ideal results with each other for a chosen phase

	max_n_computational_qubits = max_n_computational_qubits
	n_bath_qubits = n_bath_qubits
	phase = phase
	n_simulations = 100




	#runs QPE
	nonMarkovian = [];
	Markovian = [];
	original = [];


	for i in range(1,max_n_computational_qubits+1):
		nonMarkovian+=[qpe.main(print_circuit=False,n_computation_qubits=i, n_bath_qubits=n_bath_qubits, unknown_gate_phase = phase,bath_type='non-Markovian',n_simulations=n_simulations)]
		Markovian+=[qpe.main(print_circuit=False,n_computation_qubits=i, n_bath_qubits=n_bath_qubits, unknown_gate_phase = phase,bath_type='Markovian',n_simulations=n_simulations)]
		original+=[qpe.main(print_circuit=False,n_computation_qubits=i, n_bath_qubits=0, unknown_gate_phase = phase,n_simulations=n_simulations)]
	



	#Plots results

	plt.figure(figsize=(8, 8))
	ax = plt.axes()

	#Plots the difference between estimate and chosen phase angle
	diffnonMarkovian = [a[2] for a in nonMarkovian]
	diffMarkovian = [a[2] for a in Markovian]
	diffOriginal= [a[2] for a in original]

	size = len(diffnonMarkovian)
	ax.plot(range(1,size+1),diffnonMarkovian, "-o",label = 'non-Markovian')
	ax.plot(range(1,size+1),diffMarkovian, "-o",label = "Markovian")
	ax.plot(range(1,size+1),diffOriginal, "-o",label = "Original")

	fontsize = 12
	ax.set_ylabel(r"Difference of estimate from phase angle(radians)", fontsize=fontsize)
	ax.set_xlabel(r"Number of computation qubits", fontsize=fontsize)

	plt.legend()
	plt.show()

