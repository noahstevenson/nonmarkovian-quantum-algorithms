import grovers_flux_qubit
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import time

_start_tot_ = time.time()
grovers = grovers_flux_qubit.Grovers()

# path for saving file
savedir = r'C:\\Users\\Noah\\Documents\\academics\\coursework\\2019Spring\\p191\\project\\simulation\\cirq\\data\\'
filename = 'flux-qubit-2system5bath-{0}'.format(datetime.today().strftime('%Y%m%d_%H%M%S'))

# the maximum number of Grover's algorithm to sweep through
max_iterations = 10

# the number of batches to run at each iteration-point. Equivalent to multiplying n_simulations by this number.
# Used instead of n_simulations to see how spread changes
batch_size = 10

n_computational_qubits = 2
n_bath_qubits = 5
target = 3
n_simulations = 50
prob_ground = 0.821662

# dictionary to store data
data_dict = {
    'platform': 'Cirq',
    'simulation-structure': 'flux qubit, with interactions structures as in the Pyquil version',
    'Markovian': {'data': np.empty((max_iterations, batch_size))},
    'non-Markovian': {'data': np.empty((max_iterations, batch_size))},
    'n_computational_qubits': n_computational_qubits,
    'n_bath_qubits': n_bath_qubits,
    'target': target,
    'n_simulations': n_simulations,
    'prob_ground': prob_ground,
    'max_iterations': max_iterations,
}

if batch_size != 0:
    data_dict['batch_size'] = batch_size

# iterate through the number of iterations of grover's algorithm
for i in np.arange(max_iterations)+1:
    _start_ = time.time()
    print(f'{i} iterations of Grover\'s algorithm')

    # repeat for the number of batches to gain insight on probability density
    for j in range(batch_size):

        # Markovian
        _bath_type_ = 'Markovian'
        data_dict[_bath_type_]['data'][i-1, j] = grovers.main_oqs(n_computation_qubits=n_computational_qubits,
                                                                  n_bath_qubits=n_bath_qubits,
                                                                  target=target,
                                                                  n_iterations=i,
                                                                  bath_type=_bath_type_,
                                                                  amplitude_damping_constant=1,
                                                                  n_simulations=n_simulations,
                                                                  prob_ground=prob_ground)

        # non-Markovian
        _bath_type_ = 'non-Markovian'
        data_dict[_bath_type_]['data'][i-1, j] = grovers.main_oqs(n_computation_qubits=n_computational_qubits,
                                                                  n_bath_qubits=n_bath_qubits,
                                                                  target=target,
                                                                  n_iterations=i,
                                                                  bath_type=_bath_type_,
                                                                  amplitude_damping_constant=1,
                                                                  n_simulations=n_simulations,
                                                                  prob_ground=prob_ground)
    _end_ = time.time()
    _runtime_minutes_ = str((_end_-_start_)/60.)[:5]
    print(f'completed in {_runtime_minutes_} minutes \n\n')

# record total runtime
_end_tot_ = time.time()
_tot_runtime_minutes_ = str((_end_tot_ - _start_tot_) / 60.)[:5]
print(f'\n Total runtime: {_tot_runtime_minutes_} minutes \n\n')
data_dict['Total runtime'] = f'{_tot_runtime_minutes_} minutes'

# record medians, means, std
data_dict['Markovian']['median'] = np.median(data_dict['Markovian']['data'], axis=1)
data_dict['Markovian']['mean'] = np.mean(data_dict['Markovian']['data'], axis=1)
data_dict['Markovian']['std'] = np.std(data_dict['Markovian']['data'], axis=1)

data_dict['non-Markovian']['median'] = np.median(data_dict['non-Markovian']['data'], axis=1)
data_dict['non-Markovian']['mean'] = np.mean(data_dict['non-Markovian']['data'], axis=1)
data_dict['non-Markovian']['std'] = np.std(data_dict['non-Markovian']['data'], axis=1)


# save the data with Pickle
pickle.dump(data_dict, open(savedir+filename+'.pickle', "wb"))

# plot the results
plt.figure(figsize=(8, 8))
ax = plt.axes()

colors = {'Markovian': {'facecolor': '#7BAFD4', 'edgecolor': '#00356B'},
          'non-Markovian': {'facecolor': '#CD5C5C', 'edgecolor': '#800000'}
          }

for _key_ in colors:
    violinparts = ax.violinplot(np.array(data_dict[_key_]['data'].T),  # data
                                np.arange(max_iterations)+1,  # x-values: positions of violins on plot
                                showextrema=False,
                                showmedians=False,
                                )

    for pc in violinparts['bodies']:
        pc.set_facecolor(colors[_key_]['facecolor'])
        pc.set_edgecolor(colors[_key_]['edgecolor'])

    ax.plot(np.arange(max_iterations)+1, data_dict[_key_]['median'],
            ls='-', lw=2, alpha=0.5, marker='_', markersize=30, mew=2, color=colors[_key_]['edgecolor'],
            label=_key_)

fontsize = 12
ax.set_ylabel(r"Probability of measuring correct outcome $(P)$", fontsize=fontsize)
ax.set_xlabel(r"Iternations of Grover's algorithm $(n)$", fontsize=fontsize)

plt.legend()
plt.savefig(savedir+filename+'.png')
# plt.show()
