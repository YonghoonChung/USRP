import numpy as np
import random

number_of_sequences = 10
Nseq_wifi6 = 5000
Nseq_dvbt2 = 1000
possible_values = [0, 1]
dvbt2_probabilities = [2/3, 1/3]
seq_wifi6 = np.zeros((11,Nseq_wifi6))
seq_dvbt2 = np.zeros((11,Nseq_dvbt2))
for idx in range(10):
    dvbt2_variables = random.choices(possible_values, weights=dvbt2_probabilities, k=Nseq_dvbt2)


    non_dvbt2_count = dvbt2_variables.count(0)
    probabilities = [1/2, 1/2]
    random_variables2 = random.choices(possible_values,weights=probabilities, k=non_dvbt2_count*5)

    wifi6_variables = np.zeros(5000, dtype=int)

    k = 0
    for i in range(1000):
        if dvbt2_variables[i] == 0:
            for j in range(5):
                wifi6_variables[i*5+j] = random_variables2[k]
                k += 1 
                
    seq_dvbt2[idx+1] = dvbt2_variables
    seq_wifi6[idx+1] = wifi6_variables
seq_wifi6[0] = 0
seq_dvbt2[0] = 0

# seq_wifi6 = np.int32(np.round(np.random.random((number_of_sequences+1, Nseq_wifi6))))
# seq_dvbt2 = np.int32(np.round(np.random.random((number_of_sequences+1, Nseq_dvbt2))))

np.savez("randomSeq.npz", seq_wifi6=seq_wifi6, seq_dvbt2=seq_dvbt2)
print('finished generation')