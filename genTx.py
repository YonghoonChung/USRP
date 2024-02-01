"""
choose tx sequence
"""
import numpy as np
import genWifi6, genDvbt2, wifi6_detect, dvbt2_detect
import matplotlib.pyplot as plt
import scipy.signal as sg

##
Nseq_wifi6 = 5000
Nseq_dvbt2 = 1000
randseq = np.load("randomSeq.npz")
seq_wifi6 = randseq['seq_wifi6']
seq_dvbt2 = randseq['seq_dvbt2']

bits_wifi6 = np.int32(np.round(np.random.random(12*48*4)))          # random bits for wifi6 data
bits_dvbt2 = np.int32(np.round(np.random.random(4*2048)))           # random bits for dvbt2 data
wifi6_sample = genWifi6.genWifi6(bits_wifi6)
dvbt2_sample = genDvbt2.genDvbt2(bits_dvbt2)
Nwifi6 = len(wifi6_sample)
Ndvbt2 = len(dvbt2_sample)
Ndvbt2_9M143Hz = int(Ndvbt2*16/35)

##
def genTx(wifi6_seq_idx, dvbt2_seq_idx):
    wifi6_seq_idx = int(wifi6_seq_idx)
    dvbt2_seq_idx = int(dvbt2_seq_idx)
    wifi6_seq = seq_wifi6[wifi6_seq_idx]
    dvbt2_seq = seq_dvbt2[dvbt2_seq_idx]
    print("generating TX signal...")
    tx_wifi6 = np.zeros(2000*Nseq_wifi6, dtype=np.complex64)
    tx_dvbt2 = np.zeros(10000*Nseq_dvbt2, dtype=np.complex64)
    for idx in range(Nseq_wifi6):
        if wifi6_seq[idx] == 1:
            bits = np.int32(np.round(np.random.random(12*48*4)))            # random bits for wifi6 data
            tx_wifi6[idx*Nwifi6:(idx+1)*Nwifi6] = genWifi6.genWifi6(bits)
    for idx in range(Nseq_dvbt2):
        if dvbt2_seq[idx] == 1:
            bits = np.int32(np.round(np.random.random(4*2048)))             # random bits for dvbt2 data
            tx_dvbt2[idx*10000:idx*10000+Ndvbt2] += genDvbt2.genDvbt2(bits)
    tx = np.zeros(7500000, dtype=np.complex64)
    tx = np.hstack((tx,tx_wifi6+tx_dvbt2))
    
    return tx

##
if __name__ == "__main__":
    wifi6_seq_idx = 3
    dvbt2_seq_idx = 7
    tx = genTx(wifi6_seq_idx, dvbt2_seq_idx)
    plt.plot(tx.real)
    plt.show()

