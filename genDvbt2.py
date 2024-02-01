import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.signal as sg
## P2 Symbol Generation
'''
P2_Spec
8K FFT
The number of P2 = 2
Useful_carriers = 4472 * 2 = 8944
SISO
'''
# P1 preamble
dvbt2_p1 = loadmat("./data/generatedP1.mat")["data"].flatten()
dvbt2_p1 /= dvbt2_p1.std()

# The number of Active carriers
P2_Active_carriers_number = 6817

# P2_Pilot index
P2_Pilot_index = []
for i in range(0, 6817):
    if i%3 == 0:
        P2_Pilot_index.append(i)

P2_Tone_reservation_index = [ 106, 109, 110, 112, 115, 118, 133, 142, 163, 184, 206, 247, 445,
                        461, 503, 565, 602, 656, 766, 800, 922,
                        1094, 1108, 1199, 1258, 1726, 1793, 1939, 2128, 2714, 3185, 3365,
                        3541, 3655, 3770, 3863, 4066, 4190,
                        4282, 4565, 4628, 4727, 4882, 4885, 5143, 5192, 5210, 5257, 5261,
                        5459, 5651, 5809, 5830, 5986, 6020,
                        6076, 6253, 6269, 6410, 6436, 6467, 6475, 6509, 6556, 6611, 6674,
                        6685, 6689, 6691, 6695, 6698, 6701
                        ] # The number of Tone Reservation is 72

P2_Useful_carriers_index = []

for i in range(0, P2_Active_carriers_number):
    if i not in P2_Pilot_index and i not in P2_Tone_reservation_index:
        P2_Useful_carriers_index.append(i) # The number of P2_Useful_carriers is 4472

# mapping
M = 4                       # M=4 for 16-QAM
mapping_table = {           # mapping
    (0,0,0,0) :  3+3j,
    (0,0,0,1) : -3+3j,
    (0,0,1,0) :  3-3j,
    (0,0,1,1) : -3-3j,
    (0,1,0,0) :  1+3j,
    (0,1,0,1) : -1+3j,
    (0,1,1,0) :  1-3j,
    (0,1,1,1) : -1-3j,
    (1,0,0,0) :  3+1j,
    (1,0,0,1) : -3+1j,
    (1,0,1,0) :  3-1j,
    (1,0,1,1) : -3-1j,
    (1,1,0,0) :  1+1j,
    (1,1,0,1) : -1+1j,
    (1,1,1,0) :  1-1j,
    (1,1,1,1) : -1-1j
}

mapping_table = {           # gray code mapping
    (0,0,0,0) :  3+3j,
    (0,0,0,1) :  3+1j,
    (0,0,1,0) :  1+3j,
    (0,0,1,1) :  1+1j,
    (0,1,0,0) :  3-3j,
    (0,1,0,1) :  3-1j,
    (0,1,1,0) :  1-3j,
    (0,1,1,1) :  1-1j,
    (1,0,0,0) : -3+3j,
    (1,0,0,1) : -3+1j,
    (1,0,1,0) : -1+3j,
    (1,0,1,1) : -1+1j,
    (1,1,0,0) : -3-3j,
    (1,1,0,1) : -3-1j,
    (1,1,1,0) : -1-3j,
    (1,1,1,1) : -1-1j
}

'''
p1 preamble --> 2048 samples, samp rate = 9.143MHz --> 4480 samples, samp rate = 20MHz
8k ofdm with 8MHz BW --> 8192 samples, samp rate = 8MHz --> 20480 samples, samp rate = 20MHz
'''

## DVB-T2 Generation Function
def genDvbt2(bits):
    # assuming 2K modulation in 8kHz BW
    # todo: should be modified to actual dvbt2 signal
    bits_reshape = bits.reshape(-1,4)
    symbol = np.array([mapping_table[tuple(row)] for row in bits_reshape])
    symbol[0] = 0
    dvbt2_signal = np.fft.ifft(symbol)
    dvbt2_signal /= dvbt2_signal.std()
    dvbt2_9M143Hz = np.hstack((dvbt2_p1, dvbt2_signal))
    # dvbt2_9M143Hz = np.hstack((dvbt2_signal, dvbt2_signal))
    dvbt2 = sg.resample(dvbt2_9M143Hz, int(len(dvbt2_9M143Hz)*35/16))
    return dvbt2


##
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    bits = np.int32(np.round(np.random.random(4*(2**11))))         # data bits for 2k fft
    x = genDvbt2(bits)
    x = np.hstack((x,np.zeros(10000-8960, dtype=np.complex64)))
    plt.figure()
    plt.title('DVBT2')
    plt.plot(x.real)
    plt.plot(x.imag)
    plt.show()