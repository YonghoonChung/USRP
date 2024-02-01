import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

## Information for Wifi6 Signal Generation
wifi6_preamble = loadmat("./data/wifi6_preamble.mat")["preamble"].flatten()
wifi6_preamble /= wifi6_preamble.std()

# pilot sequence
pilot_seq = np.array([1,1,1,-1], np.complex64)
pilot_coef = [1,1,1,  -1,-1,-1,1,  -1,-1,-1,-1,  1,1,-1,1,  -1,-1,1,1,  -1,1,1,-1,  1,1,1,1,  1,1,-1,1,
            1,1,-1,1,  1,-1,-1,1,  1,1,-1,1,  -1,-1,-1,1,  -1,1,-1,-1,  1,-1,-1,1,  1,1,1,1,  -1,-1,1,1,
            -1,-1,1,-1,  1,-1,1,1,  -1,-1,-1,1,  1,-1,-1,-1,  -1,1,-1,-1,  1,-1,1,1,  1,1,-1,1,  -1,1,-1,1,
            -1,-1,-1,-1,  -1,1,-1,1,  1,-1,1,-1,  1,1,1,-1,  -1,1,-1,-1,  -1,1,1,1,  -1,-1,-1,-1,  -1,-1,-1
            ]

# mapping
M = 4                       # M=4 for 16-QAM
mapping_table = {           # gray code mapping
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}

sc_symbol=np.array([-26,-25,-24,-23,-22, -20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8, -6,-5,-4,-3,-2,-1,
                    1,2,3,4,5,6,  8,9,10,11,12,13,14,15,16,17,18,19,20,  22,23,24,25,26])

sc_pilot = np.array([-21,-7,7,21])

## Wifi6 Generation Function
def genWifi6(bits):
    '''
    Wifi6 Signal Generation Using 16-QAM OFDM with 1/4 GI
    '''
    NOFDM = len(bits)//(48*4)
    if len(bits)%(48*4) != 0:
        NOFDM += 1
        bits = np.hstack((bits,np.zeros(48*4-len(bits)%(48*4), dtype=np.int32)))
    bits_reshape = bits.reshape(-1,48,4)
    wifi6 = np.zeros(len(wifi6_preamble)+NOFDM*80, np.complex64)
    wifi6[:len(wifi6_preamble)] = wifi6_preamble    # add preamble
    for sigidx in range(NOFDM):
        symbol = np.zeros(64, dtype=np.complex64)
        symbol[sc_symbol] = np.array([mapping_table[tuple(row)] for row in bits_reshape[sigidx]])
        symbol[sc_pilot] = pilot_seq*pilot_coef[sigidx]
        wifi_signal = np.fft.ifft(symbol)
        wifi_signal = np.hstack((wifi_signal[-16:], wifi_signal))
        wifi6[len(wifi6_preamble)+sigidx*80:len(wifi6_preamble)+(sigidx+1)*80] = wifi_signal/wifi_signal.std()
    return wifi6


##
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    bits = np.int32(np.round(np.random.random(48*4*12)))
    x = genWifi6(bits)
    plt.figure()
    plt.title('Wi-Fi 6')
    plt.plot(x.real)
    plt.plot(x.imag)
    plt.show()