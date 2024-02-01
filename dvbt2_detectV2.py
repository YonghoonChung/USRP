import numpy as np

## constants
nC = 542
nB = 482
nC_20MHz = int(nC*35/16)
nB_20MHz = int(nB*35/16)
fshift_20MHz = 1/1024*16/35
# shift_val = np.exp(-2j*np.pi*np.arange(4096)/1024)
shift_val_10000 = np.exp(-2j*np.pi*np.arange(10000)*fshift_20MHz)

def dvbt2_detectV2(rx):
    # input should be 20MHz sampled data, not 9.143MHz sampled data
    val = shift_val_10000
    rxs = rx*val
    rxsC = np.hstack((np.zeros(nC_20MHz, dtype=np.complex64), rxs[:-nC_20MHz]))
    corrC = np.correlate(rx,rxsC)[0]
    rxsB = np.hstack((rxs[nB_20MHz:], np.zeros(nB_20MHz, dtype=np.complex64)))
    corrB = np.correlate(rx,rxsB)[0]
    detect = np.abs(corrC+corrB)/(nC_20MHz+nB_20MHz)
    # return 1 if detect>threshold else 0
    return detect

##
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import genDvbt2
    import scipy.signal as sg
    plt.close()
    bits = np.int32(np.round(np.random.random(4*2**11)))
    rx = np.hstack((genDvbt2.genDvbt2(bits),np.zeros(10000-8960, dtype=np.complex64)))
    loops = 1
    time1 = time.time_ns()
    for idx in range(loops):
        # rx_9M143Hz = sg.resample(rx, int(len(rx)*16/35))
        corr = dvbt2_detectV2(rx)
    time2 = time.time_ns()
    
    print(f"detection time: {(time2-time1)/loops/1e3:.2f} us")
    print(f'corr: {corr}')
    
    plt.figure()
    plt.plot(rx.real)
    plt.title("rx signal")
    plt.show()

    # data = np.load('err_log.npy')
