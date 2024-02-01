import numpy as np

## constants
nC = 542
nB = 482
shift_val = np.exp(-2j*np.pi*np.arange(4096)/1024)
shift_val_4560 = np.exp(-2j*np.pi*np.arange(4560)/1024)

def dvbt2_detect(rx_9M143Hz):
    # calculating shift_val inside this function takes about 2x more time for detection
    # shift_val = np.exp(-2j*np.pi*np.arange(len(rx_9M143Hz))/1024)
    if len(rx_9M143Hz) == 4096:
        val = shift_val
    elif len(rx_9M143Hz) == 4560:
        val = shift_val_4560
    else:
        val = np.exp(-2j*np.pi*np.arange(len(rx_9M143Hz))/1024)
    rxs_9M143Hz = rx_9M143Hz*val
    rxsC_9M143Hz = np.hstack((np.zeros(nC, dtype=np.complex64), rxs_9M143Hz[:-nC]))
    corrC_9M143Hz = np.correlate(rx_9M143Hz,rxsC_9M143Hz)[0]
    rxsB_9M143Hz = np.hstack((rxs_9M143Hz[nB:], np.zeros(nB, dtype=np.complex64)))
    corrB_9M143Hz = np.correlate(rx_9M143Hz,rxsB_9M143Hz)[0]
    detect = np.abs(corrC_9M143Hz+corrB_9M143Hz)/(nC+nB)
    # return 1 if detect>threshold else 0
    return detect

##
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import genDvbt2
    import scipy.signal as sg

    bits = np.int32(np.round(np.random.random(4*2**11)))
    rx = genDvbt2.genDvbt2(bits)
    loops = 2
    time1 = time.time_ns()
    for idx in range(loops):
        rx_9M143Hz = sg.resample(rx, int(len(rx)*16/35))
        corr = dvbt2_detect(rx_9M143Hz)
    time2 = time.time_ns()

    print(f"detection time: {(time2-time1)/loops/1e3:.2f} us")
    print(f'corr: {corr}')

    plt.figure()
    plt.plot(rx.real)
    plt.title("rx signal")
    plt.show()
