import numpy as np
import matplotlib.pyplot as plt
import time

def normalize(data):
    data = (data - np.mean(data)) / np.std(data)
    return data
if __name__ == "__main__":
    data = np.load("processedData.npy")
    idx = 4
    start = time.time_ns()
    normData = normalize(data[idx])
    end = time.time_ns()
    print(f'{(end-start)/1e3:.2f} us')
    plt.figure()
    plt.subplot(211)
    plt.plot(data[idx].real)
    plt.subplot(212)
    plt.plot(normData.real)
    plt.show()