import numpy as np
import matplotlib.pyplot as plt

class FIFOBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add_samples(self, samples):
        # Append the samples to the buffer
        self.buffer.extend(samples)

        # If the buffer size exceeds the capacity, remove the oldest samples
        if len(self.buffer) > self.capacity:
            excess_samples = len(self.buffer) - self.capacity
            self.buffer = self.buffer[excess_samples:]

    def get_samples(self, num_samples):
        # Get the next num_samples from the buffer, if available
        if len(self.buffer) >= num_samples:
            samples = self.buffer[:num_samples]
            self.buffer = self.buffer[num_samples:]
            return samples
        else:
            return None  # Not enough samples in the buffer



if __name__ == "__main__":
    # Initialize the FIFO buffer with a capacity of 2000 samples
    fifo = FIFOBuffer(2000)
    Ns = np.arange(2000)
    data = np.random.uniform(-1, 1, 10000)
    # Simulate adding 10000 samples in chunks of 2000
    for i in range(5):
        chunk = data[i*2000:(i+1)*2000]  # Replace this with your actual IQ data
        fifo.add_samples(chunk)
        plt.figure()
        plt.plot(Ns, fifo.get_samples(2000))
        plt.show()