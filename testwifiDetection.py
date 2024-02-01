import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import normalize,wifi6_detect

wifi6_data = np.load("wifi_data.npy")


Nwifi6_buffer = 4000
Nseq_wifi6 = 200
Nwifi6 = 2000
Nrx_buffer = 2000

wifi6_buffer = np.zeros(Nwifi6_buffer, dtype=np.complex64)
wifi6_result = np.zeros((Nseq_wifi6, 2))

# Function to update the animation
def update(frame):
    global wifi6_buffer  # Declare wifi6_buffer as a global variable
    plt.clf()  # Clear the previous frame
    wifi6_buffer = np.hstack((wifi6_buffer[Nrx_buffer:], wifi6_data[frame]))
    wifi6_buffer_norm = normalize.normalize(wifi6_buffer)
    # Plot your data here; modify this based on your data structure
    plt.plot(wifi6_buffer_norm.real)
    plt.title(f'Frame {frame}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

animatedPlot = False

if animatedPlot:
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(wifi6_data), repeat=False, interval = 50)

    # Display the animation
    plt.show()

seq_wifi6 = np.arange(1,201) % 2
for idx in range(5):
    wifi6_buffer = np.hstack((wifi6_buffer[Nrx_buffer:], wifi6_data[idx+25]))
    wifi6_buffer_norm = normalize.normalize(wifi6_buffer)
    wifi6_result[:-1,:] = wifi6_result[1:,:]
    wifi6_result[-1, 0] = wifi6_detect.wifi6_detect(wifi6_buffer_norm[:Nwifi6])
    wifi6_result[-1, 1] = wifi6_detect.wifi6_detect(wifi6_buffer_norm[Nwifi6//2:Nwifi6+Nwifi6//2])
print('Wifi6_Thread finished')

corr_wifi6 = np.correlate(wifi6_result[:,0], seq_wifi6, "full")
corr_wifi6 = np.vstack((corr_wifi6, np.correlate(wifi6_result[:,1],seq_wifi6, "full")))
wifi6_sel = 0 if (np.max(corr_wifi6[0]) > np.max(corr_wifi6[1])) else 1
wifi6_offset = np.argmax(corr_wifi6[wifi6_sel])-Nseq_wifi6+1
wifi6_valid = wifi6_result[wifi6_offset:wifi6_offset+Nseq_wifi6,wifi6_sel]