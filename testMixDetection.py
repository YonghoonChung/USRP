import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import normalize,dvbt2_detectV2,wifi6_detect

# Load wifi data
dvb_data = np.load("dvb_data.npy")
wifi6_data = np.load("wifi_data.npy")

data = wifi6_data

Nwifi6_buffer = 4000
Nseq_wifi = 200
Nwifi6 = 2000
Nrx_buffer = 2000

wifi6_buffer = np.zeros(Nwifi6_buffer, dtype=np.complex64)
wifi6_result = np.zeros((Nseq_wifi, 2))

Ts_dvbt2 = 7/64*1e-6        # DVB-T2 with sampling time = 7/64us (Table 65)
Fs_dvbt2 = 1/Ts_dvbt2
num_signals = 10000
Ndvbt2_buffer = 20000
Ndvbt2 = 10000
Nrx_buffer = 2000
dvbCount = 0
Nseq_dvbt2 = 20

dvbt2_buffer = np.zeros(Ndvbt2_buffer, dtype=np.complex64)
dvbt2_result = np.zeros((Nseq_dvbt2*2, 2))

# Function to update the animation
def update(frame):
    global dvbt2_buffer  # Declare wifi6_buffer as a global variable
    plt.clf()  # Clear the previous frame
    dvbt2_buffer = np.hstack((dvbt2_buffer[Nrx_buffer:], data[frame]))
    dvbt2_buffer_norm = normalize.normalize(dvbt2_buffer)
    # Plot your data here; modify this based on your data structure
    plt.plot(dvbt2_buffer_norm.real)
    plt.title(f'Frame {frame}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

animatedPlot = False

if animatedPlot:
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(data), repeat=False, interval = 50)

    # Display the animation
    plt.show()
count = 0
avg1 = 0
avg2 = 0
for idx in range(len(data)):
    dvbt2_buffer = np.hstack((dvbt2_buffer[Nrx_buffer:], data[idx]))
    wifi6_buffer_norm = normalize.normalize(dvbt2_buffer[-4000:])
    avg1 += dvbt2_buffer[-4000:].mean()
    avg2 += dvbt2_buffer[-4000:].std()
    wifi6_result[:-1,:] = wifi6_result[1:,:]
    wifi6_result[-1, 0] = wifi6_detect.wifi6_detect(wifi6_buffer_norm[:Nwifi6])
    wifi6_result[-1, 1] = wifi6_detect.wifi6_detect(wifi6_buffer_norm[Nwifi6//2:Nwifi6+Nwifi6//2])
    if idx % 5 == 0:
        dvbt2_buffer_norm = normalize.normalize(dvbt2_buffer)
        dvbt2_result[:-1,:] = dvbt2_result[1:,:]
        dvbt2_result[-1, 0] = dvbt2_detectV2.dvbt2_detectV2(dvbt2_buffer_norm[:Ndvbt2])
        dvbt2_result[-1, 1] = dvbt2_detectV2.dvbt2_detectV2(dvbt2_buffer_norm[Ndvbt2//2:Ndvbt2+Ndvbt2//2])
        count +=1
# plt.figure()
# plt.plot(dvbt2_result[5:,0])
# plt.show()`
dvbArray = np.arange(5,40)
wifiArray = np.arange(25,200)

print('finished')

plt.figure()
plt.subplot(211)
plt.title(f'DVBT2 Detection Correlation')
plt.plot(dvbArray,dvbt2_result[5:])
plt.subplot(212)
plt.title(f'Wifi6 Detection Correlation')
plt.plot(wifiArray, wifi6_result[25:])
plt.tight_layout()
plt.show()
