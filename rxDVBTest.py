import  numpy as np
import matplotlib.pyplot as plt
import uhd,threading, time
import dvbt2_detect,dvbt2_detectV2
from scipy.io import loadmat
import scipy.signal as sg

plt.close('all')
## Basic Info
Fs = 50e6  # 20MHz sampling rate
rf_freq = 2.62e9  # 2.62GHz carrier frequency
tx_channel = 0
txGain = 25  # dB
rx_channel = 0
rxGain = 10  # dB
spp = 2000

Ts_dvbt2 = 7/64*1e-6        # DVB-T2 with sampling time = 7/64us (Table 65)
Fs_dvbt2 = 1/Ts_dvbt2
num_signals = 10000
Ndvbt2_buffer = 10000
Nrx_buffer = 2000
dvbCount = 0


## xs DVBT2 Detection
debug = False
showPlots = True
file_save = False

n1 = np.arange(10000)
n2 = np.arange(2000)
ts1 = n1/Fs
ts2 = n2/Fs

## DVBT2 Detection Thread
dvbt2_buffer = np.zeros(Ndvbt2_buffer, dtype=np.complex64)
dvbEvent = threading.Event()
def DVBT2_func():
    global dvbt2_buffer, rx_cnt, dvbEvent,dvbCount
    flag = False

    while rx_cnt< 100:
        dvbEvent.wait()
        dvbCount += 1
        dvbEvent.clear()

    print('DVBT2_Thread finished')
    # print(f'DVBT2 Count = {dvbCount}')


## USRP Configuration
dev = uhd.libpyuhd.types.device_addr("type=x300,addr=192.168.40.2")
myusrp = uhd.usrp.MultiUSRP(dev)

myusrp.set_rx_antenna("RX2")
myusrp.set_rx_rate(Fs)
myusrp.set_rx_bandwidth(Fs/2)
myusrp.set_rx_freq(uhd.libpyuhd.types.tune_request(rf_freq), rx_channel)
myusrp.set_rx_gain(rxGain)

# Set up the stream and receive bufferx
rxstream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
# rxstream_args.args = f'spp={Ns}'  # samples per packet
rxstream_args.args = f'spp={spp}'  # samples per packet
rxstream_args.channels = [rx_channel]
rx_streamer = myusrp.get_rx_stream(rxstream_args)
recv_buffer = np.zeros((1, 2000), dtype=np.complex64)

# Start Stream
rx_metadata = uhd.types.RXMetadata()
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
rx_streamer.issue_stream_cmd(stream_cmd)
print("Finished Configuration")

rx_flag = True
usrp_configured = True

## Threads Run
time.sleep(0.0001)
rx_cnt = 0
err_log=[]
while rx_cnt<100:
    nrsamp = rx_streamer.recv(recv_buffer, rx_metadata)
    dvbt2_buffer = np.hstack((dvbt2_buffer[Nrx_buffer:], recv_buffer[0].copy()))
    err_log.append([rx_cnt,nrsamp,recv_buffer.copy()])
    rx_cnt += 1
    if rx_cnt % 5 == 0 and rx_cnt>0:
        dvbEvent.set()
## Finishing Threads
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
rx_streamer.issue_stream_cmd(stream_cmd)

if not debug:
    del rx_streamer
    del myusrp
if showPlots:
    plt.figure()
    plt.subplot(211)
    plt.plot(err_log[99][2][0].real)
    plt.subplot(212)
    plt.plot(err_log[99][2][0].imag)
    plt.show()

    count = 15
    plt.figure()
    plt.subplot(511)
    plt.plot(err_log[count][2][0].real)
    plt.subplot(512)
    plt.plot(err_log[count+1][2][0].real)
    plt.subplot(513)
    plt.plot(err_log[count+2][2][0].real)
    plt.subplot(514)
    plt.plot(err_log[count+3][2][0].real)
    plt.subplot(515)
    plt.plot(err_log[count+4][2][0].real)
    plt.show()

if file_save:
    file_path = "err_log"
    tmp = np.zeros((100,2000), dtype = np.complex64)
    for count in range(100):
        for idx in range(2000):
            tmp[count,idx] = err_log[count][2][0][idx]
    np.save(file_path, tmp)

print('finished')

