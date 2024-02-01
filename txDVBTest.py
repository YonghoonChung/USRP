import  numpy as np
import matplotlib.pyplot as plt
import uhd,threading, time
import dvbt2_detect,dvbt2_detectV2
from scipy.io import loadmat
import scipy.signal as sg

plt.close()
## Basic Info
Fs = 50e6  # 20MHz sampling rate
rf_freq = 2.62e9  # 2.62GHz carrier frequency
tx_channel = 0
txGain = 25  # dB
rx_channel = 0
rxGain = 10  # dB

Ts_dvbt2 = 7/64*1e-6        # DVB-T2 with sampling time = 7/64us (Table 65)
Fs_dvbt2 = 1/Ts_dvbt2
num_signals = 10000
Ndvbt2_buffer = 10000
Nrx_buffer = 2000
dvbCount = 0

## Tx P1-Preamble Normalization
dvbt2_org = loadmat("./data/generatedP1.mat")['data'].flatten()
dvbt2_20MHz = sg.resample(dvbt2_org, int(len(dvbt2_org)*35/16))
real_data = np.real(dvbt2_20MHz)
imag_data = np.imag(dvbt2_20MHz)
max_value = np.max([np.abs(real_data).max(), np.abs(imag_data).max()])
real_data = (real_data -real_data.mean())/max_value
imag_data = (imag_data -imag_data.mean())/max_value

dvbt2_20MHz_norm = real_data+1j*imag_data

##Tx Data and Zero pad
xs = np.zeros(num_signals, np.complex64)
real_part = np.random.uniform(-1, 1, 4480)
imaginary_part = np.random.uniform(-1, 1, 4480)
data_signal = real_part + 1j * imaginary_part



## xs formation
xs[:len(dvbt2_20MHz_norm)] = dvbt2_20MHz_norm
xs[len(dvbt2_20MHz_norm):len(dvbt2_20MHz_norm)+len(data_signal)] = data_signal

## xs DVBT2 Detection
dvbt2_9M143Hz_buffer = sg.resample(xs, int(len(xs)*16/35))
corr1 = dvbt2_detect.dvbt2_detect(dvbt2_9M143Hz_buffer)
print(corr1)
corr2 = dvbt2_detectV2.dvbt2_detectV2(xs)
print(corr2)


spp = 2000

debug = False
file_save = False
tx_flag=True

Ns=len(xs)
n1 = np.arange(Ns)
n2 = np.arange(2000)
ts1 = n1/Fs
ts2 = n2/Fs

## tx Thread

def tx_func():
    global xs, tx_flag
    tx_cnt=0
    print('transmission began')
    while tx_cnt<1000000000000:
        idx=tx_cnt%5
        num_tx_samps = tx_streamer.send(xs[idx*2000:(idx+1)*2000],tx_metadata)
        tx_cnt += 1
    tx_flag=False
    print('TxSent')




## USRP Configuration
dev = uhd.libpyuhd.types.device_addr("type=x300,addr=192.168.30.2")
myusrp = uhd.usrp.MultiUSRP(dev)
num_samps = Ns

myusrp.set_tx_antenna("TX/RX")
myusrp.set_tx_rate(Fs)
myusrp.set_tx_bandwidth(Fs/2)
myusrp.set_tx_freq(uhd.libpyuhd.types.tune_request(rf_freq), tx_channel)
myusrp.set_tx_gain(txGain)

# Tx Streamer
txstream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
# txstream_args.args = f'spp={Ns}'  # samples per packet
txstream_args.args = f'spp={spp}'  # samples per packet
tx_streamer = myusrp.get_tx_stream(txstream_args)


# Start Stream
tx_metadata = uhd.types.TXMetadata()
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
print("Finished Configuration")

usrp_configured = True

## Threads Run
txT = threading.Thread(target=tx_func)
txT.start()
time.sleep(0.0001)



## Finishing Threads
while tx_flag:
    pass
txT.join()
print('Tx and Rx Threads are done')

if not debug:
    del tx_streamer
    del myusrp
else:
    print(Ns)
    plt.figure()
    plt.subplot(411)
    plt.plot(xs.real)
    plt.subplot(412)
    plt.plot(xs.imag)
    plt.subplot(413)
    plt.plot(err_log[99][2][0].real)
    plt.subplot(414)
    plt.plot(err_log[99][2][0].imag)

    # plt.plot(err_log[99][2][0].imag)
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

