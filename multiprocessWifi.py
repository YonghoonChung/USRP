import  numpy as np
import matplotlib.pyplot as plt
import uhd,threading, time
import genWifi6,wifi6_detect
from scipy.io import loadmat
import scipy.signal as sg
from multiprocessing import Process

plt.close()
## Basic Info
Fs = 20e6  # 20MHz sampling rate
rf_freq = 2.62e9  # 2.62GHz carrier frequency
tx_channel = 0
txGain =40  # dB
rx_channel = 0
rxGain = 20  # dB

Ts_dvbt2 = 7/64*1e-6        # DVB-T2 with sampling time = 7/64us (Table 65)
Fs_dvbt2 = 1/Ts_dvbt2
num_signals = 10000
Ndvbt2_buffer = 20000
Ndvbt2 = 10000
Nrx_buffer = 2000
dvbCount = 0
Nseq_dvbt2 = 20

Nwifi6_buffer = 4000
wifiCount = 0
Nseq_wifi = 2000
Nwifi6 = 2000

## Tx Normalization
bits = np.int32(np.round(np.random.random(48*4*12)))
wifi6_data = genWifi6.genWifi6(bits)
real_data = np.real(wifi6_data)
imag_data = np.imag(wifi6_data)
max_value = np.max([np.abs(real_data).max(), np.abs(imag_data).max()])
real_data = 1*(real_data -real_data.mean())/max_value
imag_data = 1*(imag_data -imag_data.mean())/max_value
xs = real_data+1j*imag_data

## xs DVBT2 Detection
start = time.time_ns()
corr = wifi6_detect.wifi6_detect(xs)
end = time.time_ns()
print(corr)
print(f'{(end-start)/1e3:.2f}us')


spp = 2000

debug = False
file_save = False
tx_plot= False
rx_plot = True
tx_flag=True

xs = np.hstack([xs,np.zeros(len(xs), dtype = np.complex64)])
Ns=len(xs)
n1 = np.arange(Ns)
n2 = np.arange(2000)
ts1 = n1/Fs
ts2 = n2/Fs

## Tx data plot
if tx_plot:
    for idx in range(2):
        subplot = 311+idx
        plt.subplot(subplot)
        plt.title(f'')
        plt.plot(xs[idx*2000:(idx+1)*2000].real)
    plt.subplot(313)
    plt.plot(xs.real)
    plt.tight_layout()
    plt.show()
## tx Thread
def tx_func(tx_streamer):
    global xs, tx_flag
    tx_cnt=0
    # time.sleep(0.003)
    while tx_cnt<40000:
        idx=tx_cnt%2
        num_tx_samps = tx_streamer.send(xs[idx*2000:(idx+1)*2000],tx_metadata)
        tx_cnt += 1
    tx_flag=False


## Wifi6 Detection Thread
wifi6_buffer = np.zeros(Nwifi6_buffer, dtype=np.complex64)
wifi6_result = np.zeros((Nseq_wifi*2, 2))
def wifi6_func():
    global wifi6_buffer, rx_cnt, wifiCount
    flag = False
    totalTime = 0
    while rx_cnt< 200:
        wifiCount += 1
        start = time.time_ns()
        wifi6_result[:-1,:] = wifi6_result[1:,:]
        wifi6_result[-1, 0] = wifi6_detect.wifi6_detect(wifi6_buffer[:Nwifi6])
        wifi6_result[-1, 1] = wifi6_detect.wifi6_detect(wifi6_buffer[Nwifi6//2:Nwifi6+Nwifi6//2])
        end = time.time_ns()
        totalTime += (end-start)
    print('Wifi6_Thread finished')

    print(f'Wifi Count = {wifiCount}')
    print(f'Wifi Average Time  = {totalTime/wifiCount/1e3:.2f}')


## USRP Configuration
# dev = uhd.libpyuhd.types.device_addr(addr=192.168.40.1)
myusrp = uhd.usrp.MultiUSRP()
num_samps = Ns

myusrp.set_tx_antenna("TX/RX")
myusrp.set_tx_rate(Fs)
myusrp.set_tx_bandwidth(Fs/2)
myusrp.set_tx_freq(uhd.libpyuhd.types.tune_request(rf_freq), tx_channel)
myusrp.set_tx_gain(txGain)

myusrp.set_rx_antenna("RX2")
myusrp.set_rx_rate(Fs)
myusrp.set_rx_bandwidth(Fs/2)
myusrp.set_rx_freq(uhd.libpyuhd.types.tune_request(rf_freq), rx_channel)
myusrp.set_rx_gain(rxGain)

# Tx Streamer
txstream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
# txstream_args.args = f'spp={Ns}'  # samples per packet
txstream_args.args = f'spp={spp}'  # samples per packet
tx_streamer = myusrp.get_tx_stream(txstream_args)

# Set up the stream and receive bufferx
rxstream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
# rxstream_args.args = f'spp={Ns}'  # samples per packet
rxstream_args.args = f'spp={spp}'  # samples per packet
rxstream_args.channels = [rx_channel]
rx_streamer = myusrp.get_rx_stream(rxstream_args)
recv_buffer = np.zeros((1, 2000), dtype=np.complex64)

# Start Stream
tx_metadata = uhd.types.TXMetadata()
rx_metadata = uhd.types.RXMetadata()
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
rx_streamer.issue_stream_cmd(stream_cmd)
print("Finished Configuration")

rx_flag = True
usrp_configured = True

## Threads Run
txT = Process(target=tx_func, args = (tx_streamer,))
wifiT = threading.Thread(target = wifi6_func)

txT.start()
rx_cnt = 0


# time.sleep(1)
wifiT.start()
err_log=[]
print('Receiving Start')
while rx_cnt<200:
    recv_buffer = np.zeros((2000), dtype=np.complex64)
    nrsamp = rx_streamer.recv(recv_buffer, rx_metadata)
    wifi6_buffer = np.hstack((wifi6_buffer[Nrx_buffer:], recv_buffer))
    
    err_log.append([rx_cnt,nrsamp,recv_buffer])
    rx_cnt += 1
    if not tx_flag:
            print('TxSent')
    if rx_metadata.error_code != uhd.types.RXMetadataErrorCode.none:
        print(rx_metadata.strerror())
            

txT.join()
wifiT.join()

## Finishing Threads
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
rx_streamer.issue_stream_cmd(stream_cmd)

if not debug:
    del tx_streamer, rx_streamer
    del myusrp
if rx_plot:


    tmp = np.zeros((200,2000), dtype = np.complex64)
    for count in range(200):
        tmp[count] = err_log[count][2]

    fig1 = False
    fig2 = True
    idx = 5
    #######################3
    if fig1:
        pass
    #########################
    if fig2:
        plt.figure()
        idx = 0
        plt.subplot(311)
        plt.plot(xs.real)
        plt.subplot(312)
        plt.plot(np.hstack([tmp[idx].real,tmp[idx+1].real,tmp[idx+2].real,tmp[idx+3].real,tmp[idx+4].real]))
        plt.subplot(313)
        plt.plot(np.hstack([tmp[idx+5].real,tmp[idx+6].real,tmp[idx+7].real,tmp[idx+8].real,tmp[idx+9].real]))

        plt.tight_layout()
        plt.show()
    ##########################
if file_save:
    file_path = "err_log"
    np.save(file_path, tmp)

print('finished')

