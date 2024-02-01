import  numpy as np
import matplotlib.pyplot as plt
import uhd,threading, time
import dvbt2_detect,dvbt2_detectV2,genDvbt2, genWifi6,wifi6_detect, normalize
from scipy.io import loadmat
import scipy.signal as sg
from multiprocessing import Process

plt.close()
## Basic Info
Fs = 20e6  # 20MHz sampling rate
rf_freq = 2.62e9  # 2.62GHz carrier frequency
tx_channel = 0
txGain = 0  # dB
rx_channel = 0
rxGain = 30  # dB

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

spp = 2000
debug = False
file_save = False
tx_plot= False
rx_plot = True
tx_flag=True


signal_dvbt2 = False # True for DVB-T2, False for Wifi6
if signal_dvbt2:
## Tx P1-Preamble Normalization
    bits = np.int32(np.round(np.random.random(4*2**11)))
    dvbt2_20MHz = np.hstack((genDvbt2.genDvbt2(bits),np.zeros(10000-8960, dtype=np.complex64))) # 20MHz 10000 samples
    # dvbt2_org = loadmat("./data/generatedP1.mat")['data'].flatten()
    # dvbt2_20MHz = sg.resample(dvbt2_org, int(len(dvbt2_org)*35/16))
    real_data = np.real(dvbt2_20MHz)
    imag_data = np.imag(dvbt2_20MHz)
    max_value = np.max([np.abs(real_data).max(), np.abs(imag_data).max()])
    real_data = 0.5*(real_data -real_data.mean())/max_value
    imag_data = 0.5*(imag_data -imag_data.mean())/max_value
    xs = real_data+1j*imag_data
    test = xs
else:
    bits = np.int32(np.round(np.random.random(48*4*12)))
    wifi6_data = genWifi6.genWifi6(bits)
    real_data = np.real(wifi6_data)
    imag_data = np.imag(wifi6_data)
    max_value = np.max([np.abs(real_data).max(), np.abs(imag_data).max()])
    real_data = 1*(real_data -real_data.mean())/max_value
    imag_data = 1*(imag_data -imag_data.mean())/max_value
    xs = real_data+1j*imag_data
    test = np.hstack((xs, np.zeros(10000-len(xs), dtype=np.complex64)))

## xs DVBT2 Detection
dvbt2_9M143Hz_buffer = sg.resample(test, int(len(test)*16/35))
corr1 = dvbt2_detect.dvbt2_detect(dvbt2_9M143Hz_buffer)
print(corr1)
corr2 = dvbt2_detectV2.dvbt2_detectV2(test)
print(corr2)
# if signal_dvbt2:
#     for idx in range(5):
#         corr3 = wifi6_detect.wifi6_detect(normalize.normalize(test[idx*2000:(idx+1)*2000]))
#         print(corr3)
# else :
#     corr3 = wifi6_detect.wifi6_detect(normalize.normalize(test))
#     print(corr3)


xs = np.hstack([xs,np.zeros(len(xs), dtype = np.complex64)])
Ns=len(xs)
n1 = np.arange(Ns)
n2 = np.arange(2000)
ts1 = n1/Fs
ts2 = n2/Fs

## Tx data plot
if tx_plot:
    for idx in range(5):
        subplot = 611+idx
        plt.subplot(subplot)
        plt.title(f'')
        plt.plot(xs[idx*2000:(idx+1)*2000].real)
    plt.subplot(616)
    plt.plot(xs.real)
    plt.tight_layout()
    plt.show()
## tx Thread
def tx_func(tx_streamer):
    global xs, tx_flag, signal_dvbt2
    tx_cnt=0
    # time.sleep(0.003)
    if signal_dvbt2:
        mod = 10
    else:
        mod = 2
    while tx_cnt<40000:
        idx=tx_cnt%mod
        num_tx_samps = tx_streamer.send(xs[idx*2000:(idx+1)*2000],tx_metadata)
        tx_cnt += 1
        if tx_cnt % 10000 == 0:
            print(tx_cnt)
    tx_flag=False
    print('Tx Thread finished')


rf_freq = 2.62e9  # 2.62GHz carrier frequency
tx_channel = 0
txGain = 0  # dB
rx_channel = 0
rxGain = 20  # dB

## DVBT2 Detection Thread
dvbt2_buffer = np.zeros(Ndvbt2_buffer, dtype=np.complex64)
dvbEvent = threading.Event()
dvbt2_result = np.zeros((Nseq_dvbt2*2, 2))
def DVBT2_func():
    global dvbt2_buffer, rx_cnt, dvbEvent,dvbCount,dvbt2_result
    flag = False
    totalTime1 = totalTime2 = totalTime=  0
    while rx_cnt< 200:
        # dvbEvent.wait()
        dvbCount += 1
        
        start = time.time_ns()
        dvbt2_result[:-1,:] = dvbt2_result[1:,:]
        dvbt2_result[-1, 0] = dvbt2_detectV2.dvbt2_detectV2((dvbt2_buffer[:Ndvbt2]))
        end1 = time.time_ns()
        dvbt2_result[-1, 1] = dvbt2_detectV2.dvbt2_detectV2((dvbt2_buffer[5000:15000]))
        end2 = time.time_ns()
        totalTime1 += end1-start
        totalTime2 += end2-end1
        totalTime += end2-start
        dvbEvent.clear()
        dvbEvent.wait()
    print('DVBT2_Thread finished')

    print(f'DVBT2 Count = {dvbCount}')
    print(f'DVBT2 Average Time1  = {totalTime1/dvbCount/1e3:.2f}us')
    print(f'DVBT2 Average Time2  = {totalTime2/dvbCount/1e3:.2f}us')
    print(f'DVBT2 Average Time  = {totalTime/dvbCount/1e3:.2f}us')
    
wifi6_buffer = np.zeros(Nwifi6_buffer, dtype=np.complex64)
wifi6_result = np.zeros((Nseq_wifi*2, 2))
wifiEvent = threading.Event()
def wifi6_func():
    global wifi6_buffer, rx_cnt, wifiCount
    flag = False
    totalTime = 0
    wifiIdx = -4000
    while rx_cnt< 200:
        wifiCount += 1
        start = time.time_ns()
        
        # wifi6_result[:-1,:] = wifi6_result[1:,:]
        # wifi6_result[-1, 0] = wifi6_detect.wifi6_detect(dvbt2_buffer[wifiIdx:Nwifi6+wifiIdx])
        
        # wifi6_result[-1, 1] = wifi6_detect.wifi6_detect(dvbt2_buffer[Nwifi6//2+wifiIdx:Nwifi6+Nwifi6//2+wifiIdx])
        
        wifiEvent.clear()
        wifiEvent.wait()
        end = time.time_ns()
        totalTime = end-start
    print('Wifi6_Thread finished')

    print(f'Wifi Count = {wifiCount}')
    print(f'Wifi Average Time  = {totalTime/1e3:.2f}')

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
# dvbT = threading.Thread(target = DVBT2_func)
wifiT = threading.Thread(target = wifi6_func)
txT.start()
rx_cnt = 0


# time.sleep(1)
# dvbT.start()
wifiT.start()
err_log=[]
print('Receiving Start')
while rx_cnt<200:
    recv_buffer = np.zeros((2000), dtype=np.complex64)
    nrsamp = rx_streamer.recv(recv_buffer, rx_metadata)
    dvbt2_buffer = np.hstack((dvbt2_buffer[Nrx_buffer:], recv_buffer))
    err_log.append([rx_cnt,nrsamp,recv_buffer])
    rx_cnt += 1
    wifiEvent.set()
    if rx_cnt % 5 == 0:
        dvbEvent.set()
    if not tx_flag:
        print('TxSent')
    if rx_metadata.error_code != uhd.types.RXMetadataErrorCode.none:
        print(rx_metadata.strerror())
            
print('Receiving Finished')
# print('****************************************************************')
# print(dvbCount)
# print(dvbT.is_alive())
# print('****************************************************************')
txT.join()
# dvbT.join()
wifiT.join()

## Finishing Threads
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
rx_streamer.issue_stream_cmd(stream_cmd)



if not debug:
    del tx_streamer, rx_streamer
    del myusrp
if rx_plot:
    # print(Ns)
    # plt.figure()
    # plt.subplot(411)
    # plt.plot(xs.real)
    # plt.subplot(412)
    # plt.plot(xs.imag)
    # plt.subplot(413)
    # plt.plot(err_log[99][2][0].real)
    # plt.subplot(414)
    # plt.plot(err_log[99][2][0].imag)
    # plt.show()

    # count = 15
    # plt.figure()
    # plt.subplot(511)
    # plt.plot(err_log[count][2][0].real)
    # plt.subplot(512)
    # plt.plot(err_log[count+1][2][0].real)
    # plt.subplot(513)
    # plt.plot(err_log[count+2][2][0].real)
    # plt.subplot(514)
    # plt.plot(err_log[count+3][2][0].real)
    # plt.subplot(515)
    # plt.plot(err_log[count+4][2][0].real)
    # plt.show()

    tmp = np.zeros((200,2000), dtype = np.complex64)
    for count in range(200):
        tmp[count] = err_log[count][2]
    tmp2 = tmp.reshape(40,10000)

    fig1 = False
    fig2 = True
    idx = 5
    #######################3
    if fig1:
        plt.figure()
        plt.subplot(311)
        plt.title('err_log')
        plt.plot(np.hstack([err_log[idx][2][0].real,err_log[idx+1][2][0].real,err_log[idx+2][2][0].real,err_log[idx+3][2][0].real,err_log[idx+4][2][0].real]))
        plt.subplot(312)
        plt.title(f'tmp')
        plt.plot(np.hstack([tmp[idx].real,tmp[idx+1].real,tmp[idx+2].real,tmp[idx+3].real,tmp[idx+4].real]))

        plt.subplot(313)
        plt.plot(tmp2[idx//5].real)
        plt.tight_layout()
        plt.show()
    #########################
    if fig2:
        plt.figure()
        idx = 2
        plt.subplot(311)
        plt.plot(xs.real)
        plt.subplot(312)
        plt.plot(np.hstack([np.abs(tmp2[idx]),tmp2[idx+1].real,tmp2[idx+2].real,tmp2[idx+3].real,tmp2[idx+4].real]))
        # plt.plot(np.hstack([tmp2[idx].real,tmp2[idx+1].real,tmp2[idx+2].real,tmp2[idx+3].real,tmp2[idx+4].real,tmp2[idx+5].real,tmp2[idx+6].real,tmp2[idx+7].real,tmp2[idx+8].real,tmp2[idx+9].real]))
        plt.subplot(313)
        plt.plot(np.hstack([tmp2[idx+5].real,tmp2[idx+6].real,tmp2[idx+7].real,tmp2[idx+8].real,tmp2[idx+9].real]))
        # plt.plot(np.hstack([tmp2[idx+10].real,tmp2[idx+11].real,tmp2[idx+12].real,tmp2[idx+13].real,tmp2[idx+14].real,tmp2[idx+15].real,tmp2[idx+16].real,tmp2[idx+17].real,tmp2[idx+18].real,tmp2[idx+19].real]))

        plt.tight_layout()
        plt.show()

    ##########################
if file_save:
    file_path = "err_log"
    np.save(file_path, tmp2)

print('finished')