import  numpy as np
import matplotlib.pyplot as plt
import uhd
import threading, time

plt.close()
Nperiod=1000
sig_freq=1e6
Fs = 250e6/12  # 20MHz sampling rate

n=np.arange(Nperiod/sig_freq * Fs)
Ns=len(n)
ts = n/Fs

debug = False
tx_flag=True
xs = np.exp(2j*np.pi*sig_freq*ts).astype('complex64')

test = True
test2 = False
if test2:
    plt.figure()
    plt.plot(xs.real, marker='o')
    plt.show()
##

def tx_func():
    global xs, tx_flag
    tx_cnt=0
    while tx_cnt<40:
        num_tx_samps = tx_streamer.send(xs,tx_metadata)
        tx_cnt += 1
    tx_flag=False
    print('TxSent')

rf_freq = 2.62e9  # 2.62GHz carrier frequency
tx_channel = 0
txGain = 20  # dB
rx_channel = 0
rxGain = 30  # dB


# dev = uhd.libpyuhd.types.device_addr("addr=192.168.10.2")
myusrp = uhd.usrp.MultiUSRP("addr=192.168.10.2, master_clock_rate=250e6")
num_samps = Ns
# myusrp.set
myusrp.set_tx_antenna("TX/RX0")
myusrp.set_tx_rate(Fs)
# myusrp.set_tx_bandwidth(Fs)
myusrp.set_tx_freq(uhd.libpyuhd.types.tune_request(rf_freq), tx_channel)
myusrp.set_tx_gain(txGain)

myusrp.set_rx_antenna("RX1")
myusrp.set_rx_rate(Fs)
# myusrp.set_rx_bandwidth(Fs)
myusrp.set_rx_freq(uhd.libpyuhd.types.tune_request(rf_freq), rx_channel)
myusrp.set_rx_gain(rxGain)
##
# Tx Streamer
txstream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
# txstream_args.args = f'spp={Ns}'  # samples per packet
# txstream_args.args = f'spp={2000}'  # samples per packet
txstream_args.args
tx_streamer = myusrp.get_tx_stream(txstream_args)

# Set up the stream and receive buffer
rxstream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
# rxstream_args.args = f'spp={Ns}'  # samples per packet
rxstream_args.args = f'spp={2000}'  # samples per packet
rxstream_args.channels = [rx_channel]
rx_streamer = myusrp.get_rx_stream(rxstream_args)


# Start Stream
tx_metadata = uhd.types.TXMetadata()
rx_metadata = uhd.types.RXMetadata()
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
rx_streamer.issue_stream_cmd(stream_cmd)
print("Finished Configuration")

rx_flag = True
usrp_configured = True


txT = threading.Thread(target=tx_func)
txT.start()
time.sleep(0.0001)
rx_cnt = 0
err_log=[]
while rx_cnt<100:
    recv_buffer = np.zeros((1, Ns), dtype=np.complex64)
    nrsamp = rx_streamer.recv(recv_buffer, rx_metadata)
    err_log.append([rx_cnt,nrsamp,recv_buffer])
    rx_cnt += 1


while tx_flag:
    pass

print('done')

txT.join()

if not debug:
    del tx_streamer, rx_streamer
    del myusrp
print('finished')

    

if test:    
    tmp = np.zeros((len(err_log),len(err_log[0][2][0])), dtype = np.complex64)
    for count in range(len(err_log)):
        tmp[count] = err_log[count][2][0]
    plt.figure()
    # plt.plot(err_log[88][2][0].real)
    plt.plot(tmp.flatten().real)
    plt.show()
# startidx=2000
# N=4096
# ys=err_log[99][2][0][startidx:startidx+N]
# YS=np.fft.fft(ys)
# 
# N2=int(N/2)
# 
# fs=Fs
# freq=np.arange(0,N2)*fs/N
# fig,ax = plt.subplots(2)
# # ax[0].plot(samps.real[0][1000:2024],label='real')
# # ax[0].plot(samps.imag[0][1000:2024],label='imag')
# ax[0].plot(ts,err_log[99][2][0].real,ts,err_log[99][2][0].imag)
# ax[0].set_xlabel('sample')
# ax[0].ticklabel_format(axis='both',style='sci',scilimits=(0,0))
# ax[0].set_title(f'{txGain}')
# # ax[0].legend()
# 
# ax[1].plot(freq,20*np.log10(np.abs(YS[0:N2])))
# ax[1].set_xscale('log')
# #ax[1].axis([0,1e5,-80,0])
# ax[1].axes.grid(axis='both')
# 
# ax[1].set_label('Freq(Hz)')
# 
# Sp=20*np.log10(max(abs(YS)))
# fidx=np.where(freq>2*sig_freq)[0][0]
# No=20*np.log10(np.average(abs(YS[fidx:len(freq)])))
# 
# RBW=Fs/N
# 
# SNR=Sp-No-10*np.log10(2*sig_freq/RBW)
# print('SNR=%.1f (dB)'%(SNR))
# plt.show()