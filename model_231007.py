'''
목표: 10번 receive를 진행하는데, 정현파만 전송한다. 
'''
import sys
import numpy as np
import uhd
import threading
import dvbt2_detect,dvbt2_detectV2,genDvbt2, genWifi6,wifi6_detect
import matplotlib.pyplot as plt
import scipy.signal as sg
import time, csv, os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import uic
import datetime

# Import the GUI file
from gui import MyWindow, UI_class, GraphWindow

# basic info
Nwifi6 = 2000
Nwifi6_buffer = Nwifi6*2
Ndvbt2 = 10000
Ndvbt2_9M143Hz = 4096
Ndvbt2_buffer = 20000
Ndvbt2_9M143Hz_buffer = 9120    # 9975*2*16/35
# rx buffer (USRP buffer--)
Nrx_buffer = Nwifi6         # rx buffer is based on wifi6
logx = []
spp = 2000  # based on wifi6 spec
Nseq_wifi6 = 5000
Nseq_dvbt2 = 1000

wifiDetectionTime = 0
wifi6_buffer = np.zeros(Nwifi6_buffer, dtype=np.complex64)
dvbt2_buffer = np.zeros(Ndvbt2_buffer, dtype=np.complex64)
rxbuffer = np.zeros(spp, dtype=np.complex64)
wifi6_result = np.zeros((Nseq_wifi6*2, 2))
dvbt2_result = np.zeros((Nseq_dvbt2*2, 2))


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

# Create an instance of the GUI application
class GenTx(MyWindow):
    def __init__(self):
        global loopidx, rxbuffer
        # Call the super constructor
        super().__init__()
        randseq = np.load("randomSeq.npz")
        self.seq_wifi6 = randseq['seq_wifi6']
        self.seq_dvbt2 = randseq['seq_dvbt2']
        self.rx_flag = False
        self.usrp_configured = False
        self.usrp_flag = True
        self.loopidx = None
        self.rxbuffer = None
        self.wifiCount = 0
        self.dvbCount = 0
        self.txCount = 0
        self.dvbEvent = threading.Event()
        self.debugflag = True
        
    

    def tx_func(tx_streamer,tx_metadata):
        global xs, tx_flag, signal_dvbt2
        tx_cnt=0
        n = np.arange(1000)
        while tx_cnt<40000:
                idx=tx_cnt%10
                num_tx_samps = tx_streamer.send(xs[idx*2000:(idx+1)*2000],tx_metadata)
                tx_cnt += 1
                if tx_cnt % 10000 == 0:
                    print(tx_cnt)
        tx_flag=False
        print('Tx Thread finished')
        
    def rx_func(self):
        global logx, spp, wifi6_buffer,wifi6_result,dvbt2_buffer, Nwifi6, wifiDetectionTime
        recv_buffer = np.zeros((1, 2000), dtype=np.complex64)
        rxbuffer = np.zeros(spp, dtype=np.complex64)
        # rxbuffer2 = np.zeros(spp, dtype=np.complex64)
        self.rx_count = 0
        self.testLabel.setText("Receiving Data")
        
        current_time = datetime.datetime.now().strftime('%y%m%d%H%M%S')
        filename = f'./logFiles/log{current_time}.csv'

        debugSaves = []

        while self.rx_count < 200:
            recv_buffer = np.zeros((2000), dtype=np.complex64)
            nrsamp = self.rx_streamer.recv(recv_buffer, self.rx_metadata)
            dvbt2_buffer = np.hstack((dvbt2_buffer[Nrx_buffer:], recv_buffer))
            self.rx_count += 1

            logx.append([self.rx_count, nrsamp, recv_buffer])
            # time2 = time.time_ns() 
            # wifiDetectionTime += time2 - time1
            debugSaves.append([recv_buffer])

            print(len(debugSaves))

        print(f'total time from wifi : {wifiDetectionTime/1e3/self.wifiCount} us')
        with open(filename, 'w',newline='') as f: 
            write = csv.writer(f) 
            write.writerow(['Count', 'Nrsamp', 'Metadata'])
            for item in logx:
                write.writerow(item)
        print(len(debugSaves))
        np.save('debugSaves.npy', np.array(debugSaves))
        self.testLabel.setText("Receiving finished ")
        self.debugflag=False
        plt.figure()
        plt.plot(recv_buffer.real)
        plt.show()
            
    def DVBT2Dectection(self):
        global dvbt2_result,dvbt2_buffer
        
        while self.rx_flag:
            self.dvbEvent.wait()
            if self.rx_flag == False:
                break
            dvbt2_result[:-1,:] = dvbt2_result[1:,:]
            dvbt2_result[-1, 0] = dvbt2_detectV2.dvbt2_detectV2(dvbt2_buffer[:Ndvbt2])
            dvbt2_result[-1, 1] = dvbt2_detectV2.dvbt2_detectV2(dvbt2_buffer[Ndvbt2//2:Ndvbt2+Ndvbt2//2])
            self.dvbCount += 1
            self.dvbEvent.clear()
        print(self.dvbCount)

    def usrpConfig(self):
        global spp
        # global recv_buffer, rx_metadata
        # usrp = uhd.usrp.MultiUSRP()

        Fs = 20e6  # 20MHz sampling rate
        rf_freq = 2.62e9  # 2.62GHz carrier frequency
        tx_channel = 0
        txGain = 0  # dB
        rx_channel = 0
        rxGain = 30  # dB
        dev = uhd.libpyuhd.types.device_addr()
        self.myusrp = uhd.usrp.MultiUSRP(dev)
        self.num_samps = 2000

        # generate tx signal
        # data = np.load("tx_data_DVB.npz")
        # self.tx = data["tx"]
        # self.tx = np.hstack((np.zeros(len(self.tx)*2, dtype=np.complex64), self.tx, (np.zeros(len(self.tx), dtype=np.complex64))))  # zero padding tx

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
        
        self.myusrp.set_tx_antenna("TX/RX")
        self.myusrp.set_tx_rate(Fs)
        self.myusrp.set_tx_bandwidth(Fs/2)
        self.myusrp.set_tx_freq(uhd.libpyuhd.types.tune_request(rf_freq), tx_channel)
        self.myusrp.set_tx_gain(txGain)
        
        self.myusrp.set_rx_antenna("RX2")
        self.myusrp.set_rx_rate(Fs)
        self.myusrp.set_rx_bandwidth(Fs/2)
        self.myusrp.set_rx_freq(uhd.libpyuhd.types.tune_request(rf_freq), rx_channel)
        self.myusrp.set_rx_gain(rxGain)

        # Tx Streamer
        txstream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
        txstream_args.args = f'spp={spp}'  # samples per packet
        self.tx_streamer = self.myusrp.get_tx_stream(txstream_args)

        # Set up the stream and receive buffer
        rxstream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
        rxstream_args.args = f'spp={spp}'  # samples per packet
        rxstream_args.channels = [rx_channel]
        self.rx_streamer = self.myusrp.get_rx_stream(rxstream_args)
        self.recv_buffer = np.zeros((1, 2000), dtype=np.complex64)

        # Start Stream
        self.tx_metadata = uhd.types.TXMetadata()
        self.rx_metadata = uhd.types.RXMetadata()
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        print("Finished Configuration")
        self.turnOnOffUSRPButton.setEnabled(True)
        self.rx_flag = True
        self.usrp_configured = True

    def turnOnOffUSRP(self):
        if self.usrp_flag == True:
            self.RxThread = threading.Thread(target=self.rx_func, daemon=True)
            # self.DVBT2Thread = threading.Thread(target=self.DVBT2Dectection)
            self.TxThread = threading.Thread(target=self.tx_func, args = (self.tx_streamer,self.tx_metadata))
            # self.TxThread.start()
            # time.sleep(3)
            self.RxThread.start()
            self.usrp_flag = False
            self.turnOnOffUSRPButton.setText('Receiver Turn Off')
            
        else :
            self.rx_flag = False
            self.dvbEvent.set()
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            self.rx_streamer.issue_stream_cmd(stream_cmd)
            # self.TxThread.join()  # Wait for the TxThread to finish
            self.RxThread.join()  # Wait for the RxThread to finish
            # self.DVBT2Thread.join()
            self.RxThread = None
            # self.DVBT2Thread = None
            self.usrp_flag = True
            self.turnOnOffUSRPButton.setDisabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GenTx()
    window.show()
    sys.exit(app.exec_())