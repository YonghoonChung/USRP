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
from multiprocessing import Process

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
        self.txCount = 0
        # self.rx_count = 0
        
        # self.wifiRectanglesRx = []
        # self.dvbt2RectanglesRx =[]
        
        self.dvbEvent = threading.Event()
        self.debugflag = True
        self.wifiEvent = threading.Event()
        self.rxEvent = threading.Event()
        
        self.finalIndexWifi = 0
        self.finalIndexDvbt2 = 0
        
        

        
        self.err_log=[]

    def usrpConfig(self):
        global spp

        Fs = 20e6  # 20MHz sampling rate
        rf_freq = 2.62e9  # 2.62GHz carrier frequency
        tx_channel = 0
        txGain = 0  # dB
        rx_channel = 0
        rxGain = 20  # dB
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

        self.rx_flag = True
        self.usrp_configured = True
        

        self.GenerationButton.setEnabled(True)   
    def Generation(self):
        self.clearGridLayout(self.gridLayout_5)
        self.clearGridLayout(self.gridLayout_8)
        wifiIdx = self.wifiCombo.currentText()
        dvbt2Idx = self.DVBCombo.currentText()
        self.finalIndexWifi = int(wifiIdx)
        self.finalIndexDvbt2 = int(dvbt2Idx)
        
        # self.TxLabel.setText(str(self.dataWifi[int(wifiIdx)]))
        print(f'wifi : {wifiIdx}, DVB-T2 : {dvbt2Idx}')
        self.wifiRectangles = self.seq_wifi6[int(wifiIdx)]
        self.dvbt2Rectangles = self.dataDVBT2[int(dvbt2Idx)]
        self.drawRecursiveRectangles()
        print('complete.')
        self.generationClicked = True
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
            self.xs = real_data+1j*imag_data
            test = self.xs
        else:
            bits = np.int32(np.round(np.random.random(48*4*12)))
            wifi6_data = genWifi6.genWifi6(bits)
            real_data = np.real(wifi6_data)
            imag_data = np.imag(wifi6_data)
            max_value = np.max([np.abs(real_data).max(), np.abs(imag_data).max()])
            real_data = 1*(real_data -real_data.mean())/max_value
            imag_data = 1*(imag_data -imag_data.mean())/max_value
            self.xs = real_data+1j*imag_data
            test = np.hstack((self.xs, np.zeros(10000-len(self.xs), dtype=np.complex64)))
        
        self.xs = np.hstack([self.xs,np.zeros(len(self.xs), dtype = np.complex64)])
        self.TxThread = Process(target=self.tx_func, args = (self.tx_streamer,self.tx_metadata,self.xs))
        self.RxThread = threading.Thread(target=self.rx_func)
        self.dvbT = threading.Thread(target = self.DVBT2_func)
        self.wifiT = threading.Thread(target = self.wifi6_func)
        self.turnOnOffUSRPButton.setEnabled(True)
        self.GenerationButton.setDisabled(True)
        
    def tx_func(self, tx_streamer,tx_metadata,xs):
        global tx_flag, signal_dvbt2
        tx_cnt=0
        # print('++++++++ Tx Thread Started ++++++++')
        # self.GenerationButton.setText('Tx Sending')
        # print('sending?????')
        while tx_cnt<1000:
            idx=tx_cnt%2
            num_tx_samps = tx_streamer.send(xs[2000*idx:2000*(idx+1)],tx_metadata)
            tx_cnt += 1
            if tx_cnt % 10000 == 0 and tx_cnt!= 40000:
                # self.TxProgressBar.setValue(int(tx_cnt/40000*100))
                pass
        tx_flag=False
        # self.TxProgressBar.setValue(100)
        # print('++++++++Tx Thread finished ++++++++')
        # self.GenerationButton.setText('Tx Finished')
    def rx_func(self):
        global dvbt2_buffer, rx_count 
        rx_count= 0
        self.rxEvent.clear()
        # print('++++++++Rx Thread Started ++++++++')
        start = time.time_ns()
        while rx_count < 7000:
            recv_buffer = np.zeros((2000), dtype=np.complex64)
            nrsamp = self.rx_streamer.recv(recv_buffer, self.rx_metadata)
            dvbt2_buffer = np.hstack((dvbt2_buffer[Nrx_buffer:], recv_buffer))
            self.err_log.append([rx_count,nrsamp,recv_buffer])
            rx_count += 1
            self.wifiEvent.set()
            if rx_count % 5 == 0:
                self.dvbEvent.set()
        end = time.time_ns()
        # if self.rx_count % 100 == 0:
        #     print(self.rx_count)
        # print('++++++++Rx Thread Finished ++++++++')
        self.rx_flag = False
        # print(f'Time taken to send {rx_count} samples = {(end-start)/1e6} ms')
        print('Receiver Turn Off')
        self.rxEvent.set()
        # tmp = np.zeros((5000,2000), dtype = np.complex64)
        # for count in range(len(tmp)):
        #     tmp[count] = self.err_log[count][2]
        # tmp2 = tmp.reshape(1000,10000)
        
        # idx = 0
        # plt.figure()
        # plt.plot(tmp.flatten())
        # plt.tight_layout()
        # plt.show()
    def wifi6_func(self):
        global wifi6_buffer, rx_count
        print('Wifi6_Thread started')
        totalTime = 0
        wifiIdx_det = -4000
        wifiDetectionCount = 0
        while rx_count< 6000:
            wifiDetectionCount += 1
            self.wifiEvent.clear()
            start = time.time_ns()
            
            wifi6_result[:-1,:] = wifi6_result[1:,:]
            wifi6_result[-1, 0] = wifi6_detect.wifi6_detect(dvbt2_buffer[wifiIdx_det:Nwifi6+wifiIdx_det])
            
            wifi6_result[-1, 1] = wifi6_detect.wifi6_detect(dvbt2_buffer[Nwifi6//2+wifiIdx_det:Nwifi6+Nwifi6//2+wifiIdx_det])
            
            self.wifiEvent.wait()
            end = time.time_ns()
            totalTime = end-start
        print('Wifi6_Thread finished')

        print(f'Wifi Count = {wifiDetectionCount}')
        print(f'Wifi Average Time  = {totalTime/1e3:.2f}')
    def DVBT2_func(self):
        global dvbt2_buffer, rx_count,dvbt2_result
        totalTime1 = totalTime2 = totalTime = 0
        dvbDetectionCount = 0
        while rx_count < 5000:
            # dvbEvent.wait()
            dvbDetectionCount += 1
            self.dvbEvent.clear()
            start = time.time_ns()
            dvbt2_result[:-1,:] = dvbt2_result[1:,:]
            dvbt2_result[-1, 0] = dvbt2_detectV2.dvbt2_detectV2((dvbt2_buffer[:Ndvbt2]))
            end1 = time.time_ns()
            dvbt2_result[-1, 1] = dvbt2_detectV2.dvbt2_detectV2((dvbt2_buffer[5000:15000]))
            end2 = time.time_ns()
            totalTime1 += end1-start
            totalTime2 += end2-end1
            totalTime += end2-start
            self.dvbEvent.wait()
        print('DVBT2_Thread finished')
        # print(f'DVBT2 Count = {dvbDetectionCount}')
        # print(f'DVBT2 Average Time1  = {totalTime1/dvbDetectionCount/1e3:.2f}us')
        # print(f'DVBT2 Average Time2  = {totalTime2/dvbDetectionCount/1e3:.2f}us')
        print(f'DVBT2 Average Time  = {totalTime/dvbDetectionCount/1e3:.2f}us, DVBT2 Count = {dvbDetectionCount}')
    def calculation(self):
        global dvbt2_result, wifi6_result
        wifi6_valid = np.zeros(Nwifi6)
        dvbt2_valid = np.zeros(Ndvbt2)
        
        wifi6_seq = self.data['wifi6_seq']
        wifi6_seq_ones = wifi6_seq==1
        wifi6_seq_zeros = wifi6_seq==0
        dvbt2_seq = self.data['dvbt2_seq']
        dvbt2_seq_ones = dvbt2_seq==1
        dvbt2_seq_zeros = dvbt2_seq==0
        
        corr_wifi6_full = np.correlate(wifi6_result[:,0], self.dataWifi[self.finalIndexWifi], "full")
        corr_wifi6_full = np.vstack((corr_wifi6_full, np.correlate(wifi6_result[:,1], self.dataWifi[self.finalIndexWifi], "full")))
        corr_dvbt2_full = np.correlate(dvbt2_result[:,0], self.dataDVBT2[self.finalIndexDvbt2], "full")
        corr_dvbt2_full = np.vstack((corr_dvbt2_full, np.correlate(dvbt2_result[:,1], self.dataDVBT2[self.finalIndexDvbt2], "full")))
        
        wifi6_sel = 0 if (np.max(corr_wifi6_full[0]) > np.max(corr_wifi6_full[1])) else 1
        dvbt2_sel = 0 if (np.max(corr_dvbt2_full[0]) > np.max(corr_dvbt2_full[1])) else 1
        wifi6_offset = np.argmax(corr_wifi6_full[wifi6_sel])-Nseq_wifi6+1
        dvbt2_offset = np.argmax(corr_dvbt2_full[dvbt2_sel])-Nseq_dvbt2+1
        print(f"wifi6_sel: {wifi6_sel}")
        print(f"dvbt2_sel: {dvbt2_sel}")
        print(f"wifi6_offset: {wifi6_offset}")
        print(f"dvbt2_offset: {dvbt2_offset}")
        #### debugging
        wifi6_offset = 0
        dvbt2_offset = 0
        #####
        wifi6_valid = wifi6_result[wifi6_offset:wifi6_offset+Nseq_wifi6,wifi6_sel]
        dvbt2_valid = dvbt2_result[dvbt2_offset:dvbt2_offset+Nseq_dvbt2,dvbt2_sel]

        self.thresholds = np.arange(0,1,0.02)
        thresholds = self.thresholds
        self.wifi6_md = np.zeros(len(thresholds))
        self.wifi6_fa = np.zeros(len(thresholds))
        self.dvbt2_md = np.zeros(len(thresholds))
        self.dvbt2_fa = np.zeros(len(thresholds))

        for thidx in range(len(thresholds)):
            self.wifi6_md[thidx] = np.count_nonzero(wifi6_valid[wifi6_seq_ones] < thresholds[thidx])
            self.wifi6_fa[thidx] = np.count_nonzero(wifi6_valid[wifi6_seq_zeros] > thresholds[thidx])
            self.dvbt2_md[thidx] = np.count_nonzero(dvbt2_valid[dvbt2_seq_ones] < thresholds[thidx])
            self.dvbt2_fa[thidx] = np.count_nonzero(dvbt2_valid[dvbt2_seq_zeros] > thresholds[thidx])
        
        self.thidx_wifi6_min = np.argmin(np.abs(self.wifi6_md-self.wifi6_fa))
        self.thidx_dvbt2_min = np.argmin(np.abs(self.dvbt2_md-self.dvbt2_fa))
        
        self.wifi6_validBi = np.int16(wifi6_valid>self.thresholds[self.thidx_wifi6_min])
        self.dvbt2_validBi = np.int16(dvbt2_valid>self.thresholds[self.thidx_dvbt2_min])
        
        thidx_wifi6 = np.argmin(self.wifi6_md+self.wifi6_fa); print(f"wifi6_md: {self.wifi6_md[thidx_wifi6]}, wifi6_fa: {self.wifi6_fa[thidx_wifi6]}")
        thidx_dvbt2 = np.argmin(self.dvbt2_md+self.dvbt2_fa); print(f"dvbt2_md: {self.dvbt2_md[thidx_dvbt2]}, dvbt2_fa: {self.dvbt2_fa[thidx_dvbt2]}")
        wifi6_md_final = self.wifi6_md[self.thidx_wifi6_min]
        wifi6_fa_final = self.wifi6_fa[self.thidx_wifi6_min]
        dvbt2_md_final = self.dvbt2_md[self.thidx_dvbt2_min]
        dvbt2_fa_final = self.dvbt2_fa[self.thidx_dvbt2_min]
        
        self.wifiM.setText(str(wifi6_md_final))
        self.wifiF.setText(str(wifi6_fa_final))
        self.dvbM.setText(str(dvbt2_md_final))
        self.dvbF.setText(str(dvbt2_fa_final))
        
    def turnOnOffUSRP(self):
        global dvbt2_buffer

        if self.usrp_flag == True:
            self.usrp_flag = False
            print('************ USRP ON ****************')
            self.RxThread.start()
            self.TxThread.start()
            self.dvbT.start()
            self.wifiT.start()
            self.rxEvent.wait()
            print('************ TxRx Finished ****************')
            self.turnOnOffUSRPButton.setText('Receiver Turn Off')
            ################################
            self.calculation()
            self.wifiRectanglesRx = self.wifi6_validBi
            self.dvbt2RectanglesRx = self.dvbt2_validBi
            # self.wifiRectanglesRx = np.int32(np.round(np.random.random((Nseq_wifi6))))
            # self.dvbt2RectanglesRx = np.int32(np.round(np.random.random((Nseq_dvbt2))))
            ################################
            self.wifiT.join()
            self.dvbT.join()
            self.TxThread.join()
            self.RxThread.join()
            self.flag = True
            self.time = 0
            print('************ Drawing Recursive Rectangles ****************')
            self.drawRecursiveRectanglesRx()
        else :
            self.rx_flag = False
            self.dvbEvent.set()
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            self.rx_streamer.issue_stream_cmd(stream_cmd)
            self.TxThread.join()  # Wait for the TxThread to finish
            # self.RxThread.join()  # Wait for the RxThread to finish
            # self.DVBT2Thread.join()
            # self.RxThread = None
            # self.DVBT2Thread = None
            del self.tx_streamer, self.rx_streamer
            del self.myusrp
            self.usrp_flag = True
            self.turnOnOffUSRPButton.setDisabled(True)
    def drawRecursiveRectangles(self):
        if len(self.wifiRectangles) < self.wifiCount and len(self.dvbt2Rectangles) < self.dvbCount:
            self.timer1.stop()
            return

        self.clearGridLayout(self.gridLayout_5)
        self.drawRectangles(self.wifiRectangles[:self.wifiCount], 'wifi', self.wifiCount)
        self.drawRectangles(self.dvbt2Rectangles[:self.dvbCount], 'DVBT2', self.dvbCount)

        self.wifiRectangles = self.wifiRectangles[self.wifiCount:]
        self.dvbt2Rectangles = self.dvbt2Rectangles[self.dvbCount:]

        self.timer1.start(10)  # Start the timer with a 2-second interval
    def drawRecursiveRectanglesRx(self):
        if len(self.wifiRectanglesRx) < self.wifiCount and len(self.dvbt2RectanglesRx) < self.dvbCount:
            self.timer.stop()
            return

        self.clearGridLayout(self.gridLayout_8)
        self.drawRectanglesRx(self.wifiRectanglesRx[:self.wifiCount], 'wifi', self.wifiCount)
        self.drawRectanglesRx(self.dvbt2RectanglesRx[:self.dvbCount], 'DVBT2', self.dvbCount)

        self.wifiRectanglesRx = self.wifiRectanglesRx[self.wifiCount:]
        self.dvbt2RectanglesRx = self.dvbt2RectanglesRx[self.dvbCount:]

        self.timer.start(10)  # Start the timer with a 2-second interval    
    def showGraph(self):
        if not self.generationClicked:
            return
        
        tmp = np.zeros((200,2000), dtype = np.complex64)
        for count in range(200):
            tmp[count] = self.err_log[count][2]
            np.save('test', tmp)
        tmp2 = tmp.reshape(20,20000)
        graphWindow = GraphWindow(self)

        ax1 = graphWindow.figure.add_subplot(311)
        ax1.plot(tmp2[0].real, label='Missed Detection')
        ax1.legend()
        ax1.set_title("wifi6 detection")
        ax2 = graphWindow.figure.add_subplot(312)
        ax2.plot(tmp2[1].real, label='Missed Detection')
        ax2.legend()
        ax2.set_title("wifi6 detection")
        ax3 = graphWindow.figure.add_subplot(313)
        ax3.plot(tmp2[2].real, label='Missed Detection')
        ax3.legend()
        ax3.set_title("wifi6 detection")
        
        # ax2 = graphWindow.figure.add_subplot(212)
        # ax2.plot(window.thresholds, window.dvbt2_md, '-o', label='Missed Detection')
        # ax2.plot(window.thresholds, window.dvbt2_fa, '-o', label='False Alarm')
        # ax2.vlines(window.thresholds[window.thidx_dvbt2], 0, 5000, label='least sum threshold', color='r')
        # ax2.vlines(window.thresholds[window.thidx_dvbt2_min], 0, 5000, label='least diff threshold', color='g')
        # ax2.legend()
        # ax2.set_xlabel('threshold')
        # ax2.set_title("dvbt2 detection")
        
        graphWindow.figure.tight_layout()
        graphWindow.canvas.draw()
        graphWindow.exec_()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GenTx()
    window.show()
    sys.exit(app.exec_())