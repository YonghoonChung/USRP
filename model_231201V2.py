'''
목표: 10번 receive를 진행하는데, 정현파만 전송한다. 
'''
import sys
import numpy as np
import uhd
import threading,random
import dvbt2_detect,dvbt2_detectV2,genDvbt2, genWifi6,wifi6_detect,genTx,wifi6_test
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
        
        self.wifi6_md = 0 
        self.wifi6_fa = 0 
        self.dvbt2_md = 0 
        self.dvbt2_fa = 0 
        self.temp1 = []
        self.temp2 = []
        self.temp3 = []
        self.temp4 = []
        
        wifi6_valid = np.zeros(Nwifi6)
        dvbt2_valid = np.zeros(Ndvbt2)
        
        self.prevNext =50
        self.changedThreashold = 0.5

        self.err_log=[]
    def usrpConfig(self):
        global spp

        Fs = 20e6  # 20MHz sampling rate
        rf_freq = 2.62e9  # 2.62GHz carrier frequency
        tx_channel = 0
        txGain = 20  # dB
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
        self.usrpConfigButton.setDisabled(True)
    def Generation(self):
        self.clearGridLayout(self.gridLayout_5)
        self.clearGridLayout(self.gridLayout_8)
        wifiIdx = self.wifiCombo.currentText()
        dvbt2Idx = self.DVBCombo.currentText()
        
        self.wifi6Std = self.WIFI_STD.value()
        self.dvbt2Std =self.DVBT2_STD.value()
        self.wifi6DC  = self.WIFI_DC_R.value()*1e-5+1j*self.WIFI_DC_I.value()*1e-5
        self.dvbt2DC  = self.DVBT2_DC_R.value()+1j*self.DVBT2_DC_I.value()
        
        self.finalIndexWifi = int(wifiIdx)
        self.finalIndexDvbt2 = int(dvbt2Idx)
        
        # self.TxLabel.setText(str(self.dataWifi[int(wifiIdx)]))
        print(f'wifi : {wifiIdx}, DVB-T2 : {dvbt2Idx}')


        self.generationClicked = True
        
        if self.finalIndexWifi == 0 or self.finalIndexDvbt2 == 0:
            self.wifiRectangles = self.seq_wifi6[int(wifiIdx)]
            self.dvbt2Rectangles = self.seq_dvbt2[int(dvbt2Idx)]
            self.temp1 = self.wifiRectangles
            self.temp2 = self.dvbt2Rectangles
            self.drawRecursiveRectangles()
            print('complete.')
            signal_dvbt2 = True # True for DVB-T2, False for Wifi6
            if signal_dvbt2:
            ## Tx P1-Preamble Normalization
                bits = np.int32(np.round(np.random.random(4*2**11)))
                dvbt2_20MHz = np.hstack((genDvbt2.genDvbt2(bits),np.zeros(10000-8960, dtype=np.complex64))) # 20MHz 10000 samples
                # dvbt2_org = loadmat("./data/generatedP1.mat")['data'].flatten()
                # dvbt2_20MHz = sg.resample(dvlsbt2_org, int(len(dvbt2_org)*35/16))
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
        elif self.finalIndexWifi == 10000 or self.finalIndexDvbt2 == 10000:
            possible_values = [0, 1]

            # Define the corresponding probabilities
            dvbt2_probabilities = [2/3, 1/3]

            # Generate 1000 random samples based on the specified probabilities
            num_samples = 1000
            random_variables1 = random.choices(possible_values, weights=dvbt2_probabilities, k=num_samples)

            dvbt2_seq1 = random_variables1

            non_dvbt2_count = random_variables1.count(0)
            probabilities = [1/2, 1/2]
            random_variables2 = random.choices(possible_values,weights=probabilities, k=non_dvbt2_count*5)

            wifi6_seq1 = np.zeros(5000, dtype=int)

            k = 0
            for i in range(1000):
                if random_variables1[i] == 0:
                    for j in range(5):
                        wifi6_seq1[i*5+j] = random_variables2[k]
                        k += 1 
                        
            self.temp1 = wifi6_seq1
            self.temp2 = dvbt2_seq1
            self.wifiRectangles = wifi6_seq1
            self.dvbt2Rectangles = dvbt2_seq1
            self.drawRecursiveRectangles()
            print('complete.')
            print("generating TX signal...")
            
            tx_wifi6 = np.zeros(2000*Nseq_wifi6, dtype=np.complex64)
            tx_dvbt2 = np.zeros(10000*Nseq_dvbt2, dtype=np.complex64)
            for idx in range(Nseq_wifi6):
                if wifi6_seq1[idx] == 1:
                    bits = np.int32(np.round(np.random.random(12*48*4))) # random bits for wifi6 data
                    tx_wifi6[idx*Nwifi6:(idx+1)*Nwifi6] = genWifi6.genWifi6(bits)
            for idx in range(Nseq_dvbt2):
                if dvbt2_seq1[idx] == 1:
                    bits = np.int32(np.round(np.random.random(4*2048))) # random bits for dvbt2 data
                    tx_dvbt2[idx*10000:idx*10000+8960] += genDvbt2.genDvbt2(bits)
            tx = np.zeros(7500000, dtype=np.complex64)
            self.xs = np.hstack((tx,tx_wifi6+tx_dvbt2))
            
        else :
            self.wifiRectangles = self.seq_wifi6[int(wifiIdx)]
            self.dvbt2Rectangles = self.seq_dvbt2[int(dvbt2Idx)]
            self.temp1 = self.wifiRectangles
            self.temp2 = self.dvbt2Rectangles
            self.drawRecursiveRectangles()
            print('complete.')
            # self.xs = genTx.genTx(int(wifiIdx), int(dvbt2Idx))
            ################################################################ For simple test
            self.xs = wifi6_test.test()
            ################################################################
        
        
        
        # plt.figure(); plt.plot(self.xs.real);plt.show()
        self.TxThread = Process(target=self.tx_func, args = (self.tx_streamer,self.tx_metadata,self.xs))
        self.dvbT = threading.Thread(target = self.DVBT2_func)
        self.turnOnOffUSRPButton.setEnabled(True)
        self.GenerationButton.setDisabled(True)
    def tx_func(self, tx_streamer,tx_metadata,xs):
        global tx_flag, signal_dvbt2
        tx_cnt=0
        # print('++++++++ Tx Thread Started ++++++++')
        # self.GenerationButton.setText('Tx Sending')
        # print('sending?????')
        while tx_cnt<10000:
            # idx=tx_cnt%199 # for Generated Data
            idx=tx_cnt%8750 # for Generated Data
            # idx=tx_cnt%10 # for DVB-T2
            # idx=tx_cnt%2 # for WiFi6
            num_tx_samps = tx_streamer.send(xs[2000*idx:2000*(idx+1)],tx_metadata)
            tx_cnt += 1
            if tx_cnt % 10000 == 0 and tx_cnt!= 40000:
                # self.TxProgressBar.setValue(int(tx_cnt/40000*100))
                pass
        tx_flag=False
        # self.TxProgressBar.setValue(100)
        # print('++++++++Tx Thread finished ++++++++')
        # self.GenerationButton.setText('Tx Finished')
    def DVBT2_func(self):
        global dvbt2_buffer, rx_count,dvbt2_result,rx_count_End
        totalTime1 = totalTime2 = totalTime = 0
        dvbDetectionCount = 0
        
        dataload = np.load('lookupTable.npy')
        self.dvbt2DC = dataload[int(self.myusrp.get_tx_gain()//5),1,1]-1j*dataload[int(self.myusrp.get_tx_gain()//5),1,2]
        self.dvbt2Std = dataload[int(self.myusrp.get_tx_gain()//5),1,3]
        while rx_count < rx_count_End:
            # dvbEvent.wait()
            dvbDetectionCount += 1
            self.dvbEvent.clear()
            start = time.time_ns()
            test = (dvbt2_buffer-self.dvbt2DC)/self.dvbt2Std
            dvbt2_result[:-1,:] = dvbt2_result[1:,:]
            dvbt2_result[-1, 0] = dvbt2_detectV2.dvbt2_detectV2((test[:Ndvbt2]))
            dvbt2_result[-1, 1] = dvbt2_detectV2.dvbt2_detectV2((test[Ndvbt2//2:15000]))

            self.dvbEvent.wait()
            end2 = time.time_ns()
            totalTime += end2-start
        print('DVBT2_Thread finished')
        # print(f'DVBT2 Count = {dvbDetectionCount}')
        # print(f'DVBT2 Average Time1  = {totalTime1/dvbDetectionCount/1e3:.2f}us')
        # print(f'DVBT2 Average Time2  = {totalTime2/dvbDetectionCount/1e3:.2f}us')
        print(f'DVBT2 Average Time  = {totalTime/dvbDetectionCount/1e3:.2f}us, DVBT2 Count = {dvbDetectionCount}')
        np.save('test',dvbt2_result)
        # print('saved')
    def calculation(self):
        global dvbt2_result, wifi6_result
        
        if self.finalIndexWifi == 10000 or self.finalIndexDvbt2 == 10000:
            wifi6_seq = self.temp1
            dvbt2_seq = self.temp2
        else :
            wifi6_seq = self.seq_wifi6[int(self.wifiCombo.currentText())]
            dvbt2_seq = self.seq_dvbt2[int(self.DVBCombo.currentText())]
        wifi6_seq_ones = wifi6_seq==1
        wifi6_seq_zeros = wifi6_seq==0
        dvbt2_seq_ones = dvbt2_seq==1
        dvbt2_seq_zeros = dvbt2_seq==0
        
        # in 0 dB
        # wifi6_result /= 6130
        # dvbt2_result /= 9.97
        # in 5 dB
        # wifi6_result /= 730
        # dvbt2_result /= 31.85
        
        # in 5 dB
        wifi6_result /= wifi6_result.max()
        dvbt2_result /= dvbt2_result.max()
        
        corr_wifi6_full = np.correlate(wifi6_result[:,0], wifi6_seq, "full")
        corr_wifi6_full = np.vstack((corr_wifi6_full, np.correlate(wifi6_result[:,1], wifi6_seq, "full")))
        corr_dvbt2_full = np.correlate(dvbt2_result[:,0], dvbt2_seq, "full")
        corr_dvbt2_full = np.vstack((corr_dvbt2_full, np.correlate(dvbt2_result[:,1], dvbt2_seq, "full")))
        
        wifi6_sel = 0 if (np.max(corr_wifi6_full[0]) > np.max(corr_wifi6_full[1])) else 1
        dvbt2_sel = 0 if (np.max(corr_dvbt2_full[0]) > np.max(corr_dvbt2_full[1])) else 1
        wifi6_offset = np.argmax(corr_wifi6_full[wifi6_sel])-Nseq_wifi6+1
        dvbt2_offset = np.argmax(corr_dvbt2_full[dvbt2_sel])-Nseq_dvbt2+1
        print(f"wifi6_sel: {wifi6_sel}")
        print(f"dvbt2_sel: {dvbt2_sel}")
        print(f"wifi6_offset: {wifi6_offset}")
        print(f"dvbt2_offset: {dvbt2_offset}")
        #####
        self.wifi6_valid = wifi6_result[wifi6_offset:wifi6_offset+Nseq_wifi6,wifi6_sel]
        self.dvbt2_valid = dvbt2_result[dvbt2_offset:dvbt2_offset+Nseq_dvbt2,dvbt2_sel]
        
        threshold = 0.5
        
        self.wifi6_md = np.count_nonzero(self.wifi6_valid[wifi6_seq_ones] < threshold)
        self.wifi6_fa = np.count_nonzero(self.wifi6_valid[wifi6_seq_zeros] > threshold)
        self.dvbt2_md = np.count_nonzero(self.dvbt2_valid[dvbt2_seq_ones] < threshold)
        self.dvbt2_fa = np.count_nonzero(self.dvbt2_valid[dvbt2_seq_zeros] > threshold)
        
        self.wifi6_validBi = np.int16(self.wifi6_valid>threshold)
        self.dvbt2_validBi = np.int16(self.dvbt2_valid>threshold)
        
        self.wifiM.setText(str(self.wifi6_md))
        self.wifiF.setText(str(self.wifi6_fa))
        self.dvbM.setText(str(self.dvbt2_md))
        self.dvbF.setText(str(self.dvbt2_fa))
    def turnOnOffUSRP(self):
        global dvbt2_buffer, rx_count, rx_count_End,wifi6_result
        if self.usrp_flag == True:
            self.usrp_flag = False
            rx_count= 0
            wifiDetectionCount = 0
            rx_count_End = 10000
            wifiIdx_det = -4000
            wifi6_result = np.zeros((rx_count_End, 2))
            print('************ USRP ON *************************')
            self.TxThread.start()
            self.dvbT.start()
            # self.wifiT.start()
            timeTest = 0
            print('************ Receiver Turn On ****************')
            dataload = np.load('lookupTable.npy')
            self.wifi6DC = dataload[int(self.myusrp.get_tx_gain()//5),0,1]-1j*dataload[int(self.myusrp.get_tx_gain()//5),0,2]
            self.wifi6Std = dataload[int(self.myusrp.get_tx_gain()//5),0,3]
            print((self.wifi6DC),self.wifi6Std)
            while rx_count < rx_count_End:
                recv_buffer = np.zeros((2000), dtype=np.complex64)
                nrsamp = self.rx_streamer.recv(recv_buffer, self.rx_metadata)
                dvbt2_buffer = np.hstack((dvbt2_buffer[Nrx_buffer:], recv_buffer))
                self.err_log.append([rx_count,nrsamp,recv_buffer])
                rx_count += 1
                start = time.time_ns()
                test= (dvbt2_buffer[wifiIdx_det:]-self.wifi6DC)/self.wifi6Std
                # test = (dvbt2_buffer[wifiIdx_det:])
                wifi6_result[:-1,:] = wifi6_result[1:,:]
                wifi6_result[-1, 0] = wifi6_detect.wifi6_detect(test[:Nwifi6])
                wifi6_result[-1, 1] = wifi6_detect.wifi6_detect(test[Nwifi6//2:Nwifi6+Nwifi6//2])
                end = time.time_ns()
                if rx_count % 5 == 0:
                    self.dvbEvent.set()
                timeTest += end-start
            self.rx_flag = False
            np.save('test2',wifi6_result)
            print(f'Time taken to send {rx_count} samples = {timeTest/rx_count/1e3} ms')
            print('************ Receiver Turn Off ************')
            tmp = np.zeros((len(self.err_log),2000), dtype = np.complex64)
            for count in range(len(tmp)):
                tmp[count] = self.err_log[count][2]
            # save?
            # np.save(f'{self.myusrp.get_tx_gain()}dB data', tmp.flatten())
            idx = 0
            plt.figure()
            plt.plot(tmp.flatten().real)
            plt.title(f'{self.myusrp.get_tx_gain()}')
            plt.tight_layout()
            plt.show()
                        
            print('************ TxRx Finished ****************')
            self.dvbT.join()
            self.TxThread.join()
            self.dvbT = None
            self.TxThread = None
            self.turnOnOffUSRPButton.setText('Receiver Turn Off')
            ################################
            self.calculation()
            self.wifiRectanglesRx = self.wifi6_validBi
            self.temp3 = self.wifi6_validBi
            self.dvbt2RectanglesRx = self.dvbt2_validBi
            self.temp4 = self.dvbt2_validBi
            ################################
            self.flag = True
            self.time = 0
            print('************ Drawing Recursive Rectangles ****************')
            self.drawRecursiveRectanglesRx()
        else :
            self.rx_flag = False
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            self.rx_streamer.issue_stream_cmd(stream_cmd)

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
    def prev(self):
        self.clearGridLayout(self.gridLayout_5)
        self.clearGridLayout(self.gridLayout_8)
        self.prevNext -= 1
        label2 = QLabel('Time : ', self)
        label2.setAlignment(Qt.AlignRight)
        label2.setFixedHeight(13)
        label2.setMargin(0)
        self.gridLayout_5.addWidget(label2)
        msg = 'Time : '+str((self.prevNext+1)*10)+'ms'
        label2.setText(msg)
        self.drawRectangles(self.temp1[100*self.prevNext:100*self.prevNext+self.wifiCount], 'wifi', self.wifiCount)
        self.drawRectangles(self.temp2[20*self.prevNext:20*self.prevNext+self.dvbCount], 'DVBT2', self.dvbCount)
        self.drawRectanglesRx(self.temp3[100*self.prevNext:100*self.prevNext+self.wifiCount], 'wifi', self.wifiCount)
        self.drawRectanglesRx(self.temp4[20*self.prevNext:20*self.prevNext+self.dvbCount], 'DVBT2', self.dvbCount)
        
                
        if self.prevNext == 0:
            self.prevButton.setDisabled(True)
        else:
            self.prevButton.setEnabled(True)
        if self.prevNext == 50:
            self.nextButton.setDisabled(True)
        else:
            self.nextButton.setEnabled(True)
    def next(self):
        self.clearGridLayout(self.gridLayout_5)
        self.clearGridLayout(self.gridLayout_8)
        
        self.prevNext += 1
        label2 = QLabel('Time : ', self)
        label2.setAlignment(Qt.AlignRight)
        label2.setFixedHeight(13)
        label2.setMargin(0)
        self.gridLayout_5.addWidget(label2)
        msg = 'Time : '+str((self.prevNext+1)*10)+'ms'
        label2.setText(msg)
        self.drawRectangles(self.temp1[100*self.prevNext:100*self.prevNext+self.wifiCount], 'wifi', self.wifiCount)
        self.drawRectangles(self.temp2[20*self.prevNext:20*self.prevNext+self.dvbCount], 'DVBT2', self.dvbCount)
        self.drawRectanglesRx(self.temp3[100*self.prevNext:100*self.prevNext+self.wifiCount], 'wifi', self.wifiCount)
        self.drawRectanglesRx(self.temp4[20*self.prevNext:20*self.prevNext+self.dvbCount], 'DVBT2', self.dvbCount)
        
        if self.prevNext == 0:
            self.prevButton.setDisabled(True)
        else:
            self.prevButton.setEnabled(True)
        if self.prevNext == 50:
            self.nextButton.setDisabled(True)
        else:
            self.nextButton.setEnabled(True)
    def setThreshold(self):
        self.clearGridLayout(self.gridLayout_8)
        if self.finalIndexWifi == 10000 or self.finalIndexDvbt2 == 10000:
            wifi6_seq = self.temp1
            dvbt2_seq = np.array(self.temp2)
        else :
            wifi6_seq = self.seq_wifi6[int(self.wifiCombo.currentText())]
            dvbt2_seq = self.seq_dvbt2[int(self.DVBCombo.currentText())]
        wifi6_seq_ones = wifi6_seq==1
        wifi6_seq_zeros = wifi6_seq==0
        dvbt2_seq_ones = dvbt2_seq==1
        dvbt2_seq_zeros = dvbt2_seq==0
        
        self.changedThreasholder = self.thresholdSpinBox.value()
        
        self.wifi6_md = np.count_nonzero(self.wifi6_valid[wifi6_seq_ones] < self.changedThreasholder)
        self.wifi6_fa = np.count_nonzero(self.wifi6_valid[wifi6_seq_zeros] > self.changedThreasholder)
        self.dvbt2_md = np.count_nonzero(self.dvbt2_valid[dvbt2_seq_ones] < self.changedThreasholder)
        self.dvbt2_fa = np.count_nonzero(self.dvbt2_valid[dvbt2_seq_zeros] > self.changedThreasholder)
        
        self.temp3 = np.int16(self.wifi6_valid>self.changedThreasholder)
        self.temp4 = np.int16(self.dvbt2_valid>self.changedThreasholder)
        
        self.wifiM.setText(str(self.wifi6_md))
        self.wifiF.setText(str(self.wifi6_fa))
        self.dvbM.setText(str(self.dvbt2_md))
        self.dvbF.setText(str(self.dvbt2_fa))
        
        self.drawRectanglesRx(self.temp3[100*(self.prevNext-1):100*(self.prevNext-1)+self.wifiCount], 'wifi', self.wifiCount)
        self.drawRectanglesRx(self.temp4[20*(self.prevNext-1):20*(self.prevNext-1)+self.dvbCount], 'DVBT2', self.dvbCount)
        
    def showGraph(self):
        if not self.generationClicked:
            return
        
        if self.finalIndexWifi == 10000 or self.finalIndexDvbt2 == 10000:
            wifi6_seq = self.temp1
            dvbt2_seq = np.array(self.temp2)
        else :
            wifi6_seq = self.seq_wifi6[int(self.wifiCombo.currentText())]
            dvbt2_seq = self.seq_dvbt2[int(self.DVBCombo.currentText())]
        
        wifi6_seq_ones = wifi6_seq==1
        wifi6_seq_zeros = wifi6_seq==0
        dvbt2_seq_ones = dvbt2_seq==1
        dvbt2_seq_zeros = dvbt2_seq==0
        
        thresholds = np.arange(0,1,0.02)
        wifi6_md_graph = np.zeros(len(thresholds))
        wifi6_fa_graph = np.zeros(len(thresholds))
        dvbt2_md_graph = np.zeros(len(thresholds))
        dvbt2_fa_graph = np.zeros(len(thresholds))
        
        for thidx in range(len(thresholds)):
            wifi6_md_graph[thidx] = np.count_nonzero(self.wifi6_valid[wifi6_seq_ones] < thresholds[thidx])
            wifi6_fa_graph[thidx] = np.count_nonzero(self.wifi6_valid[wifi6_seq_zeros] > thresholds[thidx])
            dvbt2_md_graph[thidx] = np.count_nonzero(self.dvbt2_valid[dvbt2_seq_ones] < thresholds[thidx])
            dvbt2_fa_graph[thidx] = np.count_nonzero(self.dvbt2_valid[dvbt2_seq_zeros] > thresholds[thidx])

        
        graphWindow = GraphWindow(self)

        ax1 = graphWindow.figure.add_subplot(211)
        ax1.plot(thresholds, wifi6_md_graph, '-o', label='Missed Detection')
        ax1.plot(thresholds, wifi6_fa_graph, '-o', label='False Alarm')
        ax1.legend()
        ax1.set_xlabel('threshold')
        ax1.set_title("wifi6 detection")
        
        ax2 = graphWindow.figure.add_subplot(212)
        ax2.plot(thresholds, dvbt2_md_graph, '-o', label='Missed Detection')
        ax2.plot(thresholds, dvbt2_fa_graph, '-o', label='False Alarm')
        ax2.legend()
        ax2.set_xlabel('threshold')
        ax2.set_title("dvbt2 detection")
        
        graphWindow.figure.tight_layout()
        graphWindow.canvas.draw()
        graphWindow.exec_()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GenTx()
    window.show()
    sys.exit(app.exec_())