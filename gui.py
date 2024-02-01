import sys,uhd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
import random
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import uic
import json


UI_class = uic.loadUiType("gui.ui")[0]


class MyWindow(QMainWindow, UI_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setFixedWidth(1594)
        with open('model_value.json', 'r') as file:
            self.model_value = json.load(file)
        # signal option
        self.wifiCombo.currentIndexChanged.connect(self.changeItem)
        self.GenerationButton.clicked.connect(self.Generation)
        self.GenerationButton.setDisabled(True)
        self.showGraphButton.clicked.connect(self.showGraph)
        self.usrpConfigButton.clicked.connect(self.usrpConfig)
        self.turnOnOffUSRPButton.clicked.connect(self.turnOnOffUSRP)
        self.turnOnOffUSRPButton.setDisabled(True)
        self.turnOnOffUSRPButton.setText("Receiver Turn On")
        self.usrpUpdate.clicked.connect(self.update)
        
        self.DVBT2_DC_R.valueChanged.connect(self.printDoubleValue)
        self.DVBT2_DC_I.valueChanged.connect(self.printDoubleValue)
        self.DVBT2_STD.valueChanged.connect(self.printDoubleValue)
        
        self.centerFreqBox.valueChanged.connect(self.printDoubleValue)        
        self.txGainBox.valueChanged.connect(self.printSingleValue)
        self.rxGainBox.valueChanged.connect(self.printSingleValue)
        
        self.DVBT2_DC_BUTTON.clicked.connect(self.setNormFactor)
        self.DVBT2_STD_BUTTON.clicked.connect(self.setNormFactor)
        self.thresholdButton.clicked.connect(self.setThreshold)
        self.exitButton.clicked.connect(self.close)
        
        self.prevButton.clicked.connect(self.prev)
        self.nextButton.clicked.connect(self.next)

        self.timer1 = QTimer()  # Declare timer as an instance variable
        self.timer = QTimer()
        self.timer1.timeout.connect(self.drawRecursiveRectangles)
        self.timer.timeout.connect(self.drawRecursiveRectanglesRx)

        self.wifiRectangles = []
        self.dvbt2Rectangles = []
        self.wifiRectanglesRx = []
        self.dvbt2RectanglesRx =[]
        self.wifiIndex = 0
        self.dvbt2Index = 0
        self.time = 0
        self.flag = True
        self.wifiCount = 100
        self.dvbCount = self.wifiCount//5
        self.channel = np.ones(8)
        self.generationClicked = False

        self.randseq = np.load("randomSeq.npz")
        self.dataWifi = self.randseq['seq_wifi6']
        self.dataDVBT2 = self.randseq['seq_dvbt2']
        # self.data = np.load('tx_data.npz')
        
        self.wifi6_validBi = []
        self.dvbt2_validBi = []
        
        self.wifi6Std = 0
        self.dvbt2Std = 0
        self.wifi6DC = 0+1j
        self.dvbt2DC = 0+1j
        
        self.centerFreqBox.setValue(self.model_value['centerFreq'])
        self.txGainBox.setValue(self.model_value['txGain'])
        self.rxGainBox.setValue(self.model_value['rxGain'])
        self.DVBT2_DC_R.setValue(self.model_value['DC_R'])
        self.DVBT2_DC_I.setValue(self.model_value['DC_I'])
        self.DVBT2_STD.setValue(self.model_value['std']) 
        
        self.myusrp = None
    def changeItem(self):
        print(self.wifiCombo.currentText())

    def spinBoxChanged(self):
        self.channel[int(self.sender().objectName()[-1])-1] = self.sender().value()

    def AWGN_Func(self):
        AWGNval = self.AWGN_SPIN.value()
        print(AWGNval)

    def Generation(self):
        self.GenerationButton.setDisabled(True)
        self.clearGridLayout(self.gridLayout_5)
        wifiIdx = self.wifiCombo.currentText()
        dvbt2Idx = self.DVBCombo.currentText()
        # self.TxLabel.setText(str(self.dataWifi[int(wifiIdx)]))
        print(f'wifi : {wifiIdx}, DVB-T2 : {dvbt2Idx}')
        self.wifiRectangles = self.dataWifi[int(wifiIdx)]
        self.dvbt2Rectangles = self.dataDVBT2[int(dvbt2Idx)]
        self.time = 0
        print(self.channel)
        self.drawRecursiveRectangles()
        self.generationClicked = True
        
    def update(self):
        with open('model_value.json', 'w', encoding='utf-8') as file:
            self.model_value['centerFreq'] = self.centerFreqBox.value()
            self.model_value['txGain'] = self.txGainBox.value()
            self.model_value['rxGain'] = self.rxGainBox.value()
            json.dump(self.model_value, file, ensure_ascii=False, indent=3)
        print('saved')
        if self.myusrp is not None:
            self.myusrp.set_tx_gain(self.txGainBox.value())
            self.myusrp.set_rx_gain(self.rxGainBox.value())
            self.myusrp.set_tx_freq(uhd.libpyuhd.types.tune_request(self.centerFreqBox.value()), 0)
            self.myusrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.centerFreqBox.value()), 0)
            print(self.myusrp.get_tx_gain())
    def prev(self):
        pass
    def next(self):
        pass
    def usrpConfig(self):
        pass
    def turnOnOffUSRP(self):
        pass 
    def printDoubleValue(self) :
        pass
    def printSingleValue(self) :
        pass
    def setThreshold(self):
        pass
    def setNormFactor(self):  
        # self.wifi6Std = self.WIFI_STD.value()`9`
        self.dvbt2Std =self.DVBT2_STD.value() 
        # self.wifi6DC  = self.WIFI_DC_R.value()+1j*self.WIFI_DC_I.value()
        self.dvbt2DC  = self.DVBT2_DC_R.value()+1j*self.DVBT2_DC_I.value()

        # print(f'wifi6DC : {self.wifi6DC}, dvbt2DC : {self.dvbt2DC}')
        # print(f'wifi6Std : {self.wifi6Std}, dvbt2Std : {self.dvbt2Std}')
        with open('model_value.json', 'w', encoding='utf-8') as file:
            self.model_value['DC_R'] = self.DVBT2_DC_R.value()
            self.model_value['DC_I'] = self.DVBT2_DC_I.value()
            self.model_value['std'] = self.DVBT2_STD.value() 
            json.dump(self.model_value, file, ensure_ascii=False, indent=3)
        print('saved')
        
        self.DVB_Status.setText(f'DC = {self.DVBT2_DC_R.value()}+j{self.DVBT2_DC_I.value()}, Std = {self.DVBT2_STD.value()}')
    def drawRecursiveRectangles(self):
        if len(self.wifiRectangles) < self.wifiCount and len(self.dvbt2Rectangles) < self.dvbCount:
            self.timer1.stop()
            return
        self.clearGridLayout(self.gridLayout_5)
        self.clearGridLayout(self.gridLayout_8)
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
    
    def drawRectangles(self, data, wORd, count):
        label1 = QLabel('Time : ', self)
        label1.setAlignment(Qt.AlignRight)
        label1.setFixedHeight(13)
        label1.setMargin(0)
        if self.flag == True:
            self.time = self.time + 1
        self.flag = ~self.flag
        msg = 'Time : '+str(self.time*10)+'ms'
        label1.setText(msg)
        self.gridLayout_5.addWidget(label1)
        frame = Drawing(data, wORd, self)
        self.gridLayout_5.addWidget(frame)

    def drawRectanglesRx(self, data, wORd, count):
        label2 = QLabel('Time : ', self)
        label2.setAlignment(Qt.AlignRight)
        label2.setFixedHeight(13)
        label2.setMargin(0)
        if self.flag == True:
            self.time = self.time + 1
        self.flag = ~self.flag
        # print(self.flag)
        msg = 'Time : '+str(self.time*10)+'ms'
        label2.setText(msg)
        self.gridLayout_8.addWidget(label2)
        frame = Drawing(data, wORd, self)
        self.gridLayout_8.addWidget(frame)

    def clearGridLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def showGraph(self):
        # graphWindow = GraphWindow(self)
        # # Perform necessary operations to generate the graph using Matplotlib
        # # Example code:
        # ax = graphWindow.figure.add_subplot(111)
        # ax.plot([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])  # Sample graph data
        # graphWindow.canvas.draw()
        # graphWindow.exec_()
        pass
    def close(self):
        sys.exit()
        

class Drawing(QFrame):
    def __init__(self, data, wORd, parent=None):
        super().__init__(parent)
        self.data = data
        self.wORd = wORd
        if wORd == 'wifi':
            self.noDrawings = 100
        else:
            self.noDrawings = 20

    def paintEvent(self, event):
        # print("Main Window Width:", self.width())
        painter = QPainter(self)
        painter.setPen(QColor(0, 0, 0))
        # print(self.parentWidget().width())
        spacing = int((self.parentWidget().width() - 30) / self.noDrawings)  # Convert to integer
        for idx in range(len(self.data)):
            if self.data[idx] == 1:
                if self.wORd == "wifi":
                    painter.setBrush(QColor(0, 0, 255))
                else:
                    painter.setBrush(QColor(5,102,8))
            else:
                painter.setBrush(Qt.NoBrush)
            x = spacing * idx  # Calculate the x-coordinate based on the spacing and index
            painter.drawRect(x, 5, spacing-3, 150)

class GraphWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Result')
        layout = QVBoxLayout(self)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

if __name__ == '__main__':
    global AWGNval, wifiIdx, dvbt2Idx
    app = QApplication(sys.argv)
    Window = MyWindow()
    Window.show()
    app.exec_()
