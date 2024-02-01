## image encoding & decoding
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math



def loadImage(file_name):
    src = cv2.imread(file_name, cv2.IMREAD_COLOR)
    retval, buf = cv2.imencode('.jpg',src,[cv2.IMWRITE_WEBP_QUALITY, 100])
    return src,retval, buf

src, retval, buf = loadImage('image.jpg')

def test():
    ## buf를 usrp에 보낼 수 있는 데이터로 변환
    ##
    flat_buf = buf.flatten()
    hex_buf = np.vectorize(hex)(flat_buf)
    binary_buf = np.vectorize(lambda x: bin(int(x,16)))(hex_buf)
    bit_arrays = np.array([list(format(int(binary, 2), '08b')) for binary in binary_buf])
    flat_bit_array = np.concatenate(bit_arrays)
    # 대괄호와 따옴표 제거하고 쉼표 추가하여 하나의 배열로 변환
    flat_bit_array = np.concatenate(bit_arrays).astype(np.uint8)

    zero_array = np.zeros(96, dtype=np.uint8)
    concatenated_array = np.concatenate([flat_bit_array, zero_array])

    data = np.array(concatenated_array).reshape(-1, 192)

    ## 데이터 만드는 for 문 // usrp 보내는 데이터 아님
    for i in range (len(data)):
        pilot = np.array([1,1,1,-1])*(1+0j)
        pi = [1,1,1,1,  -1,-1,-1,1,  -1,-1,-1,-1,  1,1,-1,1,  -1,-1,1,1,  -1,1,1,-1,  1,1,1,1,  1,1,-1,1,               1,1,-1,1,  1,-1,-1,1,  1,1,-1,1,  -1,-1,-1,1,  -1,1,-1,-1,  1,-1,-1,1,  1,1,1,1,  -1,-1,1,1,           -1,-1,1,-1,  1,-1,1,1,  -1,-1,-1,1,  1,-1,-1,-1,  -1,1,-1,-1,  1,-1,1,1,  1,1,-1,1,  -1,1,-1,1,             -1,-1,-1,-1,  -1,1,-1,1,  1,-1,1,-1,  1,1,1,-1,  -1,1,-1,-1,  -1,1,1,1,  -1,-1,-1,-1,  -1,-1,-1
        ,1,1,1,1,  -1,-1,-1,1,  -1,-1,-1,-1,  1,1,-1,1,  -1,-1,1,1,  -1,1,1,-1,  1,1,1,1,  1,1,-1,1,               1,1,-1,1,  1,-1,-1,1,  1,1,-1,1,  -1,-1,-1,1,  -1,1,-1,-1,  1,-1,-1,1,  1,1,1,1,  -1,-1,1,1,           -1,-1,1,-1,  1,-1,1,1,  -1,-1,-1,1,  1,-1,-1,-1,  -1,1,-1,-1,  1,-1,1,1,  1,1,-1,1,  -1,1,-1,1,             -1,-1,-1,-1,  -1,1,-1,1,  1,-1,1,-1,  1,1,1,-1,  -1,1,-1,-1,  -1,1,1,1,  -1,-1,-1,-1,  -1,-1,-1
        ,1,1,1,1,  -1,-1,-1,1,  -1,-1,-1,-1,  1,1,-1,1,  -1,-1,1,1,  -1,1,1,-1,  1,1,1,1,  1,1,-1,1,               1,1,-1,1,  1,-1,-1,1,  1,1,-1,1,  -1,-1,-1,1,  -1,1,-1,-1,  1,-1,-1,1,  1,1,1,1,  -1,-1,1,1,           -1,-1,1,-1,  1,-1,1,1,  -1,-1,-1,1,  1,-1,-1,-1,  -1,1,-1,-1,  1,-1,1,1,  1,1,-1,1,  -1,1,-1,1,             -1,-1,-1,-1,  -1,1,-1,1,  1,-1,1,-1,  1,1,1,-1,  -1,1,-1,-1,  -1,1,1,1,  -1,-1,-1,-1,  -1,-1,-1
        ,1,1,1,1,  -1,-1,-1,1,  -1,-1,-1,-1,  1,1,-1,1,  -1,-1,1,1,  -1,1,1,-1,  1,1,1,1,  1,1,-1,1,               1,1,-1,1,  1,-1,-1,1,  1,1,-1,1,  -1,-1,-1,1,  -1,1,-1,-1,  1,-1,-1,1,  1,1,1,1,  -1,-1,1,1,           -1,-1,1,-1,  1,-1,1,1,  -1,-1,-1,1,  1,-1,-1,-1,  -1,1,-1,-1,  1,-1,1,1,  1,1,-1,1,  -1,1,-1,1,             -1,-1,-1,-1,  -1,1,-1,1,  1,-1,1,-1,  1,1,1,-1,  -1,1,-1,-1,  -1,1,1,1,  -1,-1,-1,-1,  -1,-1,-1
            ]#앞에 signal filed의 pilot 1 빼고 시작
        # Mapping
        M = 4           # M=4 for 16-QAM
        mapping_table = {       # gray code mapping
            (0,0,0,0) : -3-3j,
            (0,0,0,1) : -3-1j,
            (0,0,1,0) : -3+3j,
            (0,0,1,1) : -3+1j,
            (0,1,0,0) : -1-3j,
            (0,1,0,1) : -1-1j,
            (0,1,1,0) : -1+3j,
            (0,1,1,1) : -1+1j,
            (1,0,0,0) :  3-3j,
            (1,0,0,1) :  3-1j,
            (1,0,1,0) :  3+3j,
            (1,0,1,1) :  3+1j,
            (1,1,0,0) :  1-3j,
            (1,1,0,1) :  1-1j,
            (1,1,1,0) :  1+3j,
            (1,1,1,1) :  1+1j
        }

        #gen 16_QAM
        ofdm = 64
        DATA = np.zeros(ofdm, np.complex128)

        data4 = np.array(data[1]).reshape(-1, 4)
        output_data = (1/math.sqrt(10))*np.array([mapping_table[tuple(row)] for row in data4])

        sc_data=np.array([-26,-25,-24,-23,-22, -20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8, -6,-5,-4,-3,-2,-1,
                            1,2,3,4,5,6,  8,9,10,11,12,13,14,15,16,17,18,19,20,  22,23,24,25,26])
        sc_pilot = np.array([-21,-7,7,21])

        DATA[sc_data] = output_data
        DATA[sc_pilot] = pi[i%127]*pilot

        #ifft
        #print(np.round(DATA,3))
        #print(len(DATA))
        FINAL_IFFT = np.fft.ifft(DATA)
        FINAL_IFFT = np.hstack((FINAL_IFFT[-16:],np.tile(FINAL_IFFT,2)))[:81]
        FINAL_IFFT[0]/=2; FINAL_IFFT[-1]/=2
        #print(np.round(FINAL_IFFT,3))
        if i == 0:
            USRP = FINAL_IFFT
        else :
            USRP = np.hstack([USRP, FINAL_IFFT])

    zero_array = np.zeros(475, dtype=np.uint8)
    USRP_1 = np.concatenate([USRP, zero_array])
    USRP_2 = np.array(USRP_1).reshape(-1, 1600)
    ## buf array를 usrp로 보내고 다시 받아야 한다
    L = np.load('pre.npy')

    for i in range (len(USRP_2)):
        usrp = np.hstack([L,USRP_2[i]])
        if i == 0:
            USRP_sd = usrp
        else :
            USRP_sd = np.vstack((USRP_sd, usrp.reshape(1, -1)))
    return USRP_sd.reshape(-1)

if __name__ == '__main__':
    test = test()
    print(test.shape)