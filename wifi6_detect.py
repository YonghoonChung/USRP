import numpy as np

## constants
# sum 16 shift corr
shift_corr_16 = 144+0+0+0+0+64
shift_corr_64 = 96+96+16+16+16+16

def wifi6_detect(rx):
    # rxs1 = np.hstack((np.zeros(16, dtype=np.complex64), rx[:-16]))
    # corr1 = np.correlate(rx,rxs1)[0]
    # rx_norm = rx / rx.std()
    # rx_norm -= rx_norm.mean()
    # corr1 = np.dot(rx_norm[16:],np.conj(rx_norm[:-16]))
    # # rxs2 = np.hstack((np.zeros(64, dtype=np.complex64), rx[:-64]))
    # # corr2 = np.correlate(rx,rxs2)[0]
    # corr2 = np.dot(rx_norm[64:],np.conj(rx_norm[:-64]))
    # # detect = np.abs(corr1+corr2)/(shift_corr_16+shift_corr_64)
    corr1 = np.dot(rx[16:],np.conj(rx[:-16]))
    corr2 = np.dot(rx[64:],np.conj(rx[:-64]))
    return np.abs(corr1)+np.abs(corr2)

##
if __name__ == "__main__":
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    import time
    import genWifi6

    bits = np.int32(np.round(np.random.random(48*4*12)))
    rx = genWifi6.genWifi6(bits)
    loops = 2
    time1 = time.time_ns()
    dx11=wifi6_detect(rx)
    dx22=wifi6_detect(rx)
    # for idx in range(loops):
    #     corr = wifi6_detect(rx)
    time2 = time.time_ns()

    # print(f'corr: {corr}')
    print(f"detection time: {(time2-time1)/loops/1e3:.2f} us")
    plt.figure()
    plt.plot(rx.real)
    plt.title("rx signal")
    plt.show()

    # correlation test
    wifi6_preamble = rx[:1040]
    lstf = wifi6_preamble[:160]
    lltf = wifi6_preamble[160:320]
    lsig = wifi6_preamble[320:400]
    rlsig = wifi6_preamble[400:480]
    siga = wifi6_preamble[480:640]
    hestf = wifi6_preamble[640:720]
    heltf = wifi6_preamble[720:]

    zeros16 = np.zeros(16, dtype=np.complex64)
    zeros64 = np.zeros(64, dtype=np.complex64)

    print(f"lstf corr 16 = {np.abs(np.correlate(lstf,np.hstack((zeros16,lstf[:-16]))))[0]:.1f}")        # 160-16=144
    print(f"lstf corr 64 = {np.abs(np.correlate(lstf,np.hstack((zeros64,lstf[:-64]))))[0]:.1f}")        # 160-64=96
    print(f"lltf corr 16 = {np.abs(np.correlate(lltf,np.hstack((zeros16,lltf[:-16]))))[0]:.1f}")        # 0
    print(f"lltf corr 64 = {np.abs(np.correlate(lltf,np.hstack((zeros64,lltf[:-64]))))[0]:.1f}")        # 160-64=96
    print(f"lsig corr 16 = {np.abs(np.correlate(lsig,np.hstack((zeros16,lsig[:-16]))))[0]:.1f}")        # 0
    print(f"lsig corr 64 = {np.abs(np.correlate(lsig,np.hstack((zeros64,lsig[:-64]))))[0]:.1f}")        # GI: 16
    print(f"rlsig corr 16 = {np.abs(np.correlate(rlsig,np.hstack((zeros16,rlsig[:-16]))))[0]:.1f}")     # 0
    print(f"rlsig corr 64 = {np.abs(np.correlate(rlsig,np.hstack((zeros64,rlsig[:-64]))))[0]:.1f}")     # GI: 16
    print(f"siga corr 16 = {np.abs(np.correlate(siga,np.hstack((zeros16,siga[:-16]))))[0]:.1f}")        # 0
    print(f"siga corr 64 = {np.abs(np.correlate(siga,np.hstack((zeros64,siga[:-64]))))[0]:.1f}")        # why 16?
    print(f"hestf corr 16 = {np.abs(np.correlate(hestf,np.hstack((zeros16,hestf[:-16]))))[0]:.1f}")     # 80-16=64
    print(f"hestf corr 64 = {np.abs(np.correlate(hestf,np.hstack((zeros64,hestf[:-64]))))[0]:.1f}")     # 80-64=16
    print(f"heltf corr 16 = {np.abs(np.correlate(heltf,np.hstack((zeros16,heltf[:-16]))))[0]:.1f}")     # 0
    print(f"heltf corr 64 = {np.abs(np.correlate(heltf,np.hstack((zeros64,heltf[:-64]))))[0]:.1f}")     # 0

