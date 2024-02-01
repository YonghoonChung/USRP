import  numpy as np
import matplotlib.pyplot as plt
import uhd
import threading, time
from scipy.io import loadmat

plt.close()

dvbt2_org = loadmat("./data/generatedP1.mat")['data'].flatten()

# dvbt2_org = (dvbt2_org-dvbt2_org.mean())/(dvbt2_org.std())

# xreal=dvbt2_org.real-np.mean(dvbt2_org.real)
# ximag=dvbt2_org.imag-np.mean(dvbt2_org.imag)
# tmp = np.sqrt(np.mean(xreal**2)+np.mean(ximag**2))
# xreal = xreal / (dvbt2_org.real).std()
# ximag = ximag / tmp
# dvbt2_org = xreal + 1j*(ximag)

# real_data = np.real(dvbt2_org)
# imag_data = np.imag(dvbt2_org)
#
# real_data = ( real_data - real_data.mean() ) / real_data.std()
# imag_data = ( imag_data - imag_data.mean() ) / imag_data.std()
#
# dvbt2_org_norm = real_data + 1j * imag_data
#
# plt.plot(imag_data)
# plt.show()

# temp = (dvbt2_org - dvbt2_org.min())/(dvbt2_org.max()-dvbt2_org.min())
#########################################
dvbt2_org = loadmat("./data/generatedP1.mat")['data'].flatten()
real_data = np.real(dvbt2_org)
imag_data = np.imag(dvbt2_org)

real_data = ( real_data - real_data.min() ) / (real_data.max() - real_data.min())
imag_data = ( imag_data - imag_data.min() ) / (imag_data.max() - imag_data.min())

dvbt2_org_norm = (2*real_data-1)+(1j*imag_data*2-1j)
plt.figure()
plt.subplot(211)
plt.plot(dvbt2_org_norm.real)
plt.subplot(212)
plt.plot(dvbt2_org_norm.imag)
plt.show()
#########################################
dvbt2_org = loadmat("./data/generatedP1.mat")['data'].flatten()
real_data = np.real(dvbt2_org)
imag_data = np.imag(dvbt2_org)

real_data = ( real_data - real_data.mean() ) / real_data.std()
imag_data = ( imag_data - imag_data.mean() ) / imag_data.std()
plt.figure()
dvbt2_org_norm = real_data + 1j * imag_data
plt.subplot(211)
plt.plot(dvbt2_org_norm.real)
plt.subplot(212)
plt.plot(dvbt2_org_norm.imag)
plt.show()
#########################################
import numpy as np
import scipy.signal as sg
# Number of complex signals you want to generate
num_signals = 10000

# Generate random real and imaginary parts in the range [-1, 1]
real_part = np.random.uniform(-1, 1, num_signals)*0.1
imaginary_part = np.random.uniform(-1, 1, num_signals*0.1)

# Create complex signals
complex_signals = real_part + 1j * imaginary_part
plt.figure()
plt.plot(complex_signals.real)
plt.plot(complex_signals.imag)
plt.show()

dvbt2_org = loadmat("./data/generatedP1.mat")['data'].flatten()
dvbt2_20MHz = sg.resample(dvbt2_org, int(len(dvbt2_org)*35/16))
complex_signals[:len(dvbt2_20MHz)] = dvbt2_20MHz
plt.figure()
plt.plot(complex_signals.real)
plt.plot(complex_signals.imag)
plt.show()