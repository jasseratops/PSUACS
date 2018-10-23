import scipy.signal as sig
import sigA
import numpy as np
from numpy import pi, cos, sin, exp
import soundfile as sf
import sounddevice as sd
import sys
import matplotlib.pyplot as plt

filename = "Test_shoot_A.wav"

def main(args):
    TestShoot = sf.SoundFile(filename,mode='r')
    print TestShoot
    data  = TestShoot.read()
    fs = TestShoot.samplerate

    print data
    print np.shape(data)

    mic1 = data[:,0]
    mic2 = data[:,1]
    mic3 = data[:,2]
    mic4 = data[:,3]
    N = len(mic1)
    print fs
    times = sigA.timeVec(N,fs)

    mic3_thresh = threshold(mic3)
    print mic3_thresh
    fbInds = [mic3_thresh[0]-int(0.05*fs),mic3_thresh[0]+int(0.1*fs)]
    fbInds_crossCorr = [mic3_thresh[0]-int(0.5*fs),mic3_thresh[0]+int(0.5*fs)]


    plt.figure()
    plt.plot(times,mic1,label="mic1")
    plt.plot(times,mic2,label="mic2")
    plt.plot(times,mic3,label="mic3")
    plt.plot(times,mic4,label="mic4")
    plt.xlim(times[fbInds[0]],times[fbInds[1]])
    plt.legend()
    plt.show()

    return 0

def crossCorr(x_time,y_time,fs):


def threshold(x_time):
    thresh = 0.05
    indRange = 480
    peakInds = np.zeros(1,dtype=int)
    init = 0
    N = len(x_time)
    i = 0
    while i < N:
        endRange = i+indRange
        if endRange >=N:
            endRange=-1
        if x_time[i] >= thresh:
            peakInd = np.argmax(x_time[i:endRange]) + i
            if init == 0:
                peakInds[0]= peakInd
                init = 1
            else:
                peakInds = np.append(peakInds,peakInd)
            i += indRange
        else:
            i +=1
        if endRange == -1:
            break

    return peakInds

if __name__=='__main__':
    sys.exit(main(sys.argv))