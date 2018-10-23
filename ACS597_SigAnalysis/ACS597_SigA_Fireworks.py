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
    RfbInds = [mic3_thresh[0]-int(0.5*fs),mic3_thresh[0]+int(0.5*fs)]

    R_31, tau = crossCorr(mic3[RfbInds[0]:RfbInds[1]],mic1[RfbInds[0]:RfbInds[1]],fs)
    R_32, _      = crossCorr(mic3[RfbInds[0]:RfbInds[1]],mic2[RfbInds[0]:RfbInds[1]],fs)
    R_34, _      = crossCorr(mic3[RfbInds[0]:RfbInds[1]],mic4[RfbInds[0]:RfbInds[1]],fs)
    R_12, _      = crossCorr(mic1[RfbInds[0]:RfbInds[1]],mic2[RfbInds[0]:RfbInds[1]],fs)
    R_14, _      = crossCorr(mic1[RfbInds[0]:RfbInds[1]],mic4[RfbInds[0]:RfbInds[1]],fs)
    R_24, _      = crossCorr(mic2[RfbInds[0]:RfbInds[1]],mic4[RfbInds[0]:RfbInds[1]],fs)

    tau_31 = tau[np.argmax(abs(R_31))]
    tau_32 = tau[np.argmax(abs(R_32))]
    tau_34 = tau[np.argmax(abs(R_34))]


    print tau_31
    print tau_32
    print tau_34

    tau_12 = tau[np.argmax(abs(R_12))]
    tau_13 = -1*tau_31
    tau_14 = tau[np.argmax(abs(R_14))]
    tau_23 = -1*tau_32
    tau_24 = tau[np.argmax(abs(R_24))]


    tdVec =[tau_12,tau_13,tau_14,tau_23,tau_24,tau_34]
    for i in tdVec:
        print i

                        #  x ,   y,     z
    XY_rel_3 = np.array([[-29.0, 51.7,  0.0],    # Microphone 1
                        [28.7, 51.7,   0.0],    # Microphone 2
                        [0.0,   0.0,   0.0],    # Microphone 3
                        [0.0,  35.0,  20.0]])

    XY_rel_3_m = XY_rel_3*0.3048
    print XY_rel_3_m

    distMtx = np.zeros((len(tdVec),3))
    print distMtx

    plt.figure()
    plt.plot(times,mic1,label="mic1")
    plt.plot(times,mic2,label="mic2")
    plt.plot(times,mic3,label="mic3")
    plt.plot(times,mic4,label="mic4")
    plt.xlim(times[fbInds[0]],times[fbInds[1]])
    plt.legend()

    plt.figure()
    plt.plot(tau,R_31)

    plt.figure()

    plt.plot(times[fbInds[0]:fbInds[1]],mic1[fbInds[0]+int(tau_31*fs):fbInds[1]+int(tau_31*fs)],label="mic1")
    plt.plot(times[fbInds[0]:fbInds[1]],mic2[fbInds[0]+int(tau_32*fs):fbInds[1]+int(tau_32*fs)],label="mic2")
    plt.plot(times[fbInds[0]:fbInds[1]],mic3[fbInds[0]:fbInds[1]],label="mic3")
    plt.plot(times[fbInds[0]:fbInds[1]],mic4[fbInds[0]+int(tau_34*fs):fbInds[1]+int(tau_34*fs)],label="mic4")
    plt.legend()

    #plt.show()

    return 0

def crossCorr(x_time,y_time,fs):
    N = len(x_time)
    delT,_,_ = sigA.param(N,fs,show=False)
    S_XY = sigA.crossSpec(x_time,y_time,fs)

    R_XX = np.fft.ifft(S_XY)/delT
    sub = 0
    if N%2:
        N+=1
        sub =1

    R_XX = np.concatenate((R_XX[(N/2):],R_XX[0:N/2]))
    times = sigA.timeVec(N,fs)
    timeShift = np.concatenate((-1*times[1:(N/2)+1-sub][::-1],times[0:(N/2)]))

    return R_XX, timeShift

def threshold(x_time):
    thresh = 0.05
    indRange = 1500
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