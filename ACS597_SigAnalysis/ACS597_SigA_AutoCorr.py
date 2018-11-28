import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.signal as sig
import numpy.random as rand
import sigA
import sounddevice as sd

def autocor(x_time,fs):
    N= len(x_time)
    delT,_,T=sigA.param(N,fs,show=False)
    S_XX = sigA.dsSpec(x_time,fs)
    R_XX = np.fft.ifft(S_XX)/delT
    sub = 0
    if N%2:
        N+=1
        sub =1

    R_XX = np.concatenate((R_XX[(N/2):],R_XX[0:N/2]))
    times = sigA.timeVec(N,fs)
    timeShift = np.concatenate((-1*times[1:(N/2)+1-sub][::-1],times[0:(N/2)]))

    return R_XX, timeShift

def main(args):
    rand.seed(13)
    fs = 16.E3
    T = 0.5
    N = int(fs*T)
    delT, delF, _ = sigA.param(N,fs)

    noise = rand.randn(N)
    times = sigA.timeVec(N,fs)
    freqs = sigA.freqVec(N,fs)

    x_rms,_,_ = sigA.rms(x_time=noise)
    x_pk = max(noise)

    rat = x_pk/x_rms

    print "rat: " +str(rat)

    S_XX = sigA.dsSpec(x_time = noise,fs=fs)
    R_XX, timeShift = autocor(noise,fs)

    print np.abs(R_XX[0])

    MLS = sig.max_len_seq(13)[0]
    MLS = 2*MLS-1

    MLS_times = sigA.timeVec(len(MLS),fs)
    MLS_rms,MLS_ms,_= sigA.rms(MLS,fs)
    MLSrat = 1./MLS_rms

    print "MLSrat: " +str(MLSrat)

    S_XX_MLS = sigA.dsSpec(MLS,fs)
    R_XX_MLS, timeShiftMLS = autocor(MLS,fs)

    MLS_freqs = sigA.freqVec(len(MLS),fs)

    f = 1000.
    T_CW = 0.1
    CW_Pulse = np.zeros(int(0.3*fs))

    print len(CW_Pulse)

    for i in range(int(fs*T_CW)):
        CW_Pulse[i] = np.sin(2*np.pi*f*delT*i)

    #sd.play(0.5*CW_Pulse,fs,blocking=True)

    S_XX_CW = sigA.dsSpec(CW_Pulse,fs)
    R_XX_CW, timeShift_CW = autocor(CW_Pulse,fs)
    freqs_CW = sigA.freqVec(len(CW_Pulse),fs)

    CW_Tap = np.zeros(int(0.3*fs))
    for i in range(int(fs*T_CW)):
        if i*delT < 0.1*T_CW:
            A = i*delT/(0.1*T_CW)
        elif i*delT > 0.9*T_CW:
            A = (1. - i*delT/(T_CW))/(0.1)
        else:
            A = 1.
        CW_Tap[i] = A*np.sin(2*np.pi*f*delT*i)

    S_XX_CW_Tap = sigA.dsSpec(CW_Tap,fs)
    R_XX_CW_Tap,_ = autocor(CW_Tap,fs)

    #sd.play(0.5*CW_Tap,fs,blocking=True)

    CW_Short = np.zeros(int(0.3*fs))
    T_CW_Short = 0.01

    for i in range(int(fs*T_CW_Short)):
        if i*delT < 0.1*T_CW_Short:
            A = i*delT/(0.1*T_CW_Short)
        elif i*delT > 0.9*T_CW_Short:
            A = (1. - i*delT/(T_CW_Short))/(0.1)
        else:
            A = 1.
        CW_Short[i] = A*np.sin(2*np.pi*f*delT*i)

    S_XX_CW_Short = sigA.dsSpec(CW_Short,fs)
    R_XX_CW_Short,_ = autocor(CW_Short,fs)

    LFM = sig.chirp(times[0:int(0.1/delT)],f0=2000,f1=1000,t1 = times[int(0.1/delT)],method="linear")

    for i in range(int(fs*T_CW)):
        if i*delT < 0.1*T_CW:
            A = i*delT/(0.1*T_CW)
        elif i*delT > 0.9*T_CW:
            A = (1. - i*delT/(T_CW))/(0.1)
        else:
            A = 1.
        LFM[i] *= A

    LFM = np.append(LFM,np.zeros(int(0.2*fs)))

    S_XX_LFM = sigA.dsSpec(LFM,fs)
    R_XX_LFM,_= autocor(LFM,fs)

    #sd.play(0.5*LFM,fs)

    LFM_low = sig.chirp(times[0:int(0.1/delT)],f0=1550,f1=1450,t1 = times[int(0.1/delT)],method="linear")

    for i in range(int(fs*T_CW)):
        if i*delT < 0.1*T_CW:
            A = i*delT/(0.1*T_CW)
        elif i*delT > 0.9*T_CW:
            A = (1. - i*delT/(T_CW))/(0.1)
        else:
            A = 1.
        LFM_low[i] *= A

    LFM_low = np.append(LFM_low,np.zeros(int(0.2*fs)))

    S_XX_LFM_low = sigA.dsSpec(LFM_low,fs)
    R_XX_LFM_low,_= autocor(LFM_low,fs)

    CW_1500 = np.zeros(int(0.3*fs))
    for i in range(int(fs*T_CW)):
        if i*delT < 0.1*T_CW:
            A = i*delT/(0.1*T_CW)
        elif i*delT > 0.9*T_CW:
            A = (1. - i*delT/(T_CW))/(0.1)
        else:
            A = 1.
        CW_1500[i] = A*np.sin(2*np.pi*1500.*delT*i)


    LFM_train = LFM
    CW_train = CW_1500
    for i in range(30):
        LFM_train = np.append(LFM_train,LFM)
        CW_train = np.append(CW_train,CW_1500)


    #sd.play(LFM_train,fs,blocking=True)
    #sd.play(CW_train,fs,blocking=True)


    ### NOISE
    plt.figure()
    plt.plot(times,noise)
    plt.title("Normal Distribution Noise")
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    #plt.savefig("A1.png")

    plt.figure()
    plt.plot(freqs,np.abs(S_XX))
    plt.xlim(freqs[0]-50,freqs[N/2])
    plt.title("PSD")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [Pa^2/Hz]")
    #plt.savefig("A2.png")

    plt.figure()
    plt.plot(timeShift,np.abs(R_XX))
    plt.title("Autocorrelation")
    plt.xlabel("Time Shift [s]")
    plt.ylabel("Autocorrelation")
    #plt.savefig("A3.png")

    ### MLS
    plt.figure()
    plt.title("MLS")
    plt.plot(MLS_times,MLS)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    #plt.savefig("B1.png")


    plt.figure()
    plt.title("MLS PSD")
    plt.plot(MLS_freqs,np.abs(S_XX_MLS))
    plt.xlim(MLS_freqs[0]-50,MLS_freqs[len(MLS_freqs)/2])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [Pa^2/Hz]")
    #plt.savefig("B2.png")


    plt.figure()
    plt.title("MLS Autocorrelation")
    plt.plot(timeShiftMLS,np.abs(R_XX_MLS))
    plt.xlabel("Time Shift [s]")
    plt.ylabel("Autocorrelation")
    #plt.savefig("B3.png")

    print np.abs(R_XX_MLS[0])
    print np.abs(R_XX_MLS[1])
    print 1./len(MLS)

    ### CW PULSE

    plt.figure()
    plt.title("CW Pulse")
    plt.plot(times[0:len(CW_Pulse)],CW_Pulse)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    #plt.savefig("C1.png")

    plt.figure()
    plt.title("CW PSD")
    plt.plot(freqs_CW,S_XX_CW)
    plt.xlim(freqs_CW[0]-50,freqs_CW[len(freqs_CW)/2])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [Pa^2/Hz]")
    #plt.savefig("C2.png")


    plt.figure()
    plt.title("CW Autocorrelation")
    plt.plot(timeShift_CW,R_XX_CW)
    plt.xlabel("Time Shift [s]")
    plt.ylabel("Autocorrelation")
    #plt.savefig("C3.png")

    plt.figure()
    plt.title("CW Pulse, Tapered")
    plt.plot(times[0:len(CW_Tap)],CW_Tap)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    #plt.savefig("D1.png")


    plt.figure()
    plt.title("CW (Tapered) PSD")
    plt.plot(freqs_CW,S_XX_CW_Tap)
    plt.xlim(freqs_CW[0]-50,freqs_CW[len(freqs_CW)/2])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [Pa^2/Hz]")
    #plt.savefig("D2.png")

    plt.figure()
    plt.title("CW (Tapered) Autocorrelation")
    plt.plot(timeShift_CW,R_XX_CW_Tap)
    plt.xlabel("Time Shift [s]")
    plt.ylabel("Autocorrelation")
    #plt.savefig("D3.png")


    plt.figure()
    plt.title("CW Pulse, Short")
    plt.plot(times[0:len(CW_Tap)],CW_Short)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    #plt.savefig("E1.png")


    plt.figure()
    plt.title("CW (Short) PSD")
    plt.plot(freqs_CW,S_XX_CW_Short)
    plt.xlim(freqs_CW[0]-50,freqs_CW[len(freqs_CW)/2])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [Pa^2/Hz]")
    #plt.savefig("E2.png")

    plt.figure()
    plt.title("CW (Short) Autocorrelation")
    plt.plot(timeShift_CW,R_XX_CW_Short)
    plt.xlabel("Time Shift [s]")
    plt.ylabel("Autocorrelation")
    #plt.savefig("E3.png")


    plt.figure()
    plt.title("LFM")
    plt.plot(times[0:int(0.3/delT)],LFM)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    #plt.savefig("F1.png")


    plt.figure()
    plt.title("LFM PSD")
    plt.plot(freqs_CW,S_XX_LFM)
    plt.xlim(freqs_CW[0]-50,freqs_CW[len(freqs_CW)/2])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [Pa^2/Hz]")
    #plt.savefig("F2.png")

    plt.figure()
    plt.title("LFM Autocorrelation")
    plt.plot(timeShift_CW,R_XX_LFM)
    plt.xlabel("Time Shift [s]")
    plt.ylabel("Autocorrelation")
    #plt.savefig("F3.png")


    plt.figure()
    plt.title("LFM, Low")
    plt.plot(times[0:int(0.3/delT)],LFM_low)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    #plt.savefig("G1.png")


    plt.figure()
    plt.title("LFM (low) PSD")
    plt.plot(freqs_CW,S_XX_LFM_low)
    plt.xlim(freqs_CW[0]-50,freqs_CW[len(freqs_CW)/2])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [Pa^2/Hz]")
    #plt.savefig("G2.png")

    plt.figure()
    plt.title("LFM (low) Autocorrelation")
    plt.plot(timeShift_CW,R_XX_LFM_low)
    plt.xlabel("Time Shift [s]")
    plt.ylabel("Autocorrelation")
    #plt.savefig("G3.png")

    plt.show()

    return 0



if __name__ == '__main__':
    sys.exit(main(sys.argv))