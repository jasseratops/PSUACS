import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt
import sigA
import scipy.signal as sig
import soundfile as sf

def main(args):
    part1()
    part2()
    part3A()
    part3B()
    return 0

def part1():
    fs = 48.E3

    f_1 = 3500.
    f_2 = 500.
    T_P = 1.0
    D = 0.5

    N_P = int(fs*T_P)

    t = sigA.timeVec(N_P,fs)

    b = (f_2 - f_1)/T_P

    t_0 = f_1/b

    print b
    print t_0

    A = 0
    B = 2*pi*f_1
    C = pi*b

    print "A: " + str(A)
    print "B: " + str(B)
    print "C: " + str(C)

    phi = A+(B*t)+(C*(t**2))

    pulse = D*sin(phi)

    T = 1.2
    N = int(T*fs)
    times = sigA.timeVec(N,fs)
    x = np.zeros(N)
    x[int(0.1*fs):int(0.1*fs)+len(pulse)] = pulse
    '''
    plt.figure()
    plt.subplot(211)
    plt.plot(t,phi)
    plt.subplot(212)
    plt.plot(t,pulse)
    
    plt.figure()
    plt.plot(times,x)
    '''
    plt.figure()
    sigA.spectrogram(x,fs,1024*4,overlap=0.5,winType="hann")
    plt.ylim(0,5000)
    plt.xlim(0.1,1.1)
    plt.axhline(f_1,color="black")
    plt.axhline(f_2,color="black")
    plt.title("LFM Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    G_XX = sigA.ssSpec(x,fs)
    freqs = sigA.freqVec(N,fs)
    plt.figure()
    plt.plot(freqs[:len(G_XX)],G_XX)
    plt.axvline(f_1,color="black")
    plt.axvline(f_2,color="black")
    plt.xlim(0,freqs[len(G_XX)-1])
    plt.grid()
    plt.title("LFM: Single Sided Spectral Density")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$G_{XX}$ [Pa/${Hz^2}$]")
    #plt.xlim(0,fs/2)

    hil = sigA.hilbert(x,fs)
    env = np.abs(hil)

    plt.figure()
    plt.subplot(211)
    plt.plot(times,x)
    plt.plot(times,env)
    plt.xlim(times[int(fs*(0.1-0.01))],times[int(fs*(0.1+0.01))])
    plt.title("Leading Edge Envelope")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [WU]")
    plt.grid()

    plt.subplot(212)
    plt.plot(times,x)
    plt.plot(times,env)
    plt.xlim(times[int(fs*(1.1-0.01))],times[int(fs*(1.1+0.01))])
    plt.title("Trailing Edge Envelope")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [WU]")
    plt.grid()

    plt.show()

def part2():
    fs = 48.E3
    f0 = 2.E3
    omega0 = f0 * 2*pi

    T = 1.
    N = int(T*fs)
    f = sigA.freqVec(N,fs)
    times = sigA.timeVec(N,fs)

    H = (1j*(f/f0))/(1+1j*(f/f0))

    LSP = np.append(H[:N/2],np.conjugate(H[:N/2][::-1]))
    LSP[N/2] = np.abs(LSP[N/2])
    LSP[0] = LSP[-1] = np.abs(LSP[0])
    '''
    plt.figure()
    plt.plot(f,np.real(LSP))
    plt.plot(f,np.imag(LSP))
    '''
    impulse = np.zeros(N)
    impulse[0] = fs             # = 1/dt

    b = np.array([1.,-1.])*(2*fs/omega0)
    a = np.array([1.+(2.*fs/omega0),1.-(2.*fs/omega0)])
    print b
    print a

    impResp = sig.lfilter(b,a,impulse)

    plt.figure()
    plt.plot(times,impResp)
    plt.xlim(0,0.001)
    plt.grid()
    plt.title("Impulse Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [WU]")

    freqResp = np.fft.fft(impResp)/fs
    plt.figure()
    plt.subplot(211)
    plt.semilogx(f,np.abs(freqResp),label="from Impulse Response")
    plt.semilogx(f,np.abs(LSP),label="from Formula")
    plt.xlim(5,50E3)
    plt.grid()
    plt.title("Filter Magnitude")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.legend()

    plt.subplot(212)
    plt.semilogx(f,np.angle(freqResp),label="from Impulse Response")
    plt.semilogx(f,np.angle(LSP),label="from Formula")
    plt.xlim(5,50E3)
    plt.title("Filter Phase")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [rad]")
    plt.grid()
    plt.legend()
    plt.show()

def part3A():

    file = "HWY_extract.wav"
    data,fs = sf.read(file)
    print np.shape(data)
    print fs

    mic1 = data[:,0]
    mic2 = data[:,1]

    N = len(mic1)
    sliceLength = int(0.2*fs)
    G_XX, freqAvg,delF,_, x_ms = sigA.spectroArray(mic1,fs,sliceLength,
                                                   overlap=0.5,winType="hann")
    G_YY,_,_,_,y_ms = sigA.spectroArray(mic2,fs,sliceLength
                                        ,overlap=0.5,winType="hann")
    G_XY,_,_,_ = sigA.crossSpectroArray(mic1,mic2,fs,sliceLength,
                                        overlap=0.5,winType="hann")

    #delF = sigA.param(N,fs,show=False)[1]
    ms_G_XX_avg = sum(G_XX)*delF
    ms_G_YY_avg = sum(G_YY)*delF

    print ms_G_XX_avg
    print np.mean(x_ms)

    print ms_G_YY_avg
    print np.mean(y_ms)

    plt.figure()
    plt.loglog(freqAvg,np.abs(G_XX),label=r"$\overline{G_{XX}}$")
    plt.loglog(freqAvg,np.abs(G_YY),label=r"$\overline{G_{YY}}$")
    plt.loglog(freqAvg,np.abs(G_XY),label=r"$\overline{G_{XY}}$")
    plt.xlim(100,10E3)
    plt.ylim(1E-10,1E-4)
    plt.title("Averaged Power Spectral Densities")
    plt.ylabel(r"Power Spectral Density [Pa/${Hz^2}$]")
    plt.xlabel("Frequency [Hz]")
    plt.legend()
    plt.grid()

    coh, freq = sigA.coherence(mic1,mic2,fs,sliceLength,overlap=0.5,winType="hann")

    plt.figure()
    plt.semilogx(freq, coh)
    plt.xlim(100,10E3)
    plt.ylim(0,1.2)
    plt.title("Coherence")
    plt.ylabel(r"$\gamma^2$")
    plt.xlabel("Frequency [Hz]")
    plt.grid()

    plt.show()

def part3B():
    file = "HWY_2mic_calA.wav"
    data,fs = sf.read(file)
    mic1 = data[:,0]
    mic2 = data[:,1]
    N = len(mic1)
    f0 = 1.E3
    Pa_rms = 10.
    times = sigA.timeVec(N,fs)

    b,a = sigA.constQ(f0,1,fs)

    mic2Filt = sig.lfilter(b,a,mic2)

    plt.figure()
    plt.plot(times,mic2,label="raw")
    plt.plot(times,mic2Filt,label="filtered")
    plt.title("Calibration Channel")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [WU]")

    plt.legend()
    plt.grid()
    plt.xlim(times[0],times[-1])


    cal = mic2Filt[int(fs*55):]
    times_cal = times[int(fs*55):]
    WU_pk = np.max(cal)

    plt.figure()
    plt.plot(times_cal,cal,color="green")
    plt.xlim(55,55+5E-3)
    plt.title("Calibration Tone")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [WU]")
    plt.axhline(WU_pk,color="black")
    plt.axhline(-WU_pk,color="black")
    plt.grid()

    WU_rms = WU_pk/np.sqrt(2)
    WU_rms2 = sigA.rms(cal,show=False)[0]

    print WU_rms
    print Pa_rms/WU_rms

    print WU_rms2
    print Pa_rms/WU_rms2

    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))