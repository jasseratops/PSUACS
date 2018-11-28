import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import sigA
import matplotlib.pyplot as plt
import soundfile as sf
from datetime import date, time, datetime, timedelta
import sounddevice as sd
import scipy.signal as sig

fileName = "Two_mic_L_x16_dec4.wav"
def main(args):
    fs = 12000.
    scale = 16

    startTime = timedelta(days=0,hours=0,minutes=4,seconds=48)
    endTime = timedelta(days=0,hours=0,minutes=5,seconds=2)

    duration = endTime-startTime
    print duration

    startInd = int(startTime.total_seconds()*fs)
    noOfFrames = int(duration.total_seconds()*fs)

    flyover = sf.read(file=fileName,frames=noOfFrames,start=startInd)[0]

    CPA_time = 6.9
    CPA_ind = int(CPA_time*fs)

    tt = sigA.timeVec(len(flyover[CPA_ind:]),fs)
    tt_neg = sigA.timeVec(len(flyover[:CPA_ind]),fs)
    tt_neg = np.flip(tt_neg,0)*-1.
    tt = np.concatenate((tt_neg,tt))
    print tt

    times = sigA.timeVec(len(flyover),fs)

    print flyover
    c = 343.

    f_lo = 76.
    f_hi = 106.
    V = c*(f_hi-f_lo)/(f_hi+f_lo)
    print "V: " + str(V)

    f1 = 200.
    f2 = 174.
    t1 = 6.7
    t2 = 7.

    df_dt = (f2-f1)/(t2-t1)
    print "df_dt: " + str(df_dt)
    f0 = 181
    h = (-(V**2)*f0)/(c*(df_dt))
    print "h: " + str(h)

    h=15.
    V=50.

    sinThet = (V*tt)/(np.sqrt((h**2)+((V*tt)**2)))
    func_2nd = f0*(1./(1+(V*sinThet/c)))
    func_1st = func_2nd/2
    func_3rd = func_1st*3
    fshaft = func_1st/2
    #sd.play(flyover,fs,blocking=False)

    plt.figure()
    sigA.spectrogram(flyover,fs,sliceLength=4096,overlap=0.9,winType="hann")
    plt.ylim(0,350)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("PA 28 Flyover Spectrogram")
    plt.savefig("CPA1.png")

    plt.figure()
    sigA.spectrogram(flyover,fs,sliceLength=4096,overlap=0.9,winType="hann")
    plt.ylim(0,350)
    plt.xlim(times[0],times[-1])
    plt.plot(times,func_2nd,color="black")
    plt.plot(times,func_1st,color="black")
    plt.plot(times,func_3rd,color="black")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("PA 28 Flyover Spectrogram, Freq. Functions")
    plt.savefig("CPA2.png")

    plt.figure()
    sigA.spectrogram(flyover,fs,sliceLength=4096,overlap=0.9,winType="hann")
    plt.ylim(0,350)
    plt.xlim(times[0],times[-1])
    plt.plot(times,fshaft,color="black")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("PA 28 Flyover Spectrogram, Shaft")
    plt.savefig("CPA3.png")

    b,a = sigA.octFilter(400.,fs)
    w,h = sig.freqz(b,a)

    freqs = w*fs/(2*pi)
    mag = 20*np.log10(abs(h))

    f_a = 62.5
    f_b = 125.
    f_c = 250.

    b_a,a_a = sigA.octFilter(f_a,fs)
    b_b,a_b = sigA.octFilter(f_b,fs)
    b_c,a_c = sigA.octFilter(f_c,fs)

    flyover_a = sig.lfilter(b_a,a_a,flyover)
    flyover_b = sig.lfilter(b_b,a_b,flyover)
    flyover_c = sig.lfilter(b_c,a_c,flyover)

    avg_a = sigA.expAvging(flyover_a**2,fs,timeConst=0.125)[::100]
    avg_b = sigA.expAvging(flyover_b**2,fs,timeConst=0.125)[::100]
    avg_c = sigA.expAvging(flyover_c**2,fs,timeConst=0.125)[::100]


    plt.figure()
    plt.plot(times[::100],avg_a,label="62.5Hz")
    plt.plot(times[::100],avg_b,label="125Hz")
    plt.plot(times[::100],avg_c,label="250Hz")
    plt.title("Filtered")
    plt.xlabel("Time [s]")
    plt.ylabel(r"Amplitude [$Pa^2$]")
    plt.xlim(times[0],times[-1])
    plt.legend()
    plt.savefig("CPA4.png")

    p_ref = 20.E-6

    plt.figure()
    plt.plot(times[::100], 20*np.log10(avg_a/p_ref), label="62.5Hz")
    plt.plot(times[::100], 20*np.log10(avg_b/p_ref), label="125Hz")
    plt.plot(times[::100], 20*np.log10(avg_c/p_ref), label="250Hz")
    plt.title("Filtered, in dB")
    plt.xlabel("Time [s]")
    plt.ylabel(r"Amplitude [dB]")
    plt.xlim(times[0],times[-1])
    plt.legend()
    plt.savefig("CPA5.png")

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))