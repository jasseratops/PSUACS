import numpy as np
import wave
from datetime import date, time, datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import soundfile as sf
import sigA
import scipy.signal as sig

def indexGen(startTimeAn_EDT,startTimeRec_UTC,fs):
    startTimeAn_UTC = startTimeAn_EDT + timedelta(hours=4)                          # convert from EDT to UTC
    timeOffset = startTimeAn_UTC - startTimeRec_UTC                                 # Find time offset with relation to start of recording
    print "timeOffset: " + str(timeOffset)
    anStartInd = int(timeOffset.total_seconds()*fs)
    print "Index at " + str(startTimeAn_EDT) + " EDT: " + str(anStartInd)
    return anStartInd

def main(args):
    file_name = "AACD_208_20_47_38.wav"
    path = file_name
    wavFile = wave.open(path)
    noOfSamples = wavFile.getnframes()
    sampleRate =  wavFile.getframerate()
    noOfBits = wavFile.getsampwidth()*8

    cnvrtr = ((2**23)*(1.585E-6)*(1)*(1))/((1)*(1)*(50E-3)*200)
    print "Conversion Rate: " + str(cnvrtr)

    print "sampleRate: " + str(sampleRate)
    print "noOfBits: " + str(noOfBits)
    print "noOfSamples: " + str(noOfSamples)
    startTimeRec_UTC = timedelta(hours=20, minutes=47, seconds=38, milliseconds=441)  # set recording start time

    specDatPeriod = 220.
    specDatStartTime = timedelta(days=1,hours=5,minutes=23,seconds=0)
    specDatInd = indexGen(startTimeAn_EDT=specDatStartTime,startTimeRec_UTC=startTimeRec_UTC,fs=sampleRate)
    specDatNframes = int(specDatPeriod*sampleRate)

    specDat = (sf.read(file=path,frames=specDatNframes,start=specDatInd)[0])*cnvrtr
    print specDatNframes

    frq = 205.
    plt.figure()
    sigA.spectrogram(specDat,sampleRate,sliceLength=1024,overlap=0.5,winType="hann")
    plt.ylim(0,400)
    plt.title("Spectrogram: 5:23:00EDT - 5:26:40EDT: " + str(frq))
    plt.xlabel("time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.savefig("spec"+str(frq)+".png")

    plt.figure()
    for f0 in [frq,frq-0.2,frq+0.2]:
        bw = 0.1
        f1 = f0 - (bw/2.)
        f2 = f0 + (bw/2.)

        f0w = sigA.fWarp(f0, sampleRate)
        f1w = sigA.fWarp(f1, sampleRate)
        f2w = sigA.fWarp(f2, sampleRate)

        Q=1000.
        b,a = sigA.constQ(f0w,Q,sampleRate)

        fltrd = sig.lfilter(b,a,specDat)
        Tc = 1.
        smoothed = sigA.expAvging((fltrd**2),sampleRate,Tc)
        timesSmoothed = sigA.timeVec(specDatNframes,sampleRate)

        plt.title("Exp. Averaging: 5:23:00EDT - 5:26:40EDT: " + str(frq))
        plt.plot(timesSmoothed,smoothed,label="f0: " + str(f0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xlabel("Time [s]")
        plt.ylabel("Power [Pa^2]")
        plt.xlim(timesSmoothed[0],timesSmoothed[-1])
        plt.savefig(str(frq)+".png")

    plt.legend()

    specDatPeriod = 220.
    specDatStartTime = timedelta(days=1,hours=5,minutes=25,seconds=0)
    specDatInd = indexGen(startTimeAn_EDT=specDatStartTime,startTimeRec_UTC=startTimeRec_UTC,fs=sampleRate)
    specDatNframes = int(specDatPeriod*sampleRate)

    specDat = (sf.read(file=path,frames=specDatNframes,start=specDatInd)[0])*cnvrtr
    print specDatNframes

    frq = 255.
    plt.figure()
    sigA.spectrogram(specDat,sampleRate,sliceLength=1024,overlap=0.5,winType="hann")
    plt.ylim(0,400)
    plt.title("Spectrogram: 5:25:00EDT - 5:28:40EDT: " + str(frq))
    plt.xlabel("time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.savefig("spec"+str(frq)+".png")

    plt.figure()
    for f0 in [frq,frq-0.2,frq+0.2]:
        bw = 0.1
        f1 = f0 - (bw/2.)
        f2 = f0 + (bw/2.)

        f0w = sigA.fWarp(f0, sampleRate)
        f1w = sigA.fWarp(f1, sampleRate)
        f2w = sigA.fWarp(f2, sampleRate)

        Q=1000.
        b,a = sigA.constQ(f0w,Q,sampleRate)

        fltrd = sig.lfilter(b,a,specDat)
        Tc = 1.
        smoothed = sigA.expAvging((fltrd**2),sampleRate,Tc)
        timesSmoothed = sigA.timeVec(specDatNframes,sampleRate)

        plt.title("Exp. Averaging: 5:25:00EDT - 5:28:40EDT: " + str(frq))
        plt.plot(timesSmoothed,smoothed,label="f0: " + str(f0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xlabel("Time [s]")
        plt.ylabel("Power [Pa^2]")
        plt.xlim(timesSmoothed[0],timesSmoothed[-1])
        plt.savefig(str(frq)+".png")
    plt.legend()
    
    specDatPeriod = 220.
    specDatStartTime = timedelta(days=1, hours=5, minutes=19, seconds=0)
    specDatInd = indexGen(startTimeAn_EDT=specDatStartTime, startTimeRec_UTC=startTimeRec_UTC, fs=sampleRate)
    specDatNframes = int(specDatPeriod * sampleRate)

    specDat = (sf.read(file=path, frames=specDatNframes, start=specDatInd)[0]) * cnvrtr
    print specDatNframes

    frq = 165.
    plt.figure()
    sigA.spectrogram(specDat, sampleRate, sliceLength=1024, overlap=0.5, winType="hann")
    plt.ylim(0, 400)
    plt.title("Spectrogram: 5:19:00EDT - 5:22:40EDT: " + str(frq))
    plt.xlabel("time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.savefig("spec"+str(frq)+".png")

    plt.figure()
    for f0 in [frq, frq - 0.2, frq + 0.2]:
        bw = 0.1
        f1 = f0 - (bw / 2.)
        f2 = f0 + (bw / 2.)

        f0w = sigA.fWarp(f0, sampleRate)
        f1w = sigA.fWarp(f1, sampleRate)
        f2w = sigA.fWarp(f2, sampleRate)

        Q = f0w / (f2w - f1w)
        b, a = sigA.constQ(f0w, Q, sampleRate)

        fltrd = sig.lfilter(b, a, specDat)
        Tc = 1.
        smoothed = sigA.expAvging((fltrd ** 2), sampleRate, Tc)
        timesSmoothed = sigA.timeVec(specDatNframes, sampleRate)

        plt.title("Exp. Averaging: 5:19:00EDT - 5:22:40EDT: " + str(frq))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(timesSmoothed, smoothed, label="f0: " + str(f0))
        plt.xlabel("Time [s]")
        plt.ylabel("Power [Pa^2]")
        plt.xlim(timesSmoothed[0], timesSmoothed[-1])
        plt.savefig(str(frq)+".png")

    plt.legend()
    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))