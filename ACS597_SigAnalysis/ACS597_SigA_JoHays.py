import numpy as np
import wave
from datetime import date, time, datetime, timedelta
import matplotlib.pyplot as plt
import soundfile as sf
import sigA

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
    startTimeAn_EDT = timedelta(days=1,hours=5,minutes=20)
    startTimeRec_UTC = timedelta(hours=20, minutes=47, seconds=38, milliseconds=441)  # set recording start time

    nframes = int(10.*60.*sampleRate)

    anStartInd = indexGen(startTimeAn_EDT, startTimeRec_UTC, fs=sampleRate)

    # wavFileread(path)[chunkStart:chunkEnd]
    data = sf.read(file=path,frames=nframes,start=anStartInd)[0]
    #for i in data: print i
    for i in range(5):
        print "data[" +str(i) + "]:" + str(data[i])

    data *= cnvrtr
    t = np.linspace(0,nframes/sampleRate,nframes)

    sliceLength = 10*sampleRate

    sigA.spectrogram(data,sampleRate,sliceLength=1024,overlap=0.75,scale=False,winType="hann")
    plt.title("Spectrogram: 5:20:00EDT-5:30:00EDT")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    GxxAvg, freqAvg, _, _ = sigA.spectroArray(data,fs=sampleRate,sliceLength=sliceLength,sync=0,overlap=0.5,winType="hann")

    plt.figure()
    plt.plot(t,data)
    plt.title("Waveform: 5:20:00EDT-5:30:00EDT")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [Pa]")
    plt.xlim(t[0],t[-1])

    plt.figure()
    plt.semilogy(freqAvg,GxxAvg)
    plt.title("Spectral Density: 5:20:00EDT-5:30:00EDT")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Spectral Density [${Pa^2}$/ Hz]")
    plt.xlim(0,400)

    specDatStartTime = timedelta(days=1,hours=5,minutes=24,seconds=0)
    specDatInd = indexGen(startTimeAn_EDT=specDatStartTime,startTimeRec_UTC=startTimeRec_UTC,fs=sampleRate)
    specDatNframes = int(10.*sampleRate)

    specDat = (sf.read(file=path,frames=specDatNframes,start=specDatInd)[0])*cnvrtr
    print specDatNframes

    plt.figure()
    sigA.spectrogram(specDat,sampleRate,sliceLength=1024,overlap=0.5,winType="hann")
    plt.xlim(0,8)
    plt.title("Spectrogram: 5:24:00EDT - 5:24:08EDT")
    plt.xlabel("time [s]")
    plt.ylabel("Frequency [Hz]")

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))