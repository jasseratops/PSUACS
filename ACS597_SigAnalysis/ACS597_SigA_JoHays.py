import numpy as np
import wave
from datetime import date, time, datetime, timedelta
import matplotlib.pyplot as plt
import sounddevice as sd
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
    #folder_name = "/Users/macbookpro/PycharmProjects/PSUACS/ACS597_SigAnalysis/Jo Hays/"
    #folder_name = "C:/Users/alshehrj/PycharmProjects/PSUACS/ACS597_SigAnalysis/"
    folder_name = ""

    file_name = "AACD_208_20_47_38.wav"
    path = folder_name+file_name
    wavFile = wave.open(path)
    noOfSamples = wavFile.getnframes()
    sampleRate =  wavFile.getframerate()
    noOfBits = wavFile.getsampwidth()*8*noOfSamples

    cnvrtr = ((2**23)*(1.585E-6)*(1)*(1))/((1)*(1)*(50E-3)*200)

    print "sampleRate: " + str(sampleRate)
    print "noOfBits: " + str(noOfBits)
    print "noOfSamples: " + str(noOfSamples)
    startTimeAn_EDT = timedelta(days=1,hours=5,minutes=20)
    startTimeRec_UTC = timedelta(hours=20, minutes=47, seconds=38, milliseconds=441)  # set recording start time

    nframes = int(10.*60.*sampleRate)

    anStartInd = indexGen(startTimeAn_EDT, startTimeRec_UTC, fs=sampleRate)

    # wavFileread(path)[chunkStart:chunkEnd]
    data = (sf.read(file=path,frames=nframes,start=anStartInd)[0])*cnvrtr
    t = np.linspace(0,nframes/sampleRate,nframes)

    print data
    print t

    for i in range(5):
        print "data[" +str(i) + "]:" + str(data[i])


    sliceLength = 10*sampleRate

    #sigA.spectrogram(data,sampleRate,sliceLength=sliceLength,overlap=0.5,scale=False,winType="hann")
    GxxAvg, freqAvg, _, _ = sigA.spectroArray(data,fs=sampleRate,sliceLength=sliceLength,sync=0,overlap=0.5,winType="hann")

    plt.figure()
    plt.plot(t,data)

    plt.figure()
    plt.semilogy(freqAvg,GxxAvg)
    plt.xlim(0,400)

    specDatStartTime = timedelta(days=1,hours=5,minutes=26,seconds=0)
    specDatInd = indexGen(startTimeAn_EDT=specDatStartTime,startTimeRec_UTC=startTimeRec_UTC,fs=sampleRate)
    specDatNframes = 8*sampleRate

    specDat = (sf.read(file=path,frames=specDatNframes,start=specDatInd)[0])*cnvrtr

    plt.figure()
    sigA.spectrogram(specDat,sampleRate,sliceLength=4096,overlap=0.5,winType="hann")

    #plt.yscale("log")


    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))