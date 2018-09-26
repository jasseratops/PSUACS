import numpy as np
import wave
from datetime import date, time, datetime, timedelta
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import sigA

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

    print "sampleRate: " + str(sampleRate)
    print "noOfBits: " + str(noOfBits)
    print "noOfSamples: " + str(noOfSamples)
    startTimeAn_EDT = timedelta(days=1,hours=5,minutes=20)
    startTimeAn_UTC = startTimeAn_EDT + timedelta(hours=4)                          # convert from EDT to UTC
    startTimeRec_UTC = timedelta(hours=20,minutes=47,seconds=38,milliseconds=441)   # set recording start time
    timeOffset = startTimeAn_UTC - startTimeRec_UTC                                 # Find time offset with relation to start of recording
    print "timeOffset: " + str(timeOffset)

    anStartInd = int(timeOffset.total_seconds()*sampleRate)
    print "Index at " + str(startTimeAn_EDT) + " EDT: " + str(anStartInd)
    nframes = int(10.*60.*sampleRate)



    # wavFileread(path)[chunkStart:chunkEnd]
    data = sf.read(file=path,frames=nframes,start=anStartInd)[0]
    t = np.linspace(0,nframes,nframes)

    print data
    print t

    for i in range(5):
        print "data[" +str(i) + "]:" + str(data[i])


    sliceLength = 10*sampleRate

    #sigA.spectrogram(data,sampleRate,sliceLength=sliceLength,overlap=0.5,scale=False,winType="hann")
    GxxAvg, freqAvg, _, _ = sigA.spectroArray(data,fs=sampleRate,sliceLength=sliceLength,sync=0,overlap=0.5,winType="hann")

    plt.figure()
    plt.semilogy(freqAvg,GxxAvg)
    plt.xlim(0,400)
    #plt.yscale("log")

    plt.figure()
    plt.plot(t,data)
    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))