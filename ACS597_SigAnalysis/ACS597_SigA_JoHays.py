import numpy as np
import wave
from datetime import date, time, datetime, timedelta
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import sigA

def main(args):
    folder_name = "/Users/macbookpro/PycharmProjects/PSUACS/ACS597_SigAnalysis/Jo Hays/"

    file_name = "AACD_208_20_47_38.wav"
    path = folder_name+file_name
    wavFile = wave.open(path)
    fileSize = wavFile.getnframes()
    sampleRate =  wavFile.getframerate()
    noOfSamples = wavFile.getsampwidth()*8*fileSize

    print sampleRate
    print noOfSamples
    print fileSize

    startTime = timedelta(days=0,hours=20,minutes=47,seconds=38,milliseconds=441)
    endTime = timedelta(days=1,hours=9,minutes=20,seconds=0)
    timeOffset = endTime - startTime
    print timeOffset

    chunkStart = int(timeOffset.total_seconds()*sampleRate)
    print chunkStart
    nframes = int(10.*60.*sampleRate)
    print nframes


    # wavFileread(path)[chunkStart:chunkEnd]
    data = sf.read(file=path,frames=nframes,start=chunkStart)[0]
    t = np.linspace(0,nframes,nframes)

    print data
    print t

    sliceLength = 1000

    sigA.spectrogram(data,sampleRate,sliceLength=sliceLength,overlap=0.5,scale=False,winType="hann")
    plt.ylim(0,400)
    plt.yscale("log")
    plt.figure()
    plt.plot(t,data)
    plt.show()






if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))