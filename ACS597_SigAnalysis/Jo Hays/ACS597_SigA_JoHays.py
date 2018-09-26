import numpy as np
import wave
from datetime import date, time, datetime, timedelta

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

    startTime = timedelta(hours=20,minutes=47,seconds=38,milliseconds=441)
    endTime = timedelta(days=1,hours=9,minutes=20,seconds=0)
    timeOffset = endTime - startTime
    print timeOffset

    chunkStart = timeOffset.total_seconds()*sampleRate
    print chunkStart
    chunkEnd = chunkStart + (10.*sampleRate)

    # wavFileread(path)[chunkStart:chunkEnd]






if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))