import glob
import os
from shutil import copyfile
import subprocess
import numpy as np
import torch

newpath = '/run/media/rice/DATA/devector_data/'
speakers = set([""])

# if os.path.isfile(newpath):
#     print("yes")
# else:
#     os.mkdir(newpath)
"""
data_thchs30
"""
def getdata(path,speakers):
    length = len(path)
    files = glob.glob(path+'*/*.*')
    for file in files:
        print(file,type(file))
        speaker_name = file[length:].split('/')[0]
        speaker_folder = newpath + speaker_name
        if speaker_name in speakers:
            path, dirs, files = next(os.walk(speaker_folder+"/"))
            file_count = len(files)
            if file_count < 20:
                copyfile(file,speaker_folder+"/"+file[length:].split('/')[1])
        else:
            speakers.add(speaker_name)
            os.mkdir(speaker_folder)
    return speakers

def test_data(path):
    folders = glob.glob(path + '/*')
    for folder in folders:
        wavs = glob.glob((folder+'/*'))
        for wav in wavs:
            if (wav[-4:] != '.wav') & (wav[-4:] != '.WAV'):
                print(wav)

def WAVtowav(path):
    folders = glob.glob(path+'/*')
    for folder in folders:
        wavs = glob.glob((folder+'/*'))
        for wav in wavs:
            if (wav[-4:] == '.WAV'):
                new_wav = wav[:-4] + '.wav'
                print("new_wav " * 20)
                print(new_wav)
                if not os.path.exists(new_wav):
                    subprocess.call(['ffmpeg', '-i', wav, wav[:-4] + '.wav'])
                os.remove(wav)
def WAV_wav(path):
    wavs = glob.glob((path + '*'))
    for wav in wavs:
        if (wav[-4:] == '.WAV'):
            new_wav = wav[:-4]+'.wav'
            print("new_wav "*20)
            print(new_wav)
            if not os.path.exists(new_wav):
                subprocess.call(['ffmpeg', '-i', wav, wav[:-4] + '.wav'])
            os.remove(wav)

if __name__ == '__main__':
    #speakers = getdata('/run/media/rice/DATA/VCTK-Corpus/wav48/',speakers)
    #speakers = getdata('/run/media/rice/DATA/dataset/data_aishell/wav/dev/',speakers)
    #speakers = getdata('/run/media/rice/DATA/dataset/data_aishell/wav/test/', speakers)
    #WAV_wav('/home/rice/Desktop/audio_test/test/')

   WAVtowav('/run/media/rice/DATA/TIMIT/')