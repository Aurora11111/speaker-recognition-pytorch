#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import librosa
import numpy as np
from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
import torch
import pandas as pd
import pickle

audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))
model_path = hp.model.model_path
embedder_net = SpeechEmbedder()
embedder_net.load_state_dict(torch.load(model_path))
embedder_net.eval()


def save_traindevector():

    print("start text independent utterance feature extraction")
    os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(hp.data.test_path, exist_ok=True)    # make folder to save test file

    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    total_speaker_num = len(audio_path)
    train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("total speaker number : %d"%total_speaker_num)
    print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))

    speaker_dict = {}
    # speaker_dict['speaker_id'] = []
    # speaker_dict['data'] = []
    max = 0
    min = 100000
    count = 0
    for i, folder in enumerate(audio_path):
        print("%dth speaker processing..."%i)
        utterances_spec = []
        speakers = os.listdir(folder)
        print(folder)
        print(speakers)
        A = len('/run/media/rice/DATA/TIMIT/')
        speaker_name = folder[A:]
        print(speaker_name)
        for utter_name in speakers:
            utter_path = os.path.join(folder, utter_name)         # path of each utterance
            utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
            intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection

            for interval in intervals:
                if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                    utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                    S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                    utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
                    utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance
        # if len(utterances_spec) > max:
        #     max = len(utterances_spec)
        # if len(utterances_spec) < min:
        #     min = len(utterances_spec)
        # if len(utterances_spec) < 5:
        #     continue
        utterances_spec = np.array(utterances_spec)

        utter_index = np.random.randint(0, utterances_spec.shape[0], 20)  # select M utterances per speaker
        utterance = utterances_spec[utter_index]
        utterance = utterance[:, :, :160]  # (10,40,160) TODO implement variable length batch size
        utterance = torch.tensor(np.transpose(utterance, axes=(0, 2, 1)))  # transpose [batch, frames, n_mels]

        enrollment_embeddings = embedder_net(utterance)
        embedding = enrollment_embeddings.detach().numpy()


        # if i<train_speaker_num:      # save spectrogram as numpy file
        #     # train_x.append(embedding)
        #     # trainx_devector = np.concatenate(train_x, axis=0)
        #     # print(utter_name)
        #     if utter_name not in train_speaker_dict:
        #         train_speaker_dict[utter_name] = []
        #     train_speaker_dict[utter_name].append(embedding)
        #
        # else:
        #     # test_x.append(embedding)
        #     # testx_decector = np.concatenate(test_x, axis=0)
        #     if utter_name not in test_speaker_dict:
        #         test_speaker_dict[utter_name] = []
        #         test_speaker_dict[utter_name].append(embedding)

        speaker_dict[speaker_name] = embedding
        print(count)
        count += 1
        # speaker_dict['speaker_id'].append(utter_name[:-4])
        # speaker_dict['data'].append(embedding)
        print(speaker_dict.keys(),len(speaker_dict))
    with open('/run/media/rice/DATA/data.pkl', 'wb') as w:
        pickle.dump(obj=speaker_dict, file=w)


def save_testdevector(path):
    print(len(path))
    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    speaker_dict = {}
    max = 0
    min = 10000

    for utter_name in path:
        audios = glob.glob(utter_name + '/*')
        A = len('/run/media/rice/DATA/mixtest/')
        speaker_name = utter_name[A:]
        print(speaker_name)
        utterances_spec = []
        speaker_dict = {}
        for utter_path in audios:

            utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
            intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection
            for interval in intervals:
                if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                    utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                    S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                    utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
                    utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance
        # if len(utterances_spec) > max:
        #     max = len(utterances_spec)
        # if len(utterances_spec) < min:
        #     min = len(utterances_spec)
        # if len(utterances_spec) < 5:
        #     continue
        utterances_spec = np.array(utterances_spec)

        utter_index = np.random.randint(0, utterances_spec.shape[0], 20)  # select M utterances per speaker
        utterance = utterances_spec[utter_index]
        utterance = utterance[:, :, :160]  # (10,40,160) TODO implement variable length batch size
        utterance = torch.tensor(np.transpose(utterance, axes=(0, 2, 1)))  # transpose [batch, frames, n_mels]

        enrollment_embeddings = embedder_net(utterance)
        embedding = enrollment_embeddings.detach().numpy()

        speaker_dict[speaker_name] = embedding
        print(speaker_dict.keys(),len(speaker_dict))
        with open('/run/media/rice/DATA/OUTPUT2/'+speaker_name+'.pkl', 'wb') as w:
            pickle.dump(obj=speaker_dict, file=w)


if __name__ == "__main__":
    #save_traindevector()
    #save_testdevector(glob.glob(os.path.dirname('/run/media/rice/DATA/TIMIT/*/*.*')))
    save_testdevector(glob.glob(os.path.dirname('/run/media/rice/DATA/mixtest/*/*.*')))

