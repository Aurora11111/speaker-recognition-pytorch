#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
import torch
from torch.utils.data import DataLoader

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
import librosa
import numpy as np
predict_folder = './predict'

def test(model_path):
    utterances_spec = []
    for utter_name in os.listdir(predict_folder):
        print(utter_name)
        # print(utter_name)
        if utter_name[-4:] == '.wav':
            utter_path = os.path.join(predict_folder, utter_name)  # path of each utterance
            utter, sr = librosa.core.load(utter_path, hp.data.sr)  # load utterance audio
            intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection
            utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr  # lower bound of utterance length
            for interval in intervals:
                if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficient long,
                    utter_part = utter[interval[0]:interval[1]]  # save first and last 180 frames of spectrogram.
                    S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                          win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of utterances
                    utterances_spec.append(S[:, :hp.data.tisv_frame])  # first 180 frames of partial utterance
                    utterances_spec.append(S[:, -hp.data.tisv_frame:])  # last 180 frames of partial utterance

    utterances_spec = np.array(utterances_spec)

#    np.save(os.path.join(hp.data.train_path, "speaker.npy"))

    test_loader = utterances_spec

    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    avg_EER = 0
    device = torch.device(hp.device)
    avg_EER = 0

    predict_loader = utterances_spec

    enrollment_batch, verification_batch = torch.split(predict_loader, int(predict_loader.size(1) / 2), dim=1)
    enrollment_batch = torch.reshape(enrollment_batch, (
    hp.test.N * hp.test.M // 2, enrollment_batch.size(2), enrollment_batch.size(3)))
    verification_batch = torch.reshape(verification_batch, (
    hp.test.N * hp.test.M // 2, verification_batch.size(2), verification_batch.size(3)))

    perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
    unperm = list(perm)

    for i, j in enumerate(perm):
        unperm[j] = i

        verification_batch = verification_batch[perm]
        enrollment_embeddings = embedder_net(enrollment_batch)
        verification_embeddings = embedder_net(verification_batch)
        verification_embeddings = verification_embeddings[unperm]

        enrollment_embeddings = torch.reshape(enrollment_embeddings,
                                                  (hp.test.N, hp.test.M // 2, enrollment_embeddings.size(1)))
        verification_embeddings = torch.reshape(verification_embeddings,
                                                    (hp.test.N, hp.test.M // 2, verification_embeddings.size(1)))

        enrollment_centroids = get_centroids(enrollment_embeddings)

        sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

        diff = 1;
        EER = 0;
        EER_thresh = 0;
        EER_FAR = 0;
        EER_FRR = 0

        for thres in [0.01 * i + 0.5 for i in range(50)]:
            sim_matrix_thresh = sim_matrix > thres

            FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                        range(int(hp.test.N))])
                    / (hp.test.N - 1.0) / (float(hp.test.M / 2)) / hp.test.N)

            FRR = (sum([hp.test.M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in range(int(hp.test.N))])
                    / (float(hp.test.M / 2)) / hp.test.N)


            if diff > abs(FAR - FRR):
                diff = abs(FAR - FRR)
                EER = (FAR + FRR) / 2
                EER_thresh = thres
                EER_FAR = FAR
                EER_FRR = FRR
            avg_EER += EER
            print(
                "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))

    print("\n EER across {0} epochs: {:.4f}".format(avg_EER))


if __name__ == "__main__":
    test(hp.model.model_path)