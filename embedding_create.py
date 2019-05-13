#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim


def test(model_path):
    sequence = []
    if hp.data.data_preprocessed:
        test_dataset = SpeakerDatasetTIMITPreprocessed()
    else:
        test_dataset = SpeakerDatasetTIMIT()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=hp.test.num_workers,
                             drop_last=True)

    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    device = torch.device(hp.device)
    count = 0
    embeddings = []
    devector = []
    for e in range(hp.test.epochs):
        print("hp.test.epochs",hp.test.epochs)
        for batch_id, mel_db_batch in enumerate(test_loader):
            #print("mel_db_batch.shape",batch_id,mel_db_batch.shape)   #(1,10,160,40)
            assert hp.test.M % 2 == 0
            test_batch = mel_db_batch
            test_batch = torch.reshape(test_batch, (
            hp.test.N * hp.test.M, test_batch.size(2), test_batch.size(3)))
            #print("test_batch.shape",test_batch.shape)    #(10,160,40)

            enrollment_embeddings = embedder_net(test_batch)
            #print("enrollment_embeddings.shape", enrollment_embeddings.shape)  # (10,256)
            # enrollment_embeddings = torch.reshape(enrollment_embeddings,(hp.test.N, hp.test.M, enrollment_embeddings.size(1)))

            embedding = enrollment_embeddings.detach().numpy()
            embeddings.append(embedding)
            #print('embedding.shape', type(embedding), embedding.shape)  # (10,256)

            devector = np.concatenate(embeddings, axis=0)
            count = count + 1
    np.save('/run/media/rice/DATA/speakerdvector.npy', devector)

if __name__ == "__main__":
    test(hp.model.model_path)
