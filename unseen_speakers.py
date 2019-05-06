import logging
from glob import glob

import numpy as np
from natsort import natsorted

from constants import c
from speech_features import get_mfcc_features_390
from train import triplet_softmax_model
from utils import normalize, InputsGenerator
import random

logger = logging.getLogger(__name__)


def get_feat_from_audio(audio_reader, sr, norm_data, speaker):
    feat = get_mfcc_features_390(audio_reader, sr, max_frames=None)
    feat = normalize(feat, norm_data[speaker]['mean_train'], norm_data[speaker]['std_train'])
    return feat


def generate_features_for_unseen_speakers(audio_reader, target_speaker='p363'):
    assert target_speaker in audio_reader.all_speaker_ids
    # audio.metadata = dict()  # small cache <SPEAKER_ID -> SENTENCE_ID, filename>
    # audio.cache = dict()  # big cache <filename, data:audio librosa, blanks.>
    inputs_generator = InputsGenerator(cache_dir=audio_reader.cache_dir,
                                       audio_reader=audio_reader,
                                       max_count_per_class=1000)
    inputs = inputs_generator.generate_inputs_for_inference(target_speaker)
    return inputs


def random_line(afile):
    line = next(iter(afile))
    for num, aline in enumerate(afile, 2):
      if random.randrange(num): continue
      line = aline
    return line

import math
def  myinference_unseen_speakers(sp1, sp2):
    import pickle,glob
    sp1_feat = pickle.load(open('/run/media/rice/DATA/OUTPUT2/'+sp1+'.pkl', 'rb'))[sp1]
    sp2_feat = pickle.load(open('/run/media/rice/DATA/OUTPUT2/'+sp2+'.pkl', 'rb'))[sp2]
    files = glob.glob('/run/media/rice/DATA/OUTPUT2/*.pkl')
    file = random_line(files)
    print(file)
    randsp = file[len('/run/media/rice/DATA/OUTPUT2/'):-4]
    print(randsp)
    randsp_feat = pickle.load(open(file, 'rb'))[randsp]

    # batch_size => None (for inference).
    m = triplet_softmax_model(num_speakers_softmax=len(c.AUDIO.SPEAKERS_TRAINING_SET),
                              emb_trainable=False,
                              normalize_embeddings=True,
                              batch_size=None)

    checkpoints = natsorted(glob.glob('checkpoints/*.h5'))

    # compile_triplet_softmax_model(m, loss_on_softmax=False, loss_on_embeddings=False)
    print("m summary   "*20)
    print(m.summary())

    if len(checkpoints) != 0:
        checkpoint_file = checkpoints[-1]
        initial_epoch = int(checkpoint_file.split('/')[-1].split('.')[0].split('_')[-1])
        logger.info('Initial epoch is {}.'.format(initial_epoch))
        logger.info('Loading checkpoint: {}.'.format(checkpoint_file))
        m.load_weights(checkpoint_file)  # latest one.

    print("sp1_feat "*20)
    print(sp1_feat)
    print(type(sp1_feat))
    emb_sp1 = m.predict(np.vstack(sp1_feat))[0]
    emb_sp2 = m.predict(np.vstack(sp2_feat))[0]
    emb_randsp = m.predict(np.vstack(randsp_feat))[0]

    logger.info('Checking that L2 norm is 1.')
    logger.info(np.mean(np.linalg.norm(emb_sp1, axis=1)))
    logger.info(np.mean(np.linalg.norm(emb_sp2, axis=1)))
    logger.info(np.mean(np.linalg.norm(emb_randsp, axis=1)))

    from scipy.spatial.distance import cosine

    # note to myself:
    # embeddings are sigmoid-ed.
    # so they are between 0 and 1.
    # A hypersphere is defined on tanh.

    logger.info('Emb1.shape = {}'.format(emb_sp1.shape))
    logger.info('Emb2.shape = {}'.format(emb_sp2.shape))
    logger.info('Emb2.shape = {}'.format(emb_randsp.shape))

    emb1 = np.mean(emb_sp1, axis=0)
    emb2 = np.mean(emb_sp2, axis=0)
    emb_randsp = np.mean(emb_randsp,axis=0)


    same_sp1 = ''
    same_sp2 = ''
    same_sp_emb1 = emb1
    same_sp_emb2 = emb2
    most_same_emb1 = 1.0
    most_same_emb2 = 1.0
    files.remove('/run/media/rice/DATA/OUTPUT2/'+sp1+'.pkl')
    files.remove('/run/media/rice/DATA/OUTPUT2/'+sp2+'.pkl')
    for f in files:
        sp = f[len('/run/media/rice/DATA/OUTPUT2/'):-4]
        sp_feat = pickle.load(open(f, 'rb'))[sp]
        emb_sp = np.mean(m.predict(np.vstack(sp_feat))[0],axis=0)
        if cosine(emb1,emb_sp) < most_same_emb1:
            same_sp1 = f
            same_sp_emb1 = emb_sp
            most_same_emb1 = cosine(emb1,emb_sp)
        if cosine(emb2, emb_sp) < most_same_emb2:
            same_sp2 = f
            same_sp_emb2 = emb_sp
            most_same_emb2 = cosine(emb2, emb_sp)

    logger.info('Cosine = {}'.format(cosine(emb1, emb2)))
    logger.info('Cosine = {}'.format(most_same_emb1))
    logger.info('Cosine = {}'.format(most_same_emb2))

    cos1_2 = int(math.log10(cosine(emb1, emb2)))
    cos1_11 = int(math.log10(most_same_emb1))
    cos2_22 = int(math.log10(most_same_emb2))


    if cos1_2 > -4:
        print('they are diferent speakers')
    else:
        print('they are same speaker')
    if cos1_11 <= -4 :
        print(sp1,"is in the database")
    else:
        print(sp1,"is not in the database")
    if cos2_22 <= -4:
        print(sp2,"is in the database")
    else:
        print(sp2,"is not in the database")

def inference_unseen_speakers(audio_reader, sp1, sp2):
    sp1_feat = generate_features_for_unseen_speakers(audio_reader, target_speaker=sp1)
    sp2_feat = generate_features_for_unseen_speakers(audio_reader, target_speaker=sp2)

    # batch_size => None (for inference).
    m = triplet_softmax_model(num_speakers_softmax=len(c.AUDIO.SPEAKERS_TRAINING_SET),
                              emb_trainable=False,
                              normalize_embeddings=True,
                              batch_size=None)

    checkpoints = natsorted(glob('checkpoints/*.h5'))

    # compile_triplet_softmax_model(m, loss_on_softmax=False, loss_on_embeddings=False)
    print(m.summary())

    if len(checkpoints) != 0:
        checkpoint_file = checkpoints[-1]
        initial_epoch = int(checkpoint_file.split('/')[-1].split('.')[0].split('_')[-1])
        logger.info('Initial epoch is {}.'.format(initial_epoch))
        logger.info('Loading checkpoint: {}.'.format(checkpoint_file))
        m.load_weights(checkpoint_file)  # latest one.

    emb_sp1 = m.predict(np.vstack(sp1_feat))[0]
    emb_sp2 = m.predict(np.vstack(sp2_feat))[0]

    logger.info('Checking that L2 norm is 1.')
    logger.info(np.mean(np.linalg.norm(emb_sp1, axis=1)))
    logger.info(np.mean(np.linalg.norm(emb_sp2, axis=1)))

    from scipy.spatial.distance import cosine

    # note to myself:
    # embeddings are sigmoid-ed.
    # so they are between 0 and 1.
    # A hypersphere is defined on tanh.

    logger.info('Emb1.shape = {}'.format(emb_sp1.shape))
    logger.info('Emb2.shape = {}'.format(emb_sp2.shape))

    emb1 = np.mean(emb_sp1, axis=0)
    emb2 = np.mean(emb_sp2, axis=0)

    logger.info('Cosine = {}'.format(cosine(emb1, emb2)))
    logger.info('SAP = {}'.format( np.mean([cosine(u, v) for (u, v) in zip(emb_sp1[:-1], emb_sp1[1:])])))
    logger.info('SAN = {}'.format(np.mean([cosine(u, v) for (u, v) in zip(emb_sp1, emb_sp2)])))
    logger.info('We expect: SAP << SAN.')



def inference_embeddings(audio_reader, speaker_id):
    speaker_feat = generate_features_for_unseen_speakers(audio_reader, target_speaker=speaker_id)

    # batch_size => None (for inference).
    m = triplet_softmax_model(num_speakers_softmax=len(c.AUDIO.SPEAKERS_TRAINING_SET),
                              emb_trainable=False,
                              normalize_embeddings=True,
                              batch_size=None)

    checkpoints = natsorted(glob('checkpoints1/*.h5'))
    print(m.summary())

    if len(checkpoints) != 0:
        checkpoint_file = checkpoints[-1]
        initial_epoch = int(checkpoint_file.split('/')[-1].split('.')[0].split('_')[-1])
        logger.info('Initial epoch is {}.'.format(initial_epoch))
        logger.info('Loading checkpoint: {}.'.format(checkpoint_file))
        m.load_weights(checkpoint_file)  # latest one.

    emb_sp1 = m.predict(np.vstack(speaker_feat))[0]

    logger.info('Emb1.shape = {}'.format(emb_sp1.shape))

    np.set_printoptions(suppress=True)
    emb1 = np.mean(emb_sp1, axis=0)
