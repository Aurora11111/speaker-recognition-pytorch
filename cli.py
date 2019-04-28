import logging
import os
import shutil
import time
from argparse import ArgumentParser

from audio_reader import AudioReader
from constants import c
from utils import InputsGenerator

"""
python cli.py --regenerate_full_cache --multi_threading --cache_output_dir /run/media/rice/DATA/OUTPUT2/ --audio_dir /run/media/rice/DATA/tripletdata_test/
python cli.py --generate_training_inputs --multi_threading --cache_output_dir /run/media/rice/DATA/OUTPUT2/ --audio_dir /run/media/rice/DATA/tripletdata_test/


/run/media/rice/DATA/tripletdata_test/S0764/
python cli.py --unseen_speakers rawd1,rawd2 --audio_dir /run/media/rice/DATA/triplet_traindata/ --cache_output_dir /run/media/rice/DATA/OUTPUT/

python cli.py --update_cache --multi_threading --audio_dir /run/media/rice/DATA/tripletdata_test/E0002/ --cache_output_dir /run/media/rice/DATA/OUTPUT/
python cli.py --unseen_speakers E0002,S0002 --audio_dir  /run/media/rice/DATA/tripletdata_test/E0002/ --cache_output_dir /run/media/rice/DATA/OUTPUT/
python cli.py --unseen_speakers S0764,S0002 --audio_dir /run/media/rice/DATA/triplet_traindata/ --cache_output_dir /run/media/rice/DATA/OUTPUT/

python cli.py --unseen_speakers rawd1,S0002 --audio_dir  /run/media/rice/DATA/tripletdata_test/rawd1/ --cache_output_dir /run/media/rice/DATA/OUTPUT/

"""

def arg_parse():
    arg_p = ArgumentParser()
    # arg_p.add_argument('--audio_dir', action='store_true',default='/run/media/rice/DATA/TIMIT/') # /run/media/rice/DATA/traindata/
    # arg_p.add_argument('--cache_output_dir', action='store_true',default='/run/media/rice/DATA/OUTPUT2/audio_cache_pkl/') # /home/rice/PycharmProjects/deep-speaker-master/OUTPUT/
    # arg_p.add_argument('--regenerate_full_cache', action='store_true',default=False)
    arg_p.add_argument('--update_cache', action='store_true',default=False)
    arg_p.add_argument('--generate_training_inputs', action='store_true',default=False)
    arg_p.add_argument('--multi_threading', action='store_true',default=False)
    arg_p.add_argument('--unseen_speakers',default='1_3,2_4')  # A32_1,1_1 example.
    arg_p.add_argument('--get_embeddings',default=False)  # p225 example.
    return arg_p

def regenerate_full_cache(audio_reader, args):
    cache_output_dir = os.path.expanduser(args.cache_output_dir)
    print('The directory containing the cache is {}.'.format(cache_output_dir))
    print('Going to wipe out and regenerate the cache in 5 seconds. Ctrl+C to kill this script.')
    time.sleep(5)
    try:
        shutil.rmtree(cache_output_dir)
    except:
        pass
    os.makedirs(cache_output_dir)
    audio_reader.build_cache()


def generate_cache_from_training_inputs(audio_reader, args):
    cache_dir = os.path.expanduser(args.cache_output_dir)
    inputs_generator = InputsGenerator(cache_dir=cache_dir,
                                       audio_reader=audio_reader,
                                       max_count_per_class=1000,
                                       speakers_sub_list=None,
                                       multi_threading=args.multi_threading)
    inputs_generator.start_generation()


def main():
    args = arg_parse().parse_args()

    # audio_reader = AudioReader(input_audio_dir=args.audio_dir,
    #                            output_cache_dir=args.cache_output_dir,
    #                            sample_rate=c.AUDIO.SAMPLE_RATE,
    #                            multi_threading=args.multi_threading)
    #
    # if args.regenerate_full_cache:
    #     regenerate_full_cache(audio_reader, args)
    #     exit(1)
    #
    # if args.update_cache:
    #     audio_reader.build_cache()
    #     exit(1)
    #
    # if args.generate_training_inputs:
    #     generate_cache_from_training_inputs(audio_reader, args)
    #     exit(1)

    if args.unseen_speakers is not None:
        unseen_speakers = [x.strip() for x in args.unseen_speakers.split(',')]
        from unseen_speakers import myinference_unseen_speakers,inference_unseen_speakers
        myinference_unseen_speakers( unseen_speakers[0], unseen_speakers[1])
        exit(0)

    # if args.get_embeddings is not None:
    #     speaker_id = args.get_embeddings.strip()
    #     from unseen_speakers import inference_embeddings
    #     inference_embeddings(audio_reader, speaker_id)
    #     exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()

"""
command
"""