import json
import glob
import io
data = {
  "AUDIO": {
    "SAMPLE_RATE": 8000,
    "SPEAKERS_TRAINING_SET": [
    ],
    "SPEAKERS_TESTING_SET": [
    ]
  }
}
conf_jsonpath = '/home/rice/PycharmProjects/deep-speaker-master/conf.json'

speakerjson = data

train_speakerids = glob.glob('/run/media/rice/DATA/tripletdata_test/*')
B = len('/run/media/rice/DATA/tripletdata_test/')
print(train_speakerids)
for train_speaker in train_speakerids:
    speakerjson["AUDIO"]["SPEAKERS_TESTING_SET"].append(train_speaker[B:])
test = json.dumps(speakerjson)

with io.open(conf_jsonpath, 'w', encoding='utf8') as outfile:
    str_ = json.dumps(speakerjson,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(str_)


speakerjson = data
train_speakerids = glob.glob('/run/media/rice/DATA/TIMIT/*')
A = len('/run/media/rice/DATA/TIMIT/')
print(train_speakerids)
for train_speaker in train_speakerids:
    speakerjson["AUDIO"]["SPEAKERS_TRAINING_SET"].append(train_speaker[A:])
test = json.dumps(speakerjson)

with io.open(conf_jsonpath, 'w', encoding='utf8') as outfile:
    str_ = json.dumps(speakerjson,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(str_)