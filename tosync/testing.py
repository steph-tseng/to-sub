#%%
import pickle
from moviepy.editor import *
import pysrt
import sys
from pathlib import Path
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name
from .ffmpeg import Transcode
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# path = '/Users/steph/ML_final/tosync2/subsync/subsync/model/dataset/dataset.pickle'
# with open(path, 'rb') as f:
#     X, Y = pickle.load(f) 

# X, Y = X[0], Y[0]
# print('X', X.shape)
# print('Y', Y.shape)

model = load_model('out/ann.hdf5')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

FREQ = 16000

# The number of coefficients to extract from the mfcc
N_MFCC = 13

# The number of samples in each mfcc coefficient
HOP_LEN = 512.0

# The length (seconds) of each item in the mfcc analysis
LEN_MFCC = HOP_LEN/FREQ

filepath = '/Users/steph/ML_final/tosync2/to-sub/DATA/v1.wav'
transcode = Transcode(filepath)
offset = transcode.start
print('Transcoding...')
transcode.run()
y, sr = librosa.load(transcode.output, sr=FREQ)
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(HOP_LEN), n_mfcc=int(N_MFCC))
os.remove(transcode.output)

mfcc = mfcc.T
mfcc = mfcc[..., np.newaxis]
Y = model.predict(mfcc)
Y = Y.reshape(-1,)

clip = AudioFileClip('/Users/steph/ML_final/tosync2/to-sub/DATA/v1.wav')
dur = clip.duration
num_chunks = round(len(Y)/dur)
chunks = [ Y[i:i+num_chunks] for i in range(0, len(Y), num_chunks) ]
in_secs = [ round(sum(i)/len(i)) for i in chunks ]

path = '/Users/steph/ML_final/tosync2/to-sub/DATA/v1.txt'

with open(path) as f:
    text = f.read()
    text = text.replace('\n\n', '\n').split('\n')

WPS = 5 # Words per second

with open('DATA/v1_test.srt', 'w+') as f:
    num = 1
    times = []

    for i, value in enumerate(in_secs, start=0):
        if i > 1 and i < len(text):
            num_words = len(text[i-1].split(' '))
            if num_words > 5:
                continue
        if value == 1:
            sec = i
            times.append(sec)

    for i, time in enumerate(times):
        num_words = len(text[i].split(' '))
        if num_words > 5:
            add = 2
        else:
            add = 1
        if not text[i+1]:
            break
        if time > 3600:
            hours = time // 3600
        else:
            hours = 0
        mins = (time - hours*3600) // 60
        secs = (time - hours*3600) % 60
        print(f'{num}\n{hours:02}:{mins:02}:{secs:02},000 --> {hours:02}:{mins:02}:{secs+add:02},000\n{text[i]}\n\n')
        f.write(f'{num}\n{hours:02}:{mins:02}:{secs:02},000 --> {hours:02}:{mins:02}:{secs+add:02},000\n{text[i]}\n\n')
        num += 1    


#%%
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

ffmpeg_extract_subclip("./model/DATA/v1.wav", 0, 1003, targetname="./model/DATA/cut_v1.wav")

#%%
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import re

DATA_DIR = 'DATA'
files = os.listdir(DATA_DIR)
p = re.compile('.*mp4')
videos = [ f for f in files if p.match(f) ]
for video in videos:
    filepath = os.path.join(DATA_DIR, video)
    clip = VideoFileClip(filepath)
    dur = clip.duration
    ffmpeg_extract_subclip(filepath, 0, (dur/2), targetname="./model/DATA/cut_v1.wav")


# %%
import sys
from pathlib import Path
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name
from .media import Media

# media = [Media(m) for m in args.media if m]
media = [Media('./subsync/subsync/model/DATA/v1.mp4')]

from .net import NeuralNet
from tensorflow.keras.models import load_model
model = load_model('/Users/steph/ML_final/tosync2/subsync/subsync/out/ann.hdf5')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

for m in media:
    m.mfcc(duration=0, seek=False)
for s in m.subtitles():
    s.sync_all(model, plot=False, margin=12)
# %%
import pysrt
path = '/Users/steph/ML_final/tosync2/DATA/v1.srt'

def srt_to_transcript(filepath):
    filename, ext = os.path.splitext(filepath)
    subs = pysrt.open(filepath)
    with open(f'{filename}.txt', 'w+') as f:
        for sub in subs:
            f.write(f'{sub.text}\n')

srt_to_transcript(path)
# %%
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
model = load_model('model/out/ann.hdf5')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
plot_model(model, to_file='model.png')
# %%
import sys
from pathlib import Path
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name
from .media import *
from tensorflow.keras.models import load_model

sample_data = ['/Users/steph/ML_final/tosync2/tosync/DATA/v1.txt', '/Users/steph/ML_final/tosync2/tosync/DATA/v1.mp4']
media = [Media(m) for m in sample_data]
model = load_model('/Users/steph/ML_final/tosync2/tosync/out/ann.hdf5')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

for m in media:
  m.mfcc()
  for s in m.subtitles():
    s.determine_speech(model)
    s.to_srt()
# %%
