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

model = load_model('out/ann.hdf5')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

FREQ = 16000

# The number of coefficients to extract from the mfcc
N_MFCC = 13

# The number of samples in each mfcc coefficient
HOP_LEN = 512.0

# The length (seconds) of each item in the mfcc analysis
LEN_MFCC = HOP_LEN/FREQ

filepath = 'DATA/v1.mp4'
transcode = Transcode(filepath)
offset = transcode.start
print('Transcoding...')
transcode.run()
y, sr = librosa.load(transcode.output, sr=FREQ)
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(HOP_LEN), n_mfcc=int(N_MFCC))
clip = AudioFileClip(transcode.output)
dur = clip.duration
os.remove(transcode.output)

mfcc = mfcc.T
mfcc = mfcc[..., np.newaxis]
Y = model.predict(mfcc)
Y = Y.reshape(-1,)

num_chunks = round(len(Y)/dur)
chunks = [ Y[i:i+num_chunks] for i in range(0, len(Y), num_chunks) ]
in_secs = [ round(sum(i)/len(i)) for i in chunks ]

path = 'DATA/v1.txt'

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
# %%
