import os
import librosa
import pysrt
from pysrt import SubRipTime
import sys
from pathlib import Path
import re
from datetime import timedelta
import chardet
from moviepy.editor import *

import numpy as np

if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name
from .ffmpeg import Transcode
from .log import logger

class Media:
  """
  The Media class represents a media file to be retrieved and analyzed
  """

  # Supported media formats
  FORMATS = ['.mkv', '.mp4', '.wmv', '.avi', '.flv']

  # The frequency of the generated audio
  FREQ = 16000

  # The number of coefficients to extract from the mfcc
  N_MFCC = 13

  # The number of samples in each mfcc coefficient
  HOP_LEN = 512.0

  # The length (seconds) of each item in the mfcc analysis
  LEN_MFCC = HOP_LEN/FREQ

  def __init__(self, filepath, subtitles=None, WPS=5):
      prefix, ext = os.path.splitext(filepath)
      if ext == '.srt':
        return self.from_srt(filepath)
      if ext == '.txt':
        return self.from_txt(filepath)
      if not ext:
        raise ValueError(f'Unknown file: "{filepath}"')
      if ext not in Media.FORMATS:
        raise ValueError(f'Filetype {ext} not supported: "{filepath}"')
      self.__subtitles = subtitles
      self.filepath = os.path.abspath(filepath)
      self.filename = os.path.basename(prefix)
      self.extension = ext
      self.offset = timedelta()
      self.WPS = WPS # Words (spoken) per second
  
  def from_srt(self, filepath):
    prefix, ext = os.path.splitext(filepath)
    if ext != 'srt':
      raise ValueError('Filetype must be .srt')
    prefix = os.path.basename(re.sub(r'\.\w\w$', '', prefix))
    dir = os.path.dirname(filepath)
    for f in os.listdir(dir):
      _, ext = os.path.splitext(f)
      if f.startswith(prefix) and ext in Media.FORMATS:
        return self.__init__(os.path.join(dir, f), subtitles=[filepath])
    raise ValueError(f'No media for subtitle: "{filepath}"')

  def from_txt(self, filepath):
    prefix, ext = os.path.splitext(filepath)
    if ext != '.txt':
      raise ValueError('Filetype must be .txt')
    prefix = os.path.basename(re.sub(r'\.\w\w$', '', prefix))
    dir = os.path.dirname(filepath)
    for f in os.listdir(dir):
      _, ext = os.path.splitext(f)
      if f.startswith(prefix) and ext in Media.FORMATS:
        return self.__init__(os.path.join(dir, f), subtitles=[filepath])
    raise ValueError(f'No media for subtitle: "{filepath}"')
  
  def subtitles(self):
    if self.__subtitles is not None:
      for s in self.__subtitles:
        yield(Text(self, s))
    else:
      dir = os.path.dirname(self.filepath)
      for f in os.listdir(dir):
        if f.endswith('.txt') and f.startswith(self.filename):
          yield(Text(self, os.path.join(dir, f)))

  def mfcc(self, duration=60*15, seek=True):
    transcode = Transcode(self.filepath, duration=duration, seek=seek)
    self.offset = transcode.start
    print('Transcoding...')
    transcode.run()
    y, sr = librosa.load(transcode.output, sr=Media.FREQ)
    self.mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_len=int(Media.HOP_LEN), n_mfcc=int(Media.N_MFCC))
    os.remove(transcode.output)
    return self.mfcc

class Subtitle:
  """
  Subtitle class represents a .srt file on the disk and provides the functionality to inspect and manipulate the contents
  """

  def __init__(self, media, path):
    self.media = media
    self.path = path
    self.subs = pysrt.open(self.path)

  def srt_to_transcript(self):
    filename, _ = os.path.splitext(self.path)
    subs = pysrt.open(self.path)
    with open(f'{filename}.txt', 'w+') as f:
        for sub in subs:
            f.write(f'{sub.text}\n')
  
  def labels(self, subs=None):
        if self.media.mfcc is None:
            raise RuntimeError("Must analyse mfcc before generating labels")
        samples = len(self.media.mfcc[0])
        labels = np.zeros(samples)
        for sub in self.subs if subs is None else subs:
            start = timeToPos(sub.start - self.offset())
            end = timeToPos(sub.end - self.offset())+1
            for i in range(start, end):
                if i >= 0 and i < len(labels):
                    labels[i] = 1

        return labels

  def _find_encoding(self):
      data = None
      with open(self.path, "rb") as f:
          data = f.read()
      det = chardet.detect(data)
      return det.get("encoding")


  def offset(self):
      d = self.media.offset
      hours, remainder = divmod(d.seconds, 3600)
      minutes, seconds = divmod(remainder, 60)
      return SubRipTime(
          hours=hours, minutes=minutes, seconds=seconds,
          milliseconds=d.microseconds/1000
      )


  def logloss(self, pred, actual, margin=12):
      blocks = secondsToBlocks(margin)
      logloss = np.ones(blocks*2)
      indices = np.ones(blocks*2)
      nonzero = np.nonzero(actual)[0]
      begin = max(nonzero[0]-blocks, 0)
      end = min(nonzero[-1]+blocks, len(actual)-1)
      pred = pred[begin:end]
      actual = actual[begin:end]
      for i, offset in enumerate(range(-blocks, blocks)):
          snippet = np.roll(actual, offset)
          try:
              logloss[i] = sklearn.metrics.log_loss(snippet[blocks:-blocks], pred[blocks:-blocks])
          except (ValueError, RuntimeWarning):
              pass
          indices[i] = offset

      return indices, logloss


  def sync(self, net, safe=True, margin=12, plot=True):
      secs = 0.0
      labels = self.labels()
      mfcc = self.media.mfcc.T
      mfcc = mfcc[..., np.newaxis]
      pred = net.predict(mfcc)
      x, y = self.logloss(pred, labels, margin=margin)
      accept = True
      if safe:
          mean = np.mean(y)
          sd = np.std(y)
          accept = np.min(y) < mean - sd
      if accept:
          secs = blocksToSeconds(x[np.argmin(y)])
          print("Shift {} seconds:".format(secs))
          self.subs.shift(seconds=secs)
          self.subs.save(self.path, encoding='utf-8')
          if secs != 0.0:
              logger.info('{}: {}s'.format(self.path, secs))
      if plot:
          self.plot_logloss(x, y)
      return secs


  def sync_all(self, net, margin=16, plot=True):
      secs = 0.0
      mfcc = self.media.mfcc.T
      mfcc = mfcc[..., np.newaxis]
      pred = net.predict(mfcc)
      print("Fitting...")
      self.__sync_all_rec(self.subs, pred)
      self.clean()
      self.subs.save(self.path, encoding='utf-8')


  def __sync_all_rec(self, subs, pred, margin=16):
      if len(subs) < 3:
          return
      labels = self.labels(subs=subs)
      if np.unique(labels).size <= 1:
          return
      x, y = self.logloss(pred, labels, margin=max(margin, 0.25))
      #self.plot_logloss(x,y)
      #self.plot_labels(labels, pred)
      secs = blocksToSeconds(x[np.argmin(y)])
      subs.shift(seconds=secs)
      # call recursively
      middle = subs[len(subs)//2]
      left = subs.slice(ends_before=middle.start)
      right = subs.slice(starts_after=middle.start)
      self.__sync_all_rec(left, pred, margin=margin/2)
      self.__sync_all_rec(right, pred, margin=margin/2)


  def clean(self):
      for i, s in enumerate(self.subs):
          if i >= len(self.subs)-1:
              return
          next = self.subs[i+1]
          if s.end > next.start:
              s.end = next.start



  def plot_logloss(self, x, y):
      import matplotlib.pyplot as plt
      plt.figure()
      plt.plot(x, y)
      plt.title('logloss over shifts')
      plt.ylabel('logloss')
      plt.xlabel('shifts')
      plt.legend(['logloss'], loc='upper left')
      plt.show()

  def plot_labels(self, labels, pred):
      import matplotlib.pyplot as plt
      plt.figure()
      plt.plot([i for i in range(0,len(labels))], labels, label='labels')
      plt.title('labels vs predictions')
      plt.ylabel('value')
      plt.xlabel('time')
      plt.legend(['labels'], loc='upper left')

      plt.figure()
      plt.plot([i for i in range(0,len(pred))], pred, label='pred')
      plt.title('labels vs predictions')
      plt.ylabel('value')
      plt.xlabel('time')
      plt.legend(['pred'], loc='upper left')
      plt.show()


class Text:
  """
  Text class reads .txt file and converts it to .srt
  """

  def __init__(self, media, path):
    self.media = media
    self.path = path
    self.lines = open(self.path)
  
  def determine_speech(self, model):
    clip = AudioFileClip(self.media)
    dur = clip.duration
    mfcc = self.media.mfcc.T
    mfcc = mfcc[..., np.newaxis]
    y_pred = model.predict(mfcc)
    y_pred = y_pred.reshape(-1,)
    num_chunks = round(len(y_pred)/dur)
    chunks = [ y_pred[i:i+num_chunks] for i in range(0, len(y_pred), num_chunks) ]
    self.__secs = [ round(sum(i)/len(i)) for i in chunks ]
    return self.__secs

  def to_srt(self):
    with open(self.path) as f:
        text = f.read()
        text = text.replace('\n\n', '\n').split('\n')

    with open(f'DATA/{self.media.filename}.srt', 'w+') as f:
        num = 1
        times = []

        for i, value in enumerate(self.__secs, start=0):
          if i > 1 and i < len(text):
              num_words = len(text[i-1].split(' '))
          if num_words > self.media.WPS:
              continue
          if value == 1:
              sec = i
              times.append(sec)

        for i, time in enumerate(times):
            num_words = len(text[i].split(' '))
            if num_words > self.media.WPS:
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


# Convert timestamp to seconds
def timeToSec(t):
    total_sec = float(t.milliseconds)/1000
    total_sec += t.seconds
    total_sec += t.minutes*60
    total_sec += t.hours*60*60
    return total_sec

# Return timestamp from cell position
def timeToPos(t, freq=Media.FREQ, hop_len=Media.HOP_LEN):
    return round(timeToSec(t)/(hop_len/freq))


def secondsToBlocks(s, hop_len=Media.HOP_LEN, freq=Media.FREQ):
    return int(float(s)/(hop_len/freq))


def blocksToSeconds(h, freq=Media.FREQ, hop_len=Media.HOP_LEN):
    return float(h)*(hop_len/freq)
