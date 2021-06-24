from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
from pathlib import Path
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name
from .media import *
from .log import logger, init_logger
from tensorflow.keras.models import load_model

def run():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('media', metavar='MEDIA', type=str, nargs='+', help='media for which to synchronize subtitles')

  args = parser.parse_args()

  media = [Media(m) for m in args.media if m]
  model = load_model('model/out/ann.hdf5')
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

  for m in media:
    m.mfcc()
  for s in m.subtitles():
    s.determine_speech(model)
    s.to_srt()

