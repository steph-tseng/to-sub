# %%
from pytube import YouTube
import os
import warnings
warnings.filterwarnings('ignore')

# Dictionary of links to Youtube videos to be downloaded
links = {'1MythicalKitchen': 'https://www.youtube.com/watch?v=OfrMqoyOJVE', 
         '2GameTheorists': 'https://www.youtube.com/watch?v=Ow3bMlScrzs&t=93s', 
         '3LastWeekTonight': 'https://www.youtube.com/watch?v=abuQk-6M5R4&t=215s', 
         '4LateShow': 'https://www.youtube.com/watch?v=kwcy6nLaguY'}

# Generate data from links
def generate_test_data(links=links):
  DATA_DIR = 'DATA'
  if os.path.isdir(DATA_DIR) != True:
    os.mkdir(DATA_DIR) # Create the DATA folder if there isn't one already
  for link in links.keys():
    yt = YouTube(link) # Link to the Youtube video
    t = yt.streams.filter(only_audio=True) # Filter for just audio, remove .filter for whole video
    t[0].download('DATA')

    print('Caption options: ', yt.captions) # To get the language codes
    try:
      if yt.captions['en']:
        caption = yt.captions['en']
      elif yt.captions['a.en']:
        caption = yt.captions['a.en']
      else:
        caption = yt.captions['en.ehkg1hFWq8A']
    except:
      print('Please check caption options above')

    f = open(f"{os.path.join(DATA_DIR, yt.title)}.srt", "w")
    f.write(caption.generate_srt_captions())

# All of the videos are being downloaded double the length they should be
def download_videos_and_srt(link):
  DATA_DIR = 'DATA'
  if os.path.isdir(DATA_DIR) != True:
    os.mkdir(DATA_DIR) # Create the DATA folder if there isn't one already

  yt = YouTube(link) # Link to the Youtube video
  t = yt.streams #.filter() # Filter for just audio, remove .filter for whole video
  t[0].download('DATA')

  print('Caption options: ', yt.captions) # To get the language codes
  try:
    if yt.captions['en']:
      caption = yt.captions['en']
    elif yt.captions['a.en']:
      caption = yt.captions['a.en']
    else:
      caption = yt.captions['en.ehkg1hFWq8A']
  except:
    print('Please check caption options above')

  f = open(f"{os.path.join(DATA_DIR, yt.title)}.srt", "w")
  f.write(caption.generate_srt_captions())

download_videos_and_srt('https://www.youtube.com/watch?v=2xlol-SNQRU')
# %%
