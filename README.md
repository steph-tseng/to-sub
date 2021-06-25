# Automatic Subtitle Placement

This contains code that allows users to download videos (or just audio) from youtube as well as their respective transcripts for training data. The model itself is able to place time stamps onto a transcript for any given video.

## Features

- Machine learning model for voice activity detection (_not recognition_)
- Generates timestamps for transcript

## Dependencies

- ffmpeg (https://www.ffmpeg.org/download.html)

## Help

```
usage: to-sync [-h] [--version] [--graph] [-d SECONDS] [-m SECONDS] [-s]
                   [--logfile PATH]
                   MEDIA [MEDIA ...]

positional arguments:
  MEDIA                 media for which to synchronize subtitles

optional arguments:
  -h, --help            show this help message and exit

```

## Special thanks

[[1] tympanix/subsync whose code was invaluable](https://github.com/tympanix/subsync)
