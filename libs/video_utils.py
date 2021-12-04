import os
import youtube_dl
import ffmpeg
import pandas as pd
from tqdm import tqdm
import numpy as np
import librosa


class VideoDownloader:

    def __init__(self, links, folder="videos"):
        self.links = links
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)

    def download_video(self, video_id="twTksx_lWB4"):
        ydl = youtube_dl.YoutubeDL({'outtmpl': f'{self.folder}/%(id)s.%(ext)s', 'format': 'worst'})

        url = video_id if "http" in video_id else f'http://www.youtube.com/watch?v={video_id}'

        with ydl:
            result = ydl.extract_info(url, download=True)

        if 'entries' in result:
            return result['entries'][0]
        else:
            return result

    def download_links(self):
        metadata = pd.DataFrame({
            'link': self.links,
            'metadata': [self.download_video(i) for i in tqdm(self.links)]
        })
        return metadata


class VideoExtractor:

    def __init__(self, videos):
        self.videos = videos

    def video_to_numpy(self, file='3LPANjHlPxo.mp4'):
        probe = ffmpeg.probe(file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        disposition = pd.DataFrame(video_stream.get('disposition'), index=[file])
        video_stream.pop('disposition')
        tags = pd.DataFrame(video_stream.get('tags'), index=[file])
        video_stream.pop('tags')
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        metadata = pd.DataFrame(video_stream, index=[file]).join(disposition).join(tags)
        out, _ = ffmpeg.input(file).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True)

        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        video = video[np.random.choice(video.shape[0], int(np.log(video.shape[0])*24), replace=False), ...]

        return video, metadata

    def sound_to_numpy(self, file='3LPANjHlPxo.mp4'):
        speech, rate = librosa.load(file, sr=16000)
        return speech

    def all_to_numpy(self, video):
        vi, me = self.video_to_numpy(video)
        sp = self.sound_to_numpy(video)
        return vi, sp, me

    def files_to_numpy(self):
        return [self.all_to_numpy(i) for i in tqdm(self.videos)]
