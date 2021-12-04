import re
import os
import pickle
import time
import ffmpeg
import subprocess
import pandas as pd
from glob import glob
from libs.video_utils import VideoDownloader, VideoExtractor
from libs.speech_utils import SpeechText
from libs.clip import ClipEmbedding
from sentence_transformers import util


def download_video_ffmpeg(video_id=None):
    lnk = 'https://www.youtube.com/watch?v=' + video_id
    file_name = "/Volumes/Untitled/Download/youtube_videos_mac/" + video_id + '.mp4'
    os.system(f'ffmpeg -i $(youtube-dl -f 18 --no-check-certificate --get-url {lnk}) -ss 00:00:10 -t 00:01:00 -c:v copy -c:a copy {file_name}')


def is_video_in_dbase(file_list):
    video_db_list = []
    for file_name in file_list:
        #get the file key
        file_key_res = re.search('/([\w_-]+).mp4', file_name)

        if file_key_res:
            file_key = file_key_res.group(1)
            video_db_list.append(file_key)
    return video_db_list


if __name__ == "__main__":
    you_ds = pd.read_csv('/Volumes/Untitled/youtube_ds/ml-youtube.csv')
    movie_title_df = pd.read_csv('/Volumes/Untitled/youtube_ds/Movie_Id_Titles.csv')
    movie_rating_df = pd.read_csv('/Volumes/Untitled/youtube_ds/Movie_Recommender_Dataset.csv')

    movie_title_df_tmp = pd.merge(movie_title_df, you_ds, on='title')
    movie_rating_df = pd.merge(movie_rating_df, movie_title_df_tmp, on='item_id')

    movie_metadata_dict_old = {}
    movie_metadata_dict_new = {}

    with open('/Volumes/Untitled/youtube_trailers/movie_metadata_dict.pickle', 'wb') as handle:
        pickle.dump(movie_metadata_dict_old, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('/Volumes/Untitled/youtube_trailers/movie_metadata_dict_new.pickle', 'wb') as handle:
        pickle.dump(movie_metadata_dict_new, handle, protocol=pickle.HIGHEST_PROTOCOL)

    movie_list = list(movie_rating_df['youtubeId'].unique())
    movie_metadata_dict = {**movie_metadata_dict_new, **movie_metadata_dict_old}

    movie_set = set(movie_list)

    file_list = glob('/Volumes/Untitled/youtube_trailers/*')
    video_db_list = is_video_in_dbase(file_list)
    video_db_set = set(video_db_list)

    diff_video_set = movie_set - video_db_set
    diff_video_list = list(diff_video_set)
    print(len(diff_video_list))

    t0 = time.time()
    for lnk in diff_video_list[52:100]:
        download_video_ffmpeg(lnk)

    t1 = time.time()
    total = t1 - t0
    print("Total time: ", total)

    # dl = VideoDownloader([
    #     "https://www.tiktok.com/@positiff_ua/video/7032321262672235782?is_from_webapp=1&sender_device=pc&web_id7037629015918790150",
    #     "https://www.tiktok.com/@potapreal/video/7036860430102416645?is_from_webapp=1&sender_device=pc&web_id7037629015918790150",
    #     "https://www.tiktok.com/@nazarrrr42/video/7029339114826681605?is_from_webapp=1&sender_device=pc&web_id7037629015918790150"
    # ])

    # you_meta = dl.download_links()
    #
    # ve = VideoExtractor(glob("videos/*"))
    #
    # vdata = ve.files_to_numpy()
    #
    # spc = SpeechText("voidful/wav2vec2-xlsr-multilingual-56")
    #
    # txts = [spc.sound2text(i[1]) for i in vdata]
    #
    # ce = ClipEmbedding()
    #
    # vecs = [ce.get_embedding(vdata[i][0], txts[i]) for i in range(3)]
    #
    # util.cos_sim(vecs, vecs)