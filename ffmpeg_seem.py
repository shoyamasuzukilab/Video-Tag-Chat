import glob
import os
import time
import ffmpeg
from fractions import Fraction

def get_fps(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    return Fraction(video_stream['avg_frame_rate']).limit_denominator()

def frame_to_timecode(frame_number, fps):
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}_{seconds:02d}"

def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# スクリプトの絶対パスを取得
script_dir = os.path.dirname(os.path.abspath(__file__))

start = time.time()

# 保存先ディレクトリの絶対パスを設定
SAVE_DIR_ROOT = os.path.join(script_dir, 'Segment-Everything-Everywhere-All-At-Once/demo/seem/images')

# 保存先ディレクトリを空にする
clear_directory(SAVE_DIR_ROOT)

# "video"フォルダ内のMP4ファイル
YOUTUBE_VIDEOS_PATH = os.path.join(script_dir, 'video/*')
video_paths_all = glob.glob(YOUTUBE_VIDEOS_PATH)
YOUTUBE_VIDEOS_PATH_DELETE = os.path.join(script_dir, 'video/*.Identifier')
video_paths_delete = glob.glob(YOUTUBE_VIDEOS_PATH_DELETE)
video_paths = []
for video_path in video_paths_all:
    if video_path in video_paths_delete:
        continue
    
    video_paths.append(video_path)

for video_path in video_paths[:1]:
    print(video_path)
    save_dir = SAVE_DIR_ROOT

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # フォルダ作成

    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    original_fps = Fraction(video_stream['avg_frame_rate']).limit_denominator()
    target_fps = 0.2
    frame_interval = int(original_fps / target_fps)

    frame_number = 0
    while frame_number < int(float(video_stream['duration']) * original_fps):
        frame_time = frame_to_timecode(frame_number, original_fps)
        output_filename = os.path.join(save_dir, f"{frame_time}.jpg")
        
        timestamp = float(frame_number / original_fps)
        timestamp_str = f"{timestamp:.2f}"

        try:
            (
                ffmpeg
                .input(video_path, ss=timestamp_str)
                .output(output_filename, vframes=1)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            print('ffmpeg error:', e.stderr.decode())
            break

        frame_number += frame_interval

elapsed_time = time.time() - start
print(f'convert video to images elapsed time: {elapsed_time}')
