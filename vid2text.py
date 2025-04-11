import os
import datetime
import argparse
from pathlib import Path
from tqdm import tqdm
import ffmpeg
from faster_whisper import WhisperModel, BatchedInferencePipeline


def get_audio_duration(audio_path):
    try:
        probe = ffmpeg.probe(audio_path)
        duration = float(probe['format']['duration'])
        return duration
    except ffmpeg.Error as e:
        print(f"Error probing audio duration: {e.stderr.decode()}")
        return None

def extract_audio(video_path, output_wav):
    try:
        ffmpeg.input(video_path).output(output_wav, ar=16000, ac=1, format='wav', acodec='pcm_s16le')\
            .global_args('-loglevel', 'error')\
            .run(overwrite_output=True)
        print(f"Audio extracted successfully to: {output_wav}")
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        raise

def transcribe_audio(audio_path, start_time, model_size="small", compute_type="int8"):
    model = WhisperModel(model_size, compute_type=compute_type)
    batched_model = BatchedInferencePipeline(model=model)
    segments, info = batched_model.transcribe(audio_path, batch_size=16)
    print(f"Detected language: {info.language}")
    total_duration = get_audio_duration(audio_path)
    txt_filename = Path(audio_path).with_suffix(".txt")
    srt_filename = Path(audio_path).with_suffix(".srt")
    with open(txt_filename, "w") as txt, open(srt_filename, "w") as srt:
        with tqdm(total=total_duration, unit="sec", desc="Transcribing", unit_scale=True) as pbar:
            for segment in segments:
                start = datetime.datetime.fromtimestamp(start_time + segment.start).strftime("%Y-%m-%d %H:%M:%S") 
                end = datetime.datetime.fromtimestamp(start_time + segment.end).strftime("%Y-%m-%d %H:%M:%S") 
                txt.write(segment.text)
                srt.write(f"[{start} -> {end}] {segment.text}\n")
                pbar.update(segment.end - pbar.n)
    print(f"Transcript saved as: {txt_filename}, {srt_filename}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe video using faster-whisper.")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--model", default="small", help="Model: tiny, base, small, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-large-v3, large-v3-turbo, turbo")
    parser.add_argument("--compute", default="int8", help="Compute type: int8, int8_float32, float16, float32")
    args = parser.parse_args()
    start_time = os.path.getctime(args.video)
    audio_path = Path(args.video).with_suffix(".wav")
    print("Extracting audio from video...")
    extract_audio(args.video, str(audio_path))
    print("Transcribing with faster-whisper...")
    transcribe_audio(str(audio_path), start_time=start_time, model_size=args.model, compute_type=args.compute)
    os.remove(audio_path)

if __name__ == "__main__":
    main()
