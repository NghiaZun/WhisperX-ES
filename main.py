import os
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import subprocess

import torch
import whisperx


class VideoTranscriberX:
    def __init__(self, model_size="large-v2"):
        self.device = "cpu"
        print(f"Using device: {self.device}", flush=True)

        print(f"Loading WhisperX model: {model_size}", flush=True)
        self.model = whisperx.load_model(model_size, self.device, compute_type="float32")

    def extract_audio(self, video_path, audio_path):
        """Extract audio from video using ffmpeg"""
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}", flush=True)
            return False

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, text=True)
            if not os.path.exists(audio_path):
                print(f"Audio file was not created: {audio_path}", flush=True)
                return False
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}", flush=True)
            return False

    def transcribe_audio(self, audio_path):
        """Transcribe audio using WhisperX (CPU)"""
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}", flush=True)
            return None

        try:
            result = self.model.transcribe(str(audio_path), batch_size=1)
            if not result or "segments" not in result:
                print("No transcription result", flush=True)
                return None
            if not result["segments"]:
                return {"segments": [], "word_segments": []}
            return result
        except Exception as e:
            print(f"Transcription failed: {e}", flush=True)
            return None

    def process_video(self, video_path):
        video_path = Path(video_path)
        video_name = video_path.stem
        tmp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(tmp_dir, f"{video_name}.wav")

        result = {
            "video_path": str(video_path),
            "video_name": video_name,
            "processed_at": datetime.now().isoformat(),
            "success": False,
            "error": None,
            "transcription": None
        }

        try:
            if not video_path.exists() or not video_path.is_file():
                result["error"] = f"Video file invalid: {video_path}"
                return result

            print(f"Extracting audio from {video_path.name}...", flush=True)
            if not self.extract_audio(video_path, audio_path):
                result["error"] = "Audio extraction failed"
                return result

            print(f"Transcribing {video_path.name}...", flush=True)
            transcription = self.transcribe_audio(audio_path)
            if transcription is None:
                result["error"] = "Transcription failed"
                return result

            result["transcription"] = transcription
            result["success"] = True
            print(f"Successfully processed: {video_name}", flush=True)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result


def process_video_folder(folder_path, output_json, model_size="large-v2"):
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Folder not found or not a directory: {folder_path}", flush=True)
        return

    transcriber = VideoTranscriberX(model_size=model_size)

    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
    video_files = []
    for ext in video_extensions:
        video_files.extend(folder_path.rglob(ext))
        video_files.extend(folder_path.rglob(ext.upper()))
    video_files = sorted(list(set(video_files)))

    if not video_files:
        print(f"No video files found in {folder_path}", flush=True)
        return

    results = []
    for i, video_path in enumerate(video_files, 1):
        print(f"\n--- Processing [{i}/{len(video_files)}] ---", flush=True)
        result = transcriber.process_video(str(video_path))
        results.append(result)

        # Save backup every 5 files
        if i % 5 == 0:
            backup_file = f"{output_json}.backup_{i}"
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved backup to {backup_file}", flush=True)

    # Save final results
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved final results to {output_json}", flush=True)
    successful = sum(1 for r in results if r["success"])
    print(f"Processing complete: {successful}/{len(results)} videos successful", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe videos using WhisperX (CPU)")
    parser.add_argument("folder_path", help="Path to folder containing videos")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON file")
    parser.add_argument("-m", "--model", default="large-v2", help="WhisperX model size")

    args = parser.parse_args()
    process_video_folder(args.folder_path, args.output, args.model)
