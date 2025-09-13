import os
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import subprocess

import torch
import whisperx
from pyannote.audio import Pipeline


class VideoTranscriberX:
    def __init__(self, model_size="large-v2", hf_token=None):
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}", flush=True)

        # Load WhisperX model
        compute_type = "float32"
        print(f"Loading WhisperX model: {model_size}", flush=True)
        self.model = whisperx.load_model(model_size, device=self.device, compute_type=compute_type)

        # HuggingFace token
        self.hf_token = hf_token

        # Load alignment model
        try:
            print("Loading alignment model...", flush=True)
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code="en", device=self.device
            )
        except Exception as e:
            print(f"Alignment model failed: {e}", flush=True)
            self.align_model = None
            self.align_metadata = None

        # Load diarization pipeline if token provided
        if hf_token:
            try:
                print("Loading pyannote diarization pipeline...", flush=True)
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=hf_token
                )
            except Exception as e:
                print(f"Diarization not available: {e}", flush=True)
                self.diarization_pipeline = None
        else:
            self.diarization_pipeline = None

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
            return os.path.exists(audio_path)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}", flush=True)
            return False

    def transcribe_audio(self, audio_path):
        """Transcribe, align, and diarize audio"""
        if not os.path.exists(audio_path):
            print(f"Audio not found: {audio_path}", flush=True)
            return None

        print(f"Transcribing {audio_path}", flush=True)
        result = self.model.transcribe(str(audio_path), batch_size=16)

        # Align words
        if self.align_model and self.align_metadata:
            try:
                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    str(audio_path),
                    self.device,
                    return_char_alignments=False
                )
            except Exception as e:
                print(f"Alignment failed: {e}", flush=True)

        # Diarization
        if self.diarization_pipeline:
            try:
                print("Performing diarization...", flush=True)
                diarize_result = self.diarization_pipeline(str(audio_path))
                result = whisperx.assign_word_speakers(diarize_result, result)
            except Exception as e:
                print(f"Diarization failed: {e}", flush=True)

        return result

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
            if not self.extract_audio(video_path, audio_path):
                result["error"] = "Audio extraction failed"
                return result

            script = self.transcribe_audio(audio_path)
            if script is None:
                result["error"] = "Transcription failed"
                return result

            result["transcription"] = script
            result["success"] = True

        except Exception as e:
            result["error"] = str(e)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result


def process_video_folder(folder_path, output_json, model_size="large-v2", hf_token=None):
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Folder invalid: {folder_path}", flush=True)
        return

    transcriber = VideoTranscriberX(model_size=model_size, hf_token=hf_token)

    # Find videos
    video_extensions = ['*.mp4','*.avi','*.mov','*.mkv','*.wmv','*.flv','*.webm','*.m4v']
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
        print(f"\n--- Processing [{i}/{len(video_files)}]: {video_path.name} ---", flush=True)
        result = transcriber.process_video(str(video_path))
        results.append(result)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_json}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe videos with WhisperX (CPU + optional diarization)")
    parser.add_argument("folder_path", help="Path to folder containing videos")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON file")
    parser.add_argument("-m", "--model", default="large-v2", help="WhisperX model size")
    parser.add_argument("-t", "--token", default=None, help="HuggingFace token for diarization (optional)")
    args = parser.parse_args()

    hf_token = args.token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("No HF token, diarization disabled", flush=True)

    process_video_folder(args.folder_path, args.output, args.model, hf_token)
