import os
import json
import glob
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import torch
import whisperx


class VideoTranscriberX:
    def __init__(self, model_size="large-v2", hf_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}", flush=True)

        print(f"Loading WhisperX model: {model_size}", flush=True)
        # Fix: dùng float32 thay vì float16
        self.model = whisperx.load_model(
            model_size,
            self.device,
            compute_type="float32"
        )

        self.hf_token = hf_token
        self.diarization_pipeline = None
        self.align_model = None
        self.align_metadata = None

        if hf_token:
            try:
                print("Loading alignment + diarization models...", flush=True)
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code="en", device=self.device
                )
                self.diarization_pipeline = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token, device=self.device
                )
            except Exception as e:
                print(f"WhisperX diarization not available: {e}", flush=True)


    def extract_audio(self, video_path, audio_path):
        """Extract audio from video using ffmpeg"""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                audio_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception as e:
            print(f"FFmpeg failed: {e}", flush=True)
            return False

    def transcribe_audio(self, audio_path):
        """Transcribe + align + diarize using WhisperX"""
        try:
            print(f"Transcribing: {audio_path}", flush=True)

            # Use path directly to avoid numpy/tensor issues
            result = self.model.transcribe(audio_path)

            # Align
            if self.align_model is not None:
                try:
                    result = whisperx.align(
                        result["segments"],
                        self.align_model,
                        self.align_metadata,
                        audio_path,
                        self.device
                    )
                except Exception as e:
                    print(f"Alignment failed: {e}", flush=True)

            # Diarization
            if self.diarization_pipeline is not None:
                try:
                    diarize_segments = self.diarization_pipeline(audio_path)
                    if diarize_segments:
                        result = whisperx.assign_word_speakers(diarize_segments, result)
                except Exception as e:
                    print(f"Diarization failed: {e}", flush=True)

            return result
        except Exception as e:
            print(f"Failed to transcribe: {e}", flush=True)
            return None

    def process_video(self, video_path):
        print(f"Processing video: {video_path}", flush=True)

        video_name = Path(video_path).stem
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
            if not script:
                result["error"] = "Transcription failed"
                return result

            result["transcription"] = script
            result["success"] = True
            print(f"Done: {video_name}", flush=True)

        except Exception as e:
            result["error"] = str(e)
            print(f"Error: {e}", flush=True)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result


def process_video_folder(folder_path, output_json, model_size="large-v2", hf_token=None):
    transcriber = VideoTranscriberX(model_size=model_size, hf_token=hf_token)

    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
    video_files = []

    folder_path = Path(folder_path)
    for ext in video_extensions:
        video_files.extend(folder_path.rglob(ext))
        video_files.extend(folder_path.rglob(ext.upper()))

    print(f"Found {len(video_files)} videos", flush=True)

    results = []
    for i, video_path in enumerate(video_files, 1):
        print(f"--- [{i}/{len(video_files)}] ---", flush=True)
        result = transcriber.process_video(str(video_path))
        results.append(result)

    # Save all results at once
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_json}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", help="Path to folder with videos")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON file")
    parser.add_argument("-m", "--model", default="large-v2", help="WhisperX model size")
    parser.add_argument("-t", "--token", default=None, help="HuggingFace token for diarization")
    args = parser.parse_args()

    hf_token = args.token or os.getenv("HF_TOKEN")

    process_video_folder(
        folder_path=args.folder_path,
        output_json=args.output,
        model_size=args.model,
        hf_token=hf_token
    )
