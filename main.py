import os
import json
import glob
import torch
import whisper
import subprocess
from datetime import datetime
from pathlib import Path
import logging
import tempfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoTranscriber:
    def __init__(self, model_size="large", hf_token=None):
        """
        Initialize the video transcriber
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load Whisper model
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size, device=self.device)

        self.hf_token = hf_token
        self.diarization_pipeline = None

        # Initialize speaker diarization if token provided
        if hf_token:
            try:
                from whisperx.diarize import DiarizationPipeline
                from whisperx import load_align_model, align
                from whisperx.diarize import assign_word_speakers

                logger.info("Initializing speaker diarization pipeline")
                self.diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token)
                self.load_align_model = load_align_model
                self.align = align
                self.assign_word_speakers = assign_word_speakers

            except Exception as e:
                logger.warning(f"WhisperX not available: {e}. Speaker diarization disabled.")
                self.diarization_pipeline = None

    def extract_audio(self, video_path, audio_path):
        """Extract audio from video using ffmpeg (safe, no ALSA dependency)"""
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
            logger.error(f"FFmpeg failed to extract audio: {e}")
            return False

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        try:
            logger.info(f"Transcribing: {audio_path}")
            result = self.model.transcribe(audio_path)
            return result
        except Exception as e:
            logger.error(f"Failed to transcribe {audio_path}: {e}")
            return None

    def perform_diarization(self, audio_path, script):
        """Perform speaker diarization if available"""
        if not self.diarization_pipeline:
            return None

        try:
            logger.info("Performing speaker diarization...")
            diarized = self.diarization_pipeline(audio_path)

            model_a, metadata = self.load_align_model(
                language_code=script["language"],
                device=self.device
            )
            script_aligned = self.align(
                script["segments"],
                model_a,
                metadata,
                audio_path,
                self.device
            )

            result_segments, word_seg = list(self.assign_word_speakers(
                diarized, script_aligned
            ).values())

            transcribed_segments = []
            for segment in result_segments:
                transcribed_segments.append({
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text"),
                    "speaker": segment.get("speaker")
                })

            return transcribed_segments

        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            return None

    def process_video(self, video_path):
        """Process a single video file"""
        logger.info(f"Processing: {video_path}")

        video_name = Path(video_path).stem
        tmp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(tmp_dir, f"{video_name}.wav")

        result = {
            "video_path": str(video_path),
            "video_name": video_name,
            "processed_at": datetime.now().isoformat(),
            "success": False,
            "error": None,
            "transcription": None,
            "segments": None,
            "speaker_segments": None
        }

        try:
            if not self.extract_audio(video_path, audio_path):
                result["error"] = "Failed to extract audio"
                return result

            script = self.transcribe_audio(audio_path)
            if not script:
                result["error"] = "Failed to transcribe audio"
                return result

            result["transcription"] = script.get("text", "")
            result["segments"] = script.get("segments", [])

            if self.diarization_pipeline:
                speaker_segments = self.perform_diarization(audio_path, script)
                if speaker_segments:
                    result["speaker_segments"] = speaker_segments

            result["success"] = True
            logger.info(f"Successfully processed: {video_name}")

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            result["error"] = str(e)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result

def process_video_folder(folder_path, output_json, model_size="large", hf_token=None,
                        video_extensions=None):
    if video_extensions is None:
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']

    transcriber = VideoTranscriber(model_size=model_size, hf_token=hf_token)

    video_files = []
    for extension in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, extension)))
        video_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))

    logger.info(f"Found {len(video_files)} video files")

    if not video_files:
        logger.warning(f"No video files found in {folder_path}")
        return

    results = []
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"Processing video {i}/{len(video_files)}: {os.path.basename(video_path)}")

        result = transcriber.process_video(video_path)
        results.append(result)

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump({
                "processed_at": datetime.now().isoformat(),
                "total_videos": len(video_files),
                "processed_videos": i,
                "model_size": model_size,
                "speaker_diarization_enabled": hf_token is not None,
                "results": results
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Progress saved: {i}/{len(video_files)} completed")

    successful = sum(1 for r in results if r["success"])
    logger.info(f"Successfully processed: {successful}/{len(video_files)} videos")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Batch video transcription with speaker diarization')
    parser.add_argument('folder_path', help='Path to folder containing videos')
    parser.add_argument('-o', '--output', default='transcription_results.json',
                       help='Output JSON file path')
    parser.add_argument('-m', '--model', default='large',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size')
    parser.add_argument('-t', '--token', default=None,
                       help='Hugging Face token for speaker diarization (optional)')
    parser.add_argument('-e', '--extensions', nargs='+',
                       default=['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm'],
                       help='Video file extensions to process')

    args = parser.parse_args()

    hf_token = args.token or os.getenv('HF_TOKEN')

    if not os.path.exists(args.folder_path):
        logger.error(f"Folder not found: {args.folder_path}")
        raise SystemExit(1)

    logger.info(f"Starting batch transcription:")
    logger.info(f"  Folder: {args.folder_path}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Speaker diarization: {'Enabled' if hf_token else 'Disabled'}")

    process_video_folder(
        folder_path=args.folder_path,
        output_json=args.output,
        model_size=args.model,
        hf_token=hf_token,
        video_extensions=args.extensions
    )
