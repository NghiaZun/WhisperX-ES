import os
import json
import glob
import torch
import whisper
import librosa
from datetime import datetime
from pathlib import Path
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
from moviepy.editor import VideoFileClip
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoTranscriber:
    def __init__(self, model_size="large", hf_token=None):
        """
        Initialize the video transcriber
        
        Args:
            model_size (str): Whisper model size (tiny, base, small, medium, large)
            hf_token (str): Hugging Face token for speaker diarization
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
                
            except ImportError as e:
                logger.warning(f"WhisperX not available: {e}. Speaker diarization disabled.")
                self.diarization_pipeline = None
    
    def extract_audio(self, video_path, audio_path):
        """Extract audio from video file"""
        try:
            # Try with moviepy first
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            return True
        except Exception as e:
            logger.error(f"Failed to extract audio from {video_path}: {e}")
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
            
            # Speaker diarization
            diarized = self.diarization_pipeline(audio_path)
            
            # Align script with audio
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
            
            # Assign speakers
            result_segments, word_seg = list(self.assign_word_speakers(
                diarized, script_aligned
            ).values())
            
            # Format results
            transcribed_segments = []
            for segment in result_segments:
                transcribed_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": segment["speaker"]
                })
            
            return transcribed_segments
            
        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            return None
    
    def process_video(self, video_path):
        """Process a single video file"""
        logger.info(f"Processing: {video_path}")
        
        # Setup paths
        video_name = Path(video_path).stem
        audio_path = f"temp_{video_name}.wav"
        
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
            # Extract audio
            if not self.extract_audio(video_path, audio_path):
                result["error"] = "Failed to extract audio"
                return result
            
            # Transcribe
            script = self.transcribe_audio(audio_path)
            if not script:
                result["error"] = "Failed to transcribe audio"
                return result
            
            result["transcription"] = script["text"]
            result["segments"] = script["segments"]
            
            # Perform speaker diarization if available
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
            # Cleanup temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
        return result

def process_video_folder(folder_path, output_json, model_size="large", hf_token=None, 
                        video_extensions=None):
    """
    Process all videos in a folder and save results to JSON
    
    Args:
        folder_path (str): Path to folder containing videos
        output_json (str): Path to output JSON file
        model_size (str): Whisper model size
        hf_token (str): Hugging Face token for speaker diarization
        video_extensions (list): List of video file extensions to process
    """
    
    if video_extensions is None:
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
    
    # Initialize transcriber
    transcriber = VideoTranscriber(model_size=model_size, hf_token=hf_token)
    
    # Find all video files
    video_files = []
    for extension in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, extension)))
        video_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
    
    logger.info(f"Found {len(video_files)} video files")
    
    if not video_files:
        logger.warning(f"No video files found in {folder_path}")
        return
    
    # Process each video
    results = []
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"Processing video {i}/{len(video_files)}: {os.path.basename(video_path)}")
        
        result = transcriber.process_video(video_path)
        results.append(result)
        
        # Save intermediate results (in case of crashes)
        if i % 5 == 0 or i == len(video_files):
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump({
                    "processed_at": datetime.now().isoformat(),
                    "total_videos": len(video_files),
                    "processed_videos": i,
                    "model_size": model_size,
                    "speaker_diarization_enabled": hf_token is not None,
                    "results": results
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Intermediate save: {i}/{len(video_files)} completed")
    
    # Final save
    logger.info(f"Processing complete. Results saved to: {output_json}")
    successful = sum(1 for r in results if r["success"])
    logger.info(f"Successfully processed: {successful}/{len(video_files)} videos")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch video transcription with speaker diarization')
    parser.add_argument('folder_path', help='Path to folder containing videos')
    parser.add_argument('-o', '--output', default='transcription_results.json', 
                       help='Output JSON file path (default: transcription_results.json)')
    parser.add_argument('-m', '--model', default='large', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: large)')
    parser.add_argument('-t', '--token', default=None,
                       help='Hugging Face token for speaker diarization (optional)')
    parser.add_argument('-e', '--extensions', nargs='+', 
                       default=['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm'],
                       help='Video file extensions to process (default: common video formats)')
    
    args = parser.parse_args()
    
    # Get token from args or environment variable
    hf_token = args.token or os.getenv('HF_TOKEN')
    
    # Validate folder path
    if not os.path.exists(args.folder_path):
        logger.error(f"Folder not found: {args.folder_path}")
        exit(1)
    
    logger.info(f"Starting batch transcription:")
    logger.info(f"  Folder: {args.folder_path}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Speaker diarization: {'Enabled' if hf_token else 'Disabled'}")
    
    # Run processing
    process_video_folder(
        folder_path=args.folder_path,
        output_json=args.output,
        model_size=args.model,
        hf_token=hf_token,
        video_extensions=args.extensions
    )