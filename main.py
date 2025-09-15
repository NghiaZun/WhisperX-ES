import os
import json
import tempfile
import shutil
import gc
import psutil
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Optional, Dict, Any, List

import torch
import whisperx
from pyannote.audio import Pipeline
from tqdm import tqdm


class MemoryManager:
    """Quản lý memory cho GPU và CPU"""
    
    @staticmethod
    def get_memory_info():
        """Lấy thông tin memory hiện tại"""
        memory_info = {
            'cpu_percent': psutil.virtual_memory().percent,
            'cpu_available': psutil.virtual_memory().available / (1024**3),  # GB
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                'gpu_cached': torch.cuda.memory_reserved() / (1024**3),  # GB
                'gpu_free': (torch.cuda.get_device_properties(0).total_memory - 
                           torch.cuda.memory_allocated()) / (1024**3)  # GB
            })
        
        return memory_info
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def clear_cpu_cache():
        """Force garbage collection"""
        gc.collect()
    
    @staticmethod
    def clear_all_cache():
        """Clear both GPU and CPU cache"""
        MemoryManager.clear_gpu_cache()
        MemoryManager.clear_cpu_cache()
    
    @staticmethod
    def should_clear_memory(threshold_gpu=0.8, threshold_cpu=0.85):
        """Check if memory should be cleared"""
        memory_info = MemoryManager.get_memory_info()
        
        if torch.cuda.is_available():
            gpu_usage = memory_info['gpu_allocated'] / (memory_info['gpu_allocated'] + memory_info['gpu_free'])
            if gpu_usage > threshold_gpu:
                return True
        
        if memory_info['cpu_percent'] > threshold_cpu * 100:
            return True
        
        return False


class VideoTranscriberX:
    def __init__(self, model_size="large-v2", hf_token=None, memory_management=True):
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.memory_management = memory_management
        
        print(f"Using device: {self.device}", flush=True)
        
        # Memory info
        if self.memory_management:
            memory_info = MemoryManager.get_memory_info()
            print(f"Initial memory - CPU: {memory_info['cpu_percent']:.1f}%", flush=True)
            if 'gpu_free' in memory_info:
                print(f"Initial memory - GPU: {memory_info['gpu_free']:.1f}GB free", flush=True)

        # Load WhisperX model with P100 compatibility
        if self.device == "cpu":
            compute_type = "float32"
        else:
            # Check GPU compute capability for float16 support
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                compute_capability = torch.cuda.get_device_capability(0)
                
                # P100 has compute capability 6.0, supports float16 but not efficiently
                # Use float32 for P100 and older cards
                if "P100" in gpu_name or compute_capability[0] < 7:
                    compute_type = "float32"
                    print(f"Using float32 for {gpu_name} (compute capability {compute_capability})", flush=True)
                else:
                    compute_type = "float16"
                    print(f"Using float16 for {gpu_name} (compute capability {compute_capability})", flush=True)
            else:
                compute_type = "float32"
        
        print(f"Loading WhisperX model: {model_size} with {compute_type}", flush=True)
        self.model = whisperx.load_model(model_size, device=self.device, compute_type=compute_type)

        # HuggingFace token
        self.hf_token = hf_token

        # Load alignment model
        self.align_model = None
        self.align_metadata = None
        try:
            print("Loading alignment model...", flush=True)
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code="en", device=self.device
            )
        except Exception as e:
            print(f"Alignment model failed: {e}", flush=True)

        # Load diarization pipeline if token provided
        self.diarization_pipeline = None
        if hf_token:
            try:
                print("Loading pyannote diarization pipeline...", flush=True)
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=hf_token
                )
            except Exception as e:
                print(f"Diarization not available: {e}", flush=True)

        if self.memory_management:
            memory_info = MemoryManager.get_memory_info()
            print(f"After loading models - CPU: {memory_info['cpu_percent']:.1f}%", flush=True)
            if 'gpu_allocated' in memory_info:
                print(f"After loading models - GPU: {memory_info['gpu_allocated']:.1f}GB allocated", flush=True)

    def extract_audio(self, video_path: str, audio_path: str, progress_callback=None) -> bool:
        """Extract audio from video using ffmpeg with progress tracking"""
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
            if progress_callback:
                progress_callback("Extracting audio...")
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE, 
                check=True, 
                text=True
            )
            
            success = os.path.exists(audio_path)
            if progress_callback and success:
                progress_callback("Audio extraction completed")
            
            return success
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}", flush=True)
            return False

    def transcribe_audio(self, audio_path: str, progress_callback=None) -> Optional[Dict[str, Any]]:
        """Transcribe, align, and diarize audio with progress tracking"""
        if not os.path.exists(audio_path):
            print(f"Audio not found: {audio_path}", flush=True)
            return None

        if progress_callback:
            progress_callback("Starting transcription...")

        # Clear memory before transcription if needed
        if self.memory_management and MemoryManager.should_clear_memory():
            if progress_callback:
                progress_callback("Clearing memory cache...")
            MemoryManager.clear_all_cache()

        try:
            # Transcribe
            result = self.model.transcribe(str(audio_path), batch_size=16)
            
            if progress_callback:
                progress_callback("Transcription completed, starting alignment...")

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
                    if progress_callback:
                        progress_callback("Word alignment completed")
                except Exception as e:
                    print(f"Alignment failed: {e}", flush=True)
                    if progress_callback:
                        progress_callback("Alignment failed, continuing...")

            # Diarization
            if self.diarization_pipeline:
                try:
                    if progress_callback:
                        progress_callback("Performing speaker diarization...")
                    
                    diarize_result = self.diarization_pipeline(str(audio_path))
                    result = whisperx.assign_word_speakers(diarize_result, result)
                    
                    if progress_callback:
                        progress_callback("Speaker diarization completed")
                except Exception as e:
                    print(f"Diarization failed: {e}", flush=True)
                    if progress_callback:
                        progress_callback("Diarization failed, continuing...")

            # Clear memory after processing
            if self.memory_management:
                MemoryManager.clear_all_cache()

            return result
            
        except Exception as e:
            print(f"Transcription error: {e}", flush=True)
            if self.memory_management:
                MemoryManager.clear_all_cache()
            return None

    def process_video(self, video_path: str, progress_callback=None) -> Dict[str, Any]:
        """Process single video with progress tracking"""
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
            "transcription": None,
            "memory_info": {}
        }

        # Memory info before processing
        if self.memory_management:
            result["memory_info"]["before"] = MemoryManager.get_memory_info()

        try:
            if progress_callback:
                progress_callback(f"Processing: {video_name}")

            # Extract audio
            if not self.extract_audio(video_path, audio_path, progress_callback):
                result["error"] = "Audio extraction failed"
                return result

            # Transcribe
            script = self.transcribe_audio(audio_path, progress_callback)
            if script is None:
                result["error"] = "Transcription failed"
                return result

            result["transcription"] = script
            result["success"] = True
            
            if progress_callback:
                progress_callback(f"Completed: {video_name}")

        except Exception as e:
            result["error"] = str(e)
            if progress_callback:
                progress_callback(f"Error processing {video_name}: {str(e)}")

        finally:
            # Clean up temp files
            shutil.rmtree(tmp_dir, ignore_errors=True)
            
            # Memory info after processing
            if self.memory_management:
                result["memory_info"]["after"] = MemoryManager.get_memory_info()
                # Force cleanup after each video
                MemoryManager.clear_all_cache()

        return result

    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.memory_management:
            MemoryManager.clear_all_cache()


def process_video_folder(folder_path: str, output_json: str, model_size="large-v2", 
                        hf_token=None, memory_management=True) -> None:
    """Process all videos in folder with progress tracking"""
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Folder invalid: {folder_path}", flush=True)
        return

    # Initialize transcriber
    print("Initializing transcriber...", flush=True)
    transcriber = VideoTranscriberX(
        model_size=model_size, 
        hf_token=hf_token,
        memory_management=memory_management
    )

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

    print(f"Found {len(video_files)} video files", flush=True)
    
    results = []
    
    # Progress bar for overall progress
    with tqdm(video_files, desc="Processing videos", unit="video") as pbar:
        for i, video_path in enumerate(pbar):
            # Update progress bar description
            pbar.set_description(f"Processing: {video_path.name}")
            
            # Progress callback for individual video steps
            def progress_callback(message):
                pbar.set_postfix_str(message)
            
            # Process video
            result = transcriber.process_video(str(video_path), progress_callback)
            results.append(result)
            
            # Update progress bar with status
            status = "✓" if result["success"] else "✗"
            pbar.set_postfix_str(f"{status} {video_path.name}")
            
            # Memory info in progress bar
            if memory_management and 'memory_info' in result and 'after' in result['memory_info']:
                memory_info = result['memory_info']['after']
                if 'gpu_allocated' in memory_info:
                    pbar.set_postfix_str(f"{status} GPU: {memory_info['gpu_allocated']:.1f}GB")
                else:
                    pbar.set_postfix_str(f"{status} CPU: {memory_info['cpu_percent']:.1f}%")

    # Save results
    print(f"\nSaving results to {output_json}...", flush=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Final summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"\n--- Processing Summary ---", flush=True)
    print(f"Total videos: {len(results)}", flush=True)
    print(f"Successful: {successful}", flush=True)
    print(f"Failed: {failed}", flush=True)
    print(f"Results saved to: {output_json}", flush=True)
    
    if memory_management:
        final_memory = MemoryManager.get_memory_info()
        print(f"Final memory usage - CPU: {final_memory['cpu_percent']:.1f}%", flush=True)
        if 'gpu_allocated' in final_memory:
            print(f"Final memory usage - GPU: {final_memory['gpu_allocated']:.1f}GB", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe videos with WhisperX + Memory Management")
    parser.add_argument("folder_path", help="Path to folder containing videos")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON file")
    parser.add_argument("-m", "--model", default="large-v2", help="WhisperX model size")
    parser.add_argument("-t", "--token", default=None, help="HuggingFace token for diarization")
    parser.add_argument("--no-memory-management", action="store_true", 
                       help="Disable automatic memory management")
    
    args = parser.parse_args()

    hf_token = args.token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("No HF token provided, diarization disabled", flush=True)

    memory_management = not args.no_memory_management
    
    process_video_folder(
        args.folder_path, 
        args.output, 
        args.model, 
        hf_token,
        memory_management
    )
