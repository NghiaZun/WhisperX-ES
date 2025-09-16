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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

import torch
import torch.multiprocessing as mp
import whisperx
from pyannote.audio import Pipeline
from tqdm import tqdm


class MemoryManager:
    """Quản lý memory cho GPU và CPU"""
    
    @staticmethod
    def get_memory_info(device_id=None):
        """Lấy thông tin memory hiện tại"""
        memory_info = {
            'cpu_percent': psutil.virtual_memory().percent,
            'cpu_available': psutil.virtual_memory().available / (1024**3),  # GB
        }
        
        if torch.cuda.is_available():
            if device_id is not None:
                with torch.cuda.device(device_id):
                    memory_info.update({
                        f'gpu_{device_id}_allocated': torch.cuda.memory_allocated(device_id) / (1024**3),
                        f'gpu_{device_id}_cached': torch.cuda.memory_reserved(device_id) / (1024**3),
                        f'gpu_{device_id}_free': (torch.cuda.get_device_properties(device_id).total_memory - 
                                               torch.cuda.memory_allocated(device_id)) / (1024**3)
                    })
            else:
                # All GPUs info
                for i in range(torch.cuda.device_count()):
                    memory_info.update({
                        f'gpu_{i}_allocated': torch.cuda.memory_allocated(i) / (1024**3),
                        f'gpu_{i}_cached': torch.cuda.memory_reserved(i) / (1024**3),
                        f'gpu_{i}_free': (torch.cuda.get_device_properties(i).total_memory - 
                                        torch.cuda.memory_allocated(i)) / (1024**3)
                    })
        
        return memory_info
    
    @staticmethod
    def clear_gpu_cache(device_id=None):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            if device_id is not None:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            else:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    @staticmethod
    def clear_cpu_cache():
        """Force garbage collection"""
        gc.collect()
    
    @staticmethod
    def clear_all_cache(device_id=None):
        """Clear both GPU and CPU cache"""
        MemoryManager.clear_gpu_cache(device_id)
        MemoryManager.clear_cpu_cache()


class MultiGPUVideoTranscriberX:
    def __init__(self, model_size="large-v2", hf_token=None, memory_management=True, 
                 language="auto", device_ids=None):
        
        self.memory_management = memory_management
        self.language = language
        self.hf_token = hf_token
        self.model_size = model_size
        
        # Setup devices - T4x2 configuration
        if device_ids is None:
            if torch.cuda.is_available():
                self.device_ids = list(range(torch.cuda.device_count()))
                print(f"Auto-detected {len(self.device_ids)} GPUs: {self.device_ids}", flush=True)
            else:
                self.device_ids = ["cpu"]
                print("No CUDA available, using CPU", flush=True)
        else:
            self.device_ids = device_ids
        
        # T4-specific optimizations
        self.compute_type = "float16"  # T4 supports float16 efficiently
        self.batch_size = 32  # T4 has 16GB memory, can handle larger batches
        
        # Print GPU info for T4x2
        if torch.cuda.is_available():
            for i in self.device_ids:
                if isinstance(i, int):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    compute_cap = torch.cuda.get_device_capability(i)
                    print(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.1f}GB, Compute: {compute_cap}", flush=True)
        
        # Initialize transcribers for each GPU
        self.transcribers = {}
        self.init_transcribers()
        
        # Thread-safe queue for load balancing
        self.gpu_queue = queue.Queue()
        for device_id in self.device_ids:
            self.gpu_queue.put(device_id)
    
    def init_transcribers(self):
        """Initialize one transcriber per GPU"""
        print("Initializing transcribers on multiple GPUs...", flush=True)
        
        for device_id in self.device_ids:
            if isinstance(device_id, int):  # GPU
                device = f"cuda:{device_id}"
                compute_type = self.compute_type
            else:  # CPU
                device = "cpu"
                compute_type = "float32"
            
            print(f"Loading model on {device}...", flush=True)
            
            # Load WhisperX model
            model = whisperx.load_model("large", device="cpu",  compute_type="int8")

            
            # Initialize transcriber config
            transcriber_config = {
                'model': model,
                'device': device,
                'device_id': device_id,
                'align_model': None,
                'align_metadata': None,
                'detected_language': None,
                'diarization_pipeline': None
            }
            
            # Load diarization pipeline if token provided (only on first GPU to save memory)
            if self.hf_token and device_id == self.device_ids[0]:
                try:
                    print(f"Loading diarization pipeline on {device}...", flush=True)
                    transcriber_config['diarization_pipeline'] = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization",
                        use_auth_token=self.hf_token
                    ).to(device)
                except Exception as e:
                    print(f"Diarization not available on {device}: {e}", flush=True)
            
            self.transcribers[device_id] = transcriber_config
            
            if self.memory_management:
                memory_info = MemoryManager.get_memory_info(device_id if isinstance(device_id, int) else None)
                print(f"Memory after loading on {device}: {memory_info}", flush=True)

    def load_alignment_model(self, device_id, language_code: str):
        """Load alignment model cho ngôn ngữ cụ thể trên GPU cụ thể"""
        transcriber = self.transcribers[device_id]
        
        if (transcriber['align_model'] is not None and 
            transcriber['detected_language'] == language_code):
            return  # Đã load rồi
        
        try:
            device = transcriber['device']
            print(f"Loading alignment model for {language_code} on {device}...", flush=True)
            
            # Language mapping
            language_map = {
                'vi': 'vi', 'zh': 'zh', 'ja': 'ja', 'ko': 'ko', 'th': 'th',
                'en': 'en', 'fr': 'fr', 'de': 'de', 'es': 'es', 'it': 'it',
                'pt': 'pt', 'ru': 'ru', 'ar': 'ar', 'hi': 'hi'
            }
            
            mapped_lang = language_map.get(language_code, language_code)
            
            align_model, align_metadata = whisperx.load_align_model(
                language_code=mapped_lang, 
                device=device
            )
            
            transcriber['align_model'] = align_model
            transcriber['align_metadata'] = align_metadata
            transcriber['detected_language'] = language_code
            
            print(f"Alignment model loaded successfully for {language_code} on {device}", flush=True)
            
        except Exception as e:
            print(f"Failed to load alignment model for {language_code} on {device}: {e}", flush=True)
            transcriber['align_model'] = None
            transcriber['align_metadata'] = None

    def transcribe_audio_on_device(self, audio_path: str, device_id, progress_callback=None) -> Optional[Dict[str, Any]]:
        """Transcribe audio trên GPU cụ thể"""
        if not os.path.exists(audio_path):
            return None

        transcriber = self.transcribers[device_id]
        device = transcriber['device']
        
        if progress_callback:
            progress_callback(f"Processing on {device}...")

        # Clear memory before transcription
        if self.memory_management:
            MemoryManager.clear_all_cache(device_id if isinstance(device_id, int) else None)

        try:
            # Step 1: Transcription
            if self.language == "auto":
                result = transcriber['model'].transcribe(
                    str(audio_path), 
                    batch_size=self.batch_size,
                    language=None
                )
                detected_lang = result.get('language', 'en')
                print(f"Detected language: {detected_lang} on {device}", flush=True)
                
                # Load alignment model
                self.load_alignment_model(device_id, detected_lang)
            else:
                result = transcriber['model'].transcribe(
                    str(audio_path), 
                    batch_size=self.batch_size,
                    language=self.language
                )
                detected_lang = self.language
                
                if transcriber['align_model'] is None:
                    self.load_alignment_model(device_id, detected_lang)
            
            if progress_callback:
                progress_callback(f"Transcription done on {device}, aligning...")

            # Step 2: Word alignment
            if transcriber['align_model'] and transcriber['align_metadata']:
                try:
                    result = whisperx.align(
                        result["segments"],
                        transcriber['align_model'],
                        transcriber['align_metadata'],
                        str(audio_path),
                        device,
                        return_char_alignments=False
                    )
                    if progress_callback:
                        progress_callback(f"Alignment done on {device}")
                except Exception as e:
                    print(f"Alignment failed on {device}: {e}", flush=True)

            # Step 3: Diarization (chỉ trên GPU đầu tiên)
            if transcriber['diarization_pipeline'] and device_id == self.device_ids[0]:
                try:
                    if progress_callback:
                        progress_callback(f"Diarization on {device}...")
                    
                    diarize_result = transcriber['diarization_pipeline'](str(audio_path))
                    result = whisperx.assign_word_speakers(diarize_result, result)
                    
                    if progress_callback:
                        progress_callback(f"Diarization done on {device}")
                except Exception as e:
                    print(f"Diarization failed on {device}: {e}", flush=True)

            result['detected_language'] = detected_lang
            result['processed_on_device'] = device

            # Clear memory
            if self.memory_management:
                MemoryManager.clear_all_cache(device_id if isinstance(device_id, int) else None)

            return result
            
        except Exception as e:
            print(f"Transcription error on {device}: {e}", flush=True)
            if self.memory_management:
                MemoryManager.clear_all_cache(device_id if isinstance(device_id, int) else None)
            return None

    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video"""
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path)
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            return os.path.exists(audio_path)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}", flush=True)
            return False

    def process_video(self, video_path: str, progress_callback=None) -> Dict[str, Any]:
        """Process single video với load balancing"""
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
            "device_used": None
        }

        # Get available GPU
        try:
            device_id = self.gpu_queue.get(timeout=1)
        except queue.Empty:
            # Fallback to first GPU if queue empty
            device_id = self.device_ids[0]

        try:
            if progress_callback:
                progress_callback(f"Processing {video_name} on device {device_id}")

            # Extract audio
            if not self.extract_audio(video_path, audio_path):
                result["error"] = "Audio extraction failed"
                return result

            # Transcribe
            script = self.transcribe_audio_on_device(
                audio_path, 
                device_id, 
                progress_callback
            )
            
            if script is None:
                result["error"] = "Transcription failed"
                return result

            result["transcription"] = script
            result["device_used"] = self.transcribers[device_id]['device']
            result["success"] = True

        except Exception as e:
            result["error"] = str(e)

        finally:
            # Return device to queue
            self.gpu_queue.put(device_id)
            
            # Cleanup
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result


def process_video_folder_multi_gpu(folder_path: str, output_json: str, model_size="large-v2",
                                  hf_token=None, memory_management=True, language="auto",
                                  max_workers=2) -> None:
    """Process videos với multi-GPU support"""
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}", flush=True)
        return

    # Initialize multi-GPU transcriber
    print("Initializing multi-GPU transcriber for T4x2...", flush=True)
    transcriber = MultiGPUVideoTranscriberX(
        model_size=model_size,
        hf_token=hf_token,
        memory_management=memory_management,
        language=language
    )

    # Find video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
    video_files = []
    for ext in video_extensions:
        video_files.extend(folder_path.rglob(ext))
        video_files.extend(folder_path.rglob(ext.upper()))
    
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        print("No video files found", flush=True)
        return

    print(f"Found {len(video_files)} videos. Processing with {len(transcriber.device_ids)} devices...", flush=True)

    results = []
    
    # Process with ThreadPoolExecutor for parallel GPU usage
    with ThreadPoolExecutor(max_workers=min(max_workers, len(transcriber.device_ids))) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(transcriber.process_video, str(video_path)): video_path 
            for video_path in video_files
        }
        
        # Progress bar
        with tqdm(total=len(video_files), desc="Processing videos", unit="video") as pbar:
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    status = "✓" if result["success"] else "✗"
                    device_used = result.get("device_used", "unknown")
                    pbar.set_postfix_str(f"{status} {video_path.name} on {device_used}")
                    
                except Exception as e:
                    error_result = {
                        "video_path": str(video_path),
                        "video_name": video_path.stem,
                        "success": False,
                        "error": str(e),
                        "processed_at": datetime.now().isoformat()
                    }
                    results.append(error_result)
                    pbar.set_postfix_str(f"✗ Error: {video_path.name}")
                
                pbar.update(1)

    # Save results
    print(f"Saving results to {output_json}...", flush=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n=== Multi-GPU Processing Summary ===", flush=True)
    print(f"Total videos: {len(results)}", flush=True)
    print(f"Successful: {successful}", flush=True)
    print(f"Failed: {len(results) - successful}", flush=True)
    print(f"Devices used: {len(transcriber.device_ids)}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-GPU Video Transcription for T4x2")
    parser.add_argument("folder_path", help="Path to video folder")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON")
    parser.add_argument("-m", "--model", default="large-v2", help="WhisperX model")
    parser.add_argument("-t", "--token", help="HuggingFace token")
    parser.add_argument("-l", "--language", default="auto", help="Language code or 'auto'")
    parser.add_argument("-w", "--workers", type=int, default=2, help="Max parallel workers")
    parser.add_argument("--no-memory-management", action="store_true")
    
    args = parser.parse_args()
    
    hf_token = args.token or os.getenv("HF_TOKEN")
    memory_management = not args.no_memory_management
    
    process_video_folder_multi_gpu(
        args.folder_path,
        args.output,
        args.model,
        hf_token,
        memory_management,
        args.language,
        args.workers
    )
