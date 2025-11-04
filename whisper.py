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
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

import torch
import whisper
from tqdm import tqdm


class MemoryManager:
    """Quản lý memory cho GPU và CPU"""

    @staticmethod
    def get_memory_info(device_id=None):
        memory_info = {
            'cpu_percent': psutil.virtual_memory().percent,
            'cpu_available': psutil.virtual_memory().available / (1024**3),
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
        gc.collect()

    @staticmethod
    def clear_all_cache(device_id=None):
        MemoryManager.clear_gpu_cache(device_id)
        MemoryManager.clear_cpu_cache()


class MultiGPUVideoTranscriber:
    def __init__(self, model_size="large-v2", memory_management=True, 
                 language="auto", device_ids=None):

        self.memory_management = memory_management
        self.language = language
        self.model_size = model_size

        # Setup devices
        if device_ids is None:
            if torch.cuda.is_available():
                self.device_ids = list(range(torch.cuda.device_count()))
                print(f"Auto-detected {len(self.device_ids)} GPUs: {self.device_ids}", flush=True)
            else:
                self.device_ids = ["cpu"]
                print("No CUDA available, using CPU", flush=True)
        else:
            self.device_ids = device_ids

        # Print GPU info
        if torch.cuda.is_available():
            for i in self.device_ids:
                if isinstance(i, int):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    compute_cap = torch.cuda.get_device_capability(i)
                    print(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.1f}GB, Compute: {compute_cap}", flush=True)

        # Initialize transcribers
        self.transcribers = {}
        self.init_transcribers()

        # Thread-safe queue
        self.gpu_queue = queue.Queue()
        for device_id in self.device_ids:
            self.gpu_queue.put(device_id)

    def init_transcribers(self):
        if torch.cuda.is_available():
            print("Initializing transcribers on multiple GPUs...", flush=True)
        else:
            print("Initializing transcriber on CPU...", flush=True)

        for device_id in self.device_ids:
            if isinstance(device_id, int):  # GPU
                device = f"cuda:{device_id}"
            else:  # CPU
                device = "cpu"

            print(f"Loading Whisper model '{self.model_size}' on {device}...", flush=True)

            model = whisper.load_model(self.model_size, device=device)

            transcriber_config = {
                'model': model,
                'device': device,
                'device_id': device_id
            }

            self.transcribers[device_id] = transcriber_config

            if self.memory_management:
                memory_info = MemoryManager.get_memory_info(device_id if isinstance(device_id, int) else None)
                print(f"Memory after loading on {device}: {memory_info}", flush=True)

    def transcribe_audio_on_device(self, audio_path: str, device_id, 
                                   progress_callback=None) -> Optional[Dict[str, Any]]:
        if not os.path.exists(audio_path):
            return None

        transcriber = self.transcribers[device_id]
        device = transcriber['device']

        if progress_callback:
            progress_callback(f"Processing on {device}...")

        if self.memory_management:
            MemoryManager.clear_all_cache(device_id if isinstance(device_id, int) else None)

        try:
            # Whisper transcribe options
            options = {
                'fp16': torch.cuda.is_available(),  # Use FP16 on GPU
                'verbose': False
            }
            
            # Set language
            if self.language != "auto":
                options['language'] = self.language

            result = transcriber['model'].transcribe(
                str(audio_path),
                **options
            )

            if progress_callback:
                progress_callback(f"Transcription done on {device}")

            # Format result to be consistent
            formatted_result = {
                'text': result['text'],
                'segments': result['segments'],
                'language': result['language'],
                'processed_on_device': device
            }

            if self.memory_management:
                MemoryManager.clear_all_cache(device_id if isinstance(device_id, int) else None)

            return formatted_result

        except Exception as e:
            print(f"Transcription error on {device}: {e}", flush=True)
            if self.memory_management:
                MemoryManager.clear_all_cache(device_id if isinstance(device_id, int) else None)
            return None

    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        cmd = ["ffmpeg", "-y", "-i", str(video_path),
               "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            return os.path.exists(audio_path)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}", flush=True)
            return False

    def process_video(self, video_path: str, progress_callback=None) -> Dict[str, Any]:
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

        try:
            device_id = self.gpu_queue.get(timeout=1)
        except queue.Empty:
            device_id = self.device_ids[0]

        try:
            if progress_callback:
                progress_callback(f"Processing {video_name} on device {device_id}")

            if not self.extract_audio(video_path, audio_path):
                result["error"] = "Audio extraction failed"
                return result

            script = self.transcribe_audio_on_device(audio_path, device_id, progress_callback)

            if script is None:
                result["error"] = "Transcription failed"
                return result

            result["transcription"] = script
            result["device_used"] = self.transcribers[device_id]['device']
            result["success"] = True

        except Exception as e:
            result["error"] = str(e)

        finally:
            self.gpu_queue.put(device_id)
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result


def process_video_folder_multi_gpu(folder_path: str, output_json: str, 
                                   model_size="large-v2", memory_management=True, 
                                   language="auto", max_workers=2) -> None:
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}", flush=True)
        return

    print("Initializing multi-GPU transcriber...", flush=True)
    transcriber = MultiGPUVideoTranscriber(
        model_size=model_size,
        memory_management=memory_management,
        language=language
    )

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

    with ThreadPoolExecutor(max_workers=min(max_workers, len(transcriber.device_ids))) as executor:
        future_to_video = {
            executor.submit(transcriber.process_video, str(video_path)): video_path
            for video_path in video_files
        }

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

    print(f"Saving results to {output_json}...", flush=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    successful = sum(1 for r in results if r["success"])
    print(f"\n=== Processing Summary ===", flush=True)
    print(f"Total videos: {len(results)}", flush=True)
    print(f"Successful: {successful}", flush=True)
    print(f"Failed: {len(results) - successful}", flush=True)
    print(f"Devices used: {len(transcriber.device_ids)}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-GPU Video Transcription with Whisper")
    parser.add_argument("folder_path", help="Path to video folder")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON")
    parser.add_argument("-m", "--model", default="large-v2", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model size")
    parser.add_argument("-l", "--language", default="auto", help="Language code or 'auto'")
    parser.add_argument("-w", "--workers", type=int, default=2, help="Max parallel workers")
    parser.add_argument("--no-memory-management", action="store_true")

    args = parser.parse_args()

    memory_management = not args.no_memory_management

    process_video_folder_multi_gpu(
        args.folder_path,
        args.output,
        args.model,
        memory_management,
        args.language,
        args.workers
    )
