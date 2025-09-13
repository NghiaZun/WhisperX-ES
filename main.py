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
        
        # Sửa lỗi: kiểm tra device và compute_type phù hợp
        if self.device == "cuda":
            compute_type = "float16"  # GPU dùng float16 để tối ưu
        else:
            compute_type = "float32"  # CPU dùng float32
            
        self.model = whisperx.load_model(
            model_size,
            self.device,
            compute_type=compute_type
        )

        self.hf_token = hf_token
        self.diarization_pipeline = None
        self.align_model = None
        self.align_metadata = None

        if hf_token:
            try:
                print("Loading alignment + diarization models...", flush=True)
                # Sửa lỗi: tự động detect ngôn ngữ hoặc cho phép config
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code="en", device=self.device
                )
                self.diarization_pipeline = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token, device=self.device
                )
                print("Alignment + diarization models loaded successfully", flush=True)
            except Exception as e:
                print(f"Warning: Diarization not available: {e}", flush=True)
                self.diarization_pipeline = None
                self.align_model = None
                self.align_metadata = None

    def extract_audio(self, video_path, audio_path):
        """Extract audio from video using ffmpeg"""
        try:
            # Sửa lỗi: kiểm tra file input tồn tại
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}", flush=True)
                return False
                
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),  # Đảm bảo path là string
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(audio_path)  # Đảm bảo path là string
            ]
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE,  # Capture stderr để debug
                check=True,
                text=True
            )
            
            # Kiểm tra file audio được tạo thành công
            if not os.path.exists(audio_path):
                print(f"Audio file was not created: {audio_path}", flush=True)
                return False
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr if e.stderr else str(e)}", flush=True)
            return False
        except Exception as e:
            print(f"Audio extraction failed: {e}", flush=True)
            return False

    def transcribe_audio(self, audio_path):
        """Transcribe + align + diarize using WhisperX"""
        try:
            print(f"Transcribing: {audio_path}", flush=True)

            # Sửa lỗi: kiểm tra file audio tồn tại
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}", flush=True)
                return None

            # Transcribe
            result = self.model.transcribe(str(audio_path), batch_size=16)
            
            # Kiểm tra kết quả transcription
            if not result or "segments" not in result:
                print("No transcription result", flush=True)
                return None
                
            if not result["segments"]:
                print("No speech detected", flush=True)
                return {"segments": [], "word_segments": []}

            # Align (nếu có model)
            if self.align_model is not None and self.align_metadata is not None:
                try:
                    print("Performing alignment...", flush=True)
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
                    # Tiếp tục với kết quả không align

            # Diarization (nếu có pipeline)
            if self.diarization_pipeline is not None:
                try:
                    print("Performing diarization...", flush=True)
                    diarize_segments = self.diarization_pipeline(str(audio_path))
                    if diarize_segments and hasattr(diarize_segments, 'segments'):
                        result = whisperx.assign_word_speakers(diarize_segments, result)
                        print("Diarization completed", flush=True)
                except Exception as e:
                    print(f"Diarization failed: {e}", flush=True)
                    # Tiếp tục với kết quả không diarize

            return result
            
        except Exception as e:
            print(f"Transcription failed: {e}", flush=True)
            import traceback
            print(f"Full error: {traceback.format_exc()}", flush=True)
            return None

    def process_video(self, video_path):
        print(f"Processing video: {video_path}", flush=True)

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
            # Kiểm tra file video tồn tại và có thể đọc
            if not video_path.exists():
                result["error"] = f"Video file not found: {video_path}"
                return result
                
            if not video_path.is_file():
                result["error"] = f"Path is not a file: {video_path}"
                return result

            # Extract audio
            print("Extracting audio...", flush=True)
            if not self.extract_audio(video_path, audio_path):
                result["error"] = "Audio extraction failed"
                return result

            # Transcribe
            print("Starting transcription...", flush=True)
            script = self.transcribe_audio(audio_path)
            if script is None:
                result["error"] = "Transcription failed"
                return result

            result["transcription"] = script
            result["success"] = True
            print(f"Successfully processed: {video_name}", flush=True)

        except Exception as e:
            result["error"] = str(e)
            print(f"Error processing {video_name}: {e}", flush=True)
            import traceback
            print(f"Full traceback: {traceback.format_exc()}", flush=True)

        finally:
            # Cleanup temp files
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Could not cleanup temp dir: {e}", flush=True)

        return result


def process_video_folder(folder_path, output_json, model_size="large-v2", hf_token=None):
    """Process all videos in a folder"""
    
    # Validate input path
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
        
    if not folder_path.is_dir():
        print(f"Error: Path is not a directory: {folder_path}")
        return
    
    # Initialize transcriber
    try:
        transcriber = VideoTranscriberX(model_size=model_size, hf_token=hf_token)
    except Exception as e:
        print(f"Failed to initialize transcriber: {e}")
        return

    # Find video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
    video_files = []

    for ext in video_extensions:
        video_files.extend(folder_path.rglob(ext))
        video_files.extend(folder_path.rglob(ext.upper()))

    # Remove duplicates and sort
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        print(f"No video files found in: {folder_path}")
        return
        
    print(f"Found {len(video_files)} video files", flush=True)

    # Process videos
    results = []
    for i, video_path in enumerate(video_files, 1):
        print(f"\n--- Processing [{i}/{len(video_files)}] ---", flush=True)
        print(f"File: {video_path.name}", flush=True)
        
        try:
            result = transcriber.process_video(str(video_path))
            results.append(result)
            
            # Save intermediate results every 5 files
            if i % 5 == 0:
                backup_file = f"{output_json}.backup_{i}"
                with open(backup_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved backup to {backup_file}", flush=True)
                
        except Exception as e:
            print(f"Failed to process {video_path}: {e}", flush=True)
            # Add error result
            results.append({
                "video_path": str(video_path),
                "video_name": video_path.stem,
                "processed_at": datetime.now().isoformat(),
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "transcription": None
            })

    # Save final results
    try:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Tạo thư mục nếu chưa có
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved final results to {output_path}", flush=True)
        
        # Print summary
        successful = sum(1 for r in results if r["success"])
        print(f"\nProcessing complete: {successful}/{len(results)} videos successful", flush=True)
        
    except Exception as e:
        print(f"Failed to save results: {e}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transcribe videos using WhisperX")
    parser.add_argument("folder_path", help="Path to folder containing videos")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON file")
    parser.add_argument("-m", "--model", default="large-v2", help="WhisperX model size")
    parser.add_argument("-t", "--token", default=None, help="HuggingFace token for diarization")
    
    args = parser.parse_args()

    # Get HF token from args or environment
    hf_token = args.token or os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Warning: No HuggingFace token provided. Diarization will be disabled.")
        print("Set HF_TOKEN environment variable or use --token argument for speaker diarization.")

    process_video_folder(
        folder_path=args.folder_path,
        output_json=args.output,
        model_size=args.model,
        hf_token=hf_token
    )
