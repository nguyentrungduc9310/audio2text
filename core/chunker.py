import os
import subprocess

from pydub import AudioSegment

from core.temp_manager import make_temp_file


class AudioChunker:

    def __init__(self, chunk_minutes=10, overlap_seconds=30):

        self.chunk_ms = chunk_minutes * 60 * 1000

        self.overlap_ms = overlap_seconds * 1000

    def _get_duration_ms(self, audio_path):

        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", audio_path]

        result = subprocess.run(cmd, capture_output=True, text=True)

        return float(result.stdout.strip()) * 1000

    def needs_chunking(self, audio_path):

        return self._get_duration_ms(audio_path) > self.chunk_ms

    def split(self, audio_path):

        audio = AudioSegment.from_file(audio_path)

        duration = len(audio)

        if duration <= self.chunk_ms:
            return [audio_path], [0.0]

        chunks = []

        offsets = []

        start = 0

        tmp_path = None

        try:

            while start < duration:

                end = min(start + self.chunk_ms, duration)

                chunk = audio[start:end]

                tmp_path = make_temp_file(suffix=".mp3")

                chunk.export(tmp_path, format="mp3")

                chunks.append(tmp_path)

                tmp_path = None

                offsets.append(start / 1000.0)

                if end >= duration:
                    break

                start = end - self.overlap_ms

        except Exception:

            # Clean up the current file if export failed before append
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

            # Clean up all previously created chunk files
            for path in chunks:
                if os.path.exists(path):
                    os.remove(path)

            raise

        return chunks, offsets

    def merge_segments(self, all_segments, offsets):

        merged = []

        for i, (segments, offset) in enumerate(zip(all_segments, offsets)):

            for seg in segments:

                adjusted = {
                    "start": seg["start"] + offset,
                    "end": seg["end"] + offset,
                    "text": seg["text"]
                }

                if i > 0 and merged:

                    last = merged[-1]

                    if (adjusted["start"] <= last["end"]
                            and adjusted["text"].strip() == last["text"].strip()):
                        continue

                merged.append(adjusted)

        return merged

    def cleanup(self, chunk_paths, original_path):

        for path in chunk_paths:

            if path != original_path and os.path.exists(path):
                os.remove(path)
