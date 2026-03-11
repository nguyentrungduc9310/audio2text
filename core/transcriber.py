import logging

from core.model_manager import ModelManager
from core.chunker import AudioChunker

logger = logging.getLogger(__name__)


class Transcriber:

    def __init__(self):

        self.manager = ModelManager()

        self.chunker = AudioChunker()

    def transcribe(self, audio_file, model_name, vad_filter=True, on_progress=None, on_segment=None):

        logger.info("Loading model '%s'...", model_name)
        model = self.manager.load(model_name)
        logger.info("Model '%s' ready", model_name)

        if self.chunker.needs_chunking(audio_file):
            logger.info("File requires chunking, switching to chunked transcription")
            return self._transcribe_chunked(audio_file, model_name, vad_filter, on_progress=on_progress, on_segment=on_segment)

        logger.info("Transcribing '%s' (single pass)...", audio_file)

        segments, info = model.transcribe(
            audio_file,
            beam_size=5,
            vad_filter=vad_filter
        )

        duration = info.duration if info.duration and info.duration > 0 else None

        text = ""

        segs = []

        for seg in segments:

            text += seg.text + " "

            seg_dict = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            }

            segs.append(seg_dict)

            if on_segment:
                on_segment(seg_dict)

            if on_progress and duration:
                progress = min(seg.end / duration, 1.0)
                on_progress(progress)

        language = info.language

        logger.info("Transcription complete — language=%s, segments=%d", language, len(segs))

        return text.strip(), segs, language

    def _transcribe_chunked(self, audio_file, model_name, vad_filter=True, on_progress=None, on_segment=None):

        model = self.manager.load(model_name)

        logger.info("Splitting audio into chunks...")
        chunks, offsets = self.chunker.split(audio_file)
        logger.info("Split into %d chunk(s)", len(chunks))

        num_chunks = len(chunks)

        try:

            all_segments = []

            language = None

            for i, chunk_path in enumerate(chunks, 1):

                logger.info("Transcribing chunk %d/%d...", i, num_chunks)

                segments, info = model.transcribe(
                    chunk_path,
                    beam_size=5,
                    vad_filter=vad_filter
                )

                chunk_duration = info.duration if info.duration and info.duration > 0 else None

                segs = []

                offset = offsets[i - 1]

                for seg in segments:

                    segs.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text
                    })

                    if on_segment:
                        on_segment({
                            "start": seg.start + offset,
                            "end": seg.end + offset,
                            "text": seg.text
                        })

                    if on_progress and chunk_duration:
                        chunk_progress = min(seg.end / chunk_duration, 1.0)
                        overall = ((i - 1) + chunk_progress) / num_chunks
                        on_progress(min(overall, 1.0))

                all_segments.append(segs)

                if on_progress:
                    on_progress(i / num_chunks)

                if language is None:
                    language = info.language

                logger.info("Chunk %d/%d done — %d segments", i, num_chunks, len(segs))

        finally:

            self.chunker.cleanup(chunks, audio_file)

        merged = self.chunker.merge_segments(all_segments, offsets)

        text = " ".join(seg["text"] for seg in merged).strip()

        return text, merged, language
