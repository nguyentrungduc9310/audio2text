from faster_whisper import WhisperModel


class Transcriber:

    def __init__(self):

        self.models = {}

    def load_model(self, name):

        if name not in self.models:

            model = WhisperModel(
                name,
                compute_type="int8"
            )

            self.models[name] = model

        return self.models[name]

    def transcribe(self, audio_file, model_name):

        model = self.load_model(model_name)

        segments, info = model.transcribe(
            audio_file,
            beam_size=5
        )

        text = ""
        segs = []

        for seg in segments:

            text += seg.text + " "

            segs.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            })

        return text.strip(), segs