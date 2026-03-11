import threading

from faster_whisper import WhisperModel


class ModelManager:

    _instance = None

    _lock = threading.Lock()

    def __new__(cls):

        with cls._lock:

            if cls._instance is None:

                cls._instance = super().__new__(cls)

                cls._instance._models = {}

                cls._instance._model_lock = threading.Lock()

        return cls._instance

    def load(self, name):

        with self._model_lock:

            if name not in self._models:

                model = WhisperModel(
                    name,
                    compute_type="int8"
                )

                self._models[name] = model

        return self._models[name]

    def get_loaded(self):

        return list(self._models.keys())
