import uuid
import threading
from enum import Enum
from datetime import datetime


class JobStatus(str, Enum):

    PENDING = "pending"

    PROCESSING = "processing"

    COMPLETED = "completed"

    FAILED = "failed"


class QueueManager:

    def __init__(self):

        self._jobs = {}

        self._lock = threading.Lock()

    def create_job(self, filename, model):

        job_id = str(uuid.uuid4())

        job = {
            "id": job_id,
            "filename": filename,
            "model": model,
            "status": JobStatus.PENDING,
            "progress": 0,
            "segments": [],
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat()
        }

        with self._lock:

            self._jobs[job_id] = job

        return job_id

    def update_status(self, job_id, status, progress=None, result=None, error=None):

        with self._lock:

            if job_id not in self._jobs:
                return

            self._jobs[job_id]["status"] = status

            if progress is not None:
                self._jobs[job_id]["progress"] = progress

            if result is not None:
                self._jobs[job_id]["result"] = result

            if error is not None:
                self._jobs[job_id]["error"] = error

    def add_segment(self, job_id, segment):

        with self._lock:

            if job_id in self._jobs:
                self._jobs[job_id]["segments"].append(segment)

    def get_job(self, job_id):

        with self._lock:

            return self._jobs.get(job_id)

    def list_jobs(self):

        with self._lock:

            return list(self._jobs.values())
