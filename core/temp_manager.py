import logging
import os
import shutil
import tempfile
import time

logger = logging.getLogger(__name__)

# Project-local temp directory (gitignored)
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")


def get_temp_dir():
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR


def cleanup_temp_dir():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)


def make_temp_file(suffix=""):
    get_temp_dir()
    fd, path = tempfile.mkstemp(suffix=suffix, dir=TEMP_DIR)
    os.close(fd)
    return path


def cleanup_stale_files(max_age_seconds=3600):
    """Remove temp files older than max_age_seconds."""
    if not os.path.exists(TEMP_DIR):
        return 0

    now = time.time()
    removed = 0

    for filename in os.listdir(TEMP_DIR):
        filepath = os.path.join(TEMP_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            if now - os.path.getmtime(filepath) > max_age_seconds:
                os.remove(filepath)
                removed += 1
        except OSError:
            pass

    if removed:
        logger.info("Cleaned up %d stale temp file(s)", removed)

    return removed
