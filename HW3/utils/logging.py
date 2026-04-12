import logging
import datetime
import os
import time
from functools import wraps


def setup_logger():
    logger = logging.getLogger("HW3")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                  "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    os.makedirs("./logs", exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_handler = logging.FileHandler(f"./logs/run_{now}.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)


def measure_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # More precise for duration than datetime

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        duration = end_time - start_time
        logger = logging.getLogger("HW3")
        logger.info(f"'{func.__name__}' executed in {duration:.2f}s")
        return result
    return wrapper
