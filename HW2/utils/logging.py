import logging
import datetime
import os


def setup_logger():
    logger = logging.getLogger("HW2")
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
