import logging
import multiprocessing


def setup_logger(log_level=logging.INFO, filename='log.log', use_file_handler=True, use_stdout_handler=True):
    """Configure logging"""
    # Create a logger
    logger = multiprocessing.get_logger()
    logger.setLevel(log_level)  # Set the logging level

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s [%(process)d|%(levelname)s] %(message)s')

    # this bit will make sure we won't have
    # duplicated messages in the output
    if not len(logger.handlers):
        # Create a stream handler (stdout)
        if use_stdout_handler:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(log_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        if use_file_handler:
            file_handler = logging.FileHandler(filename)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
