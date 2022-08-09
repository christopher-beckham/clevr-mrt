import logging

def get_logger():
    logger = logging.getLogger(__name__)    
    
    # https://stackoverflow.com/questions/533048/how-to-log-source-file-name-and-line-number-in-python
    logging.basicConfig(format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG)
    
    # Disable overly verbose stuff from PIL
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    
    return logger