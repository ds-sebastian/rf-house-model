import logging
from pathlib import Path

loggerName = Path(__file__).stem
logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s') # create logging formatter
logger = logging.getLogger(loggerName) # create logger
logger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler() # create console handler
consoleHandler.setLevel(logging.WARNING)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler) # Add console handler to logger