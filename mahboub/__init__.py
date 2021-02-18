import os
import sys
from pathlib import  Path as PosixPath

if sys.version_info < (3, 5):
    raise RuntimeError('Mahboub Chatbot supports Python 3.5 or higher.')

DATA_DIR = (
    os.getenv('DRQA_DATA') or
    os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'data')
)

from . import tokenizers
from . import retriever

