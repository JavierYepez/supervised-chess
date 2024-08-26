import os
from pathlib import Path


class Settings:

    SRC_PATH = Path(os.path.dirname(os.path.realpath(__file__))).absolute()
    PROJECT_PATH = SRC_PATH.parent.absolute()
