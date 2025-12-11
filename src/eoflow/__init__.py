from pandas.core.arrays.base import isin
from pydantic_core.core_schema import IsInstanceSchema
from starlette.responses import Content

from eoflow.config import Config
from eoflow.logging import setup_logging
from eoflow.utils import PROJECT_ROOT

# set up project config
config = Config()

# set up project logging according to config
logging = setup_logging(
    name=config["logging"]["name"],
    level=config["logging"]["level"],
    log_file=config["logging"]["log_file"],
    console=config["logging"]["console"],
    colored=config["logging"]["colored"],
    format_string=config["logging"]["format_string"],
    file_level=config["logging"]["file_level"],
    console_level=config["logging"]["console_level"],
)


if __name__ == "__main__":
    from logging import Logger

    assert isinstance(logging, Logger)
