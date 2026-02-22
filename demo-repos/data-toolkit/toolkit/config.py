"""Configuration constants for the data toolkit."""

# Version
VERSION = "0.3.1"

# Default settings
DEFAULT_BATCH_SIZE = 1000
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# Supported file formats
SUPPORTED_FORMATS = ["csv", "json", "tsv", "xlsx"]

# Column type mappings
TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}


def get_default_config():
    """Return default configuration dict."""
    return {
        "batch_size": DEFAULT_BATCH_SIZE,
        "timeout": DEFAULT_TIMEOUT,
        "max_retries": MAX_RETRIES,
        "formats": SUPPORTED_FORMATS,
    }
