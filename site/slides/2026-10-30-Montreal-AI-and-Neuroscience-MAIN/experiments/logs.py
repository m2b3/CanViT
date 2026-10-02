import logging


def setup() -> None:
    """Timestamped INFO lines, without the Hugging Face client's line per HTTP request."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
