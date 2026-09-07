import logging

import nltk
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from simgrep.config import load_app_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_NAME = load_app_config().model


def cache_model_and_tokenizer() -> list[str]:
    failures: list[str] = []
    try:
        AutoTokenizer.from_pretrained(MODEL_NAME)
        snapshot_download(MODEL_NAME)
    except Exception as exc:
        logger.error("Error caching model assets for %s: %s", MODEL_NAME, exc)
        failures.append(f"model_cache: {exc}")
    return failures


def cache_nltk_data() -> list[str]:
    failures: list[str] = []
    packages = [
        ("punkt", "tokenizers/punkt"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ]
    for pkg_id, pkg_path in packages:
        try:
            nltk.data.find(pkg_path)
        except LookupError:
            if not nltk.download(pkg_id, quiet=True):
                failures.append(f"nltk:{pkg_id}: download returned false")
        except Exception as exc:
            failures.append(f"nltk:{pkg_id}: {exc}")
    return failures


if __name__ == "__main__":
    logger.info("Caching dense Hugging Face model and NLTK data.")
    failures = [*cache_model_and_tokenizer(), *cache_nltk_data()]
    if failures:
        for failure in failures:
            logger.error(" - %s", failure)
        raise SystemExit(1)
    logger.info("Cache complete.")
