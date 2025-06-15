import logging
import sys

import nltk

try:
    from simgrep.adapters.hf_chunker import load_tokenizer
    from simgrep.adapters.sentence_embedder import _load_embedding_model as load_embedding_model
    from simgrep.core.models import SimgrepConfig
except ImportError:
    print(
        "Failed to import from simgrep.adapters. " "Ensure simgrep is installed in editable mode (`make install` or `uv pip install -e .`).",
        file=sys.stderr,
    )
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_NAME = SimgrepConfig().default_embedding_model_name


def cache_model_and_tokenizer():
    """Downloads and caches the specified model and its tokenizer."""
    logger.info(f"Ensuring tokenizer is cached for: {MODEL_NAME}")
    try:
        load_tokenizer(MODEL_NAME)
        logger.info(f"Tokenizer for {MODEL_NAME} is ready.")
    except Exception as e:
        logger.error(f"Error caching tokenizer for {MODEL_NAME}: {e}")

    logger.info(f"Ensuring sentence transformer model is cached for: {MODEL_NAME}")
    try:
        load_embedding_model(MODEL_NAME)
        logger.info(f"SentenceTransformer model {MODEL_NAME} is ready.")
    except Exception as e:
        logger.error(f"Error caching SentenceTransformer model {MODEL_NAME}: {e}")


def cache_nltk_data():
    """Downloads NLTK packages required by unstructured to avoid download during tests."""
    logger.info("Ensuring NLTK packages are cached...")
    packages = [
        ("punkt", "tokenizers/punkt"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ]
    for pkg_id, pkg_path in packages:
        try:
            nltk.data.find(pkg_path)
            logger.info(f"NLTK '{pkg_id}' is ready.")
        except LookupError:
            logger.info(f"Downloading NLTK '{pkg_id}'...")
            nltk.download(pkg_id, quiet=True)
            logger.info(f"NLTK '{pkg_id}' downloaded.")


if __name__ == "__main__":
    logger.info("Starting Hugging Face model and NLTK data caching process...")
    cache_model_and_tokenizer()
    cache_nltk_data()
    logger.info("Model and data caching process finished.")
