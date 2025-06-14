import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from simgrep.adapters.hf_chunker import load_tokenizer
    from simgrep.adapters.sentence_embedder import _load_embedding_model as load_embedding_model
except ImportError:
    logger.error("Failed to import from simgrep.adapters. " "Ensure simgrep is installed in editable mode (`make install` or `uv pip install -e .`).")
    raise

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


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


if __name__ == "__main__":
    logger.info("Starting Hugging Face model caching process...")
    cache_model_and_tokenizer()
    logger.info("Model caching process finished.")