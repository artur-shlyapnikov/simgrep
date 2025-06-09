import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
except ImportError:
    logger.error(
        "Failed to import sentence_transformers or transformers. "
        "Ensure they are installed in the environment."
    )
    raise

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def cache_model_and_tokenizer():
    """Downloads and caches the specified model and its tokenizer."""
    logger.info(f"Attempting to download and cache tokenizer for: {MODEL_NAME}")
    try:
        AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info(
            f"Tokenizer for {MODEL_NAME} processed successfully (cached or downloaded)."
        )
    except Exception as e:
        logger.error(f"Error caching tokenizer for {MODEL_NAME}: {e}")

    logger.info(
        f"Attempting to download and cache sentence transformer model: {MODEL_NAME}"
    )
    try:
        SentenceTransformer(MODEL_NAME)
        logger.info(
            f"SentenceTransformer model {MODEL_NAME} processed successfully (cached or downloaded)."
        )
    except Exception as e:
        logger.error(f"Error caching SentenceTransformer model {MODEL_NAME}: {e}")


if __name__ == "__main__":
    logger.info("Starting Hugging Face model caching process...")
    cache_model_and_tokenizer()
    logger.info("Model caching process finished.")
