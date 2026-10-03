"""Interface for LLM and embedding models."""

import os
import logging

from dotenv import load_dotenv
from groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config

load_dotenv()

logger = logging.getLogger(__name__)


def create_watsonx_embedding():
    """
    Create a local Hugging Face embedding model.

    The original lab used Watsonx embeddings. For the
    standalone local version, we use a free local model.
    """
    embedding_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    logger.info("Created local Hugging Face embedding model")
    return embedding_model


def create_watsonx_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    decoding_method: str = "sample"
):
    """
    Create a Groq client.

    The function keeps the original name so the rest of
    the project can continue using the existing interface.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Make sure it is defined in your .env file."
        )

    client = Groq(api_key=api_key)

    return client


def change_llm_model(new_model_id: str) -> None:
    """Change the configured LLM model."""
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")