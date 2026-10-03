"""Module for querying indexed LinkedIn profile data."""

import logging
from typing import Any

from llama_index.core import VectorStoreIndex, PromptTemplate

from modules.llm_interface import create_watsonx_llm
import config

logger = logging.getLogger(__name__)


def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Generate three interesting facts from the profile."""

    try:
        client = create_watsonx_llm(
            temperature=0.0,
            max_new_tokens=500,
            decoding_method="sample"
        )

        retriever = index.as_retriever(
            similarity_top_k= 10 #config.SIMILARITY_TOP_K
        )

        nodes = retriever.retrieve(
            "career education work experience skills"
        )

        context_str = "\n\n".join(
            node.node.get_text() for node in nodes
        )

        prompt = config.INITIAL_FACTS_TEMPLATE.format(
            context_str=context_str
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=900
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in generate_initial_facts: {e}")
        return "Failed to generate initial facts."


def answer_user_query(
    index: VectorStoreIndex,
    user_query: str
) -> Any:
    """Answer a user question using retrieved profile context."""

    try:
        client = create_watsonx_llm(
            temperature=0.0,
            max_new_tokens=500,
            decoding_method="greedy"
        )

        retriever = index.as_retriever(
            similarity_top_k=config.SIMILARITY_TOP_K
        )

        source_nodes = retriever.retrieve(user_query)

        context_str = "\n\n".join(
            node.node.get_text()
            for node in source_nodes
        )

        prompt = config.USER_QUESTION_TEMPLATE.format(
            context_str=context_str,
            query_str=user_query
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in answer_user_query: {e}")
        return "Failed to get an answer."