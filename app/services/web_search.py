# app/services/web_search.py
import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_exa import ExaSearchRetriever

from app.core.config import settings

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Collapse runs of whitespace so snippets read as normal sentences."""
    return re.sub(r"\s+", " ", text).strip()


def _highlights_text(highlights: Any) -> str:
    if isinstance(highlights, list | tuple):
        parts = [clean_text(str(h)) for h in highlights if str(h).strip()]
        return "\n".join(parts)
    if highlights:
        return clean_text(str(highlights))
    return ""


def format_sources(docs: list[Document]) -> str:
    """Render Exa results as plain-text <source> blocks.

    Highlights are joined as readable text, not a Python list, so neither the
    model nor the research parser sees list brackets, quotes or escaped
    newlines.
    """
    blocks = []
    for doc in docs:
        url = doc.metadata.get("url", "")
        text = _highlights_text(doc.metadata.get("highlights")) or clean_text(doc.page_content)
        if not text:
            continue
        blocks.append(f"<source>\n<url>{url}</url>\n<highlights>\n{text}\n</highlights>\n</source>")
    return "\n\n".join(blocks)


@tool
def retrieve_web_content(query: str) -> str:
    """Fetch the top 6 web results for a query, with highlighted passages, via Exa."""
    logger.info("Retrieving web content for query: %s", query)
    retriever = ExaSearchRetriever(
        k=6,
        highlights=True,
        exa_api_key=settings.EXA_API_KEY,
    )
    docs = retriever.invoke(query)
    logger.info("Retrieved %d documents", len(docs))
    return format_sources(docs)
