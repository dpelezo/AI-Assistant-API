# app/services/web_search.py
from typing import List
from langchain_exa import ExaSearchRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from app.core.config import settings

@tool
def retrieve_web_content(query: str) -> List[str]:
    """Fetch top-6 highlighted documents via ExaSearch."""
    print(f"Retrieving web content for query: {query}")
    retriever = ExaSearchRetriever(
        k=6,
        highlights=True,
        exa_api_key=settings.EXA_API_KEY
    )

    document_prompt = PromptTemplate.from_template(
        """
        <source>
            <url>{url}</url>
            <highlights>{highlights}</highlights>
        </source>
        """
    )
    document_chain = (
        RunnableLambda(
            lambda doc: {
                "highlights": doc.metadata.get("highlights", "No highlights"),
                "url": doc.metadata["url"],
            }
        )
        | document_prompt
    )
    retrieval_chain = retriever | document_chain.map()
    docs = retrieval_chain.invoke(query)
    print(f"Retrieved {len(docs)} documents")
    return docs
