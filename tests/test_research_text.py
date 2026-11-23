"""Tests for search-result formatting and the research fallback report."""

import asyncio

from langchain_core.documents import Document

from app.services import deep_research
from app.services.web_search import format_sources

DOCS = [
    Document(
        page_content="",
        metadata={
            "url": "https://dictionary.cambridge.org/dictionary/english/hello",
            "highlights": [
                "used when meeting or greeting someone:\n\nHello, Paul. I haven't seen you for months.",
                "  something that is said at the beginning of a phone conversation  ",
            ],
        },
    ),
    Document(
        page_content="Fallback page text",
        metadata={"url": "https://example.com/hello", "highlights": []},
    ),
]


def run(coro):
    return asyncio.run(coro)


def test_format_sources_gives_clean_text():
    text = format_sources(DOCS)

    assert "I haven't seen you for months." in text
    assert "\\n" not in text  # no escaped newlines
    assert "['" not in text  # no Python list syntax
    assert "<url>https://example.com/hello</url>" in text
    assert "Fallback page text" in text


def test_contexts_keep_apostrophes_and_source_urls():
    contexts = run(deep_research.process_search_results("hello", "hello", format_sources(DOCS)))

    assert len(contexts) == 2
    first_text, _, first_url = contexts[0].partition(deep_research.SOURCE_SEPARATOR)
    assert "I haven't seen you for months." in first_text
    assert first_url == "https://dictionary.cambridge.org/dictionary/english/hello"


def test_fallback_report_lists_sources_when_the_model_fails():
    # conftest makes get_model raise, as it would with no API credit.
    contexts = run(deep_research.process_search_results("hello", "hello", format_sources(DOCS)))
    report = run(deep_research.generate_final_report("hello", contexts))

    assert report.startswith('I couldn\'t write a full report on "hello" right now')
    assert "1. [dictionary.cambridge.org](https://dictionary.cambridge.org/dictionary/english/hello)" in report
    assert "2. [example.com](https://example.com/hello)" in report
    assert "Finding 1" not in report
    assert "\\n" not in report


def test_fallback_report_trims_long_excerpts():
    context = ("word " * 200).strip() + deep_research.SOURCE_SEPARATOR + "https://example.com"
    report = deep_research.sources_fallback_report("q", [context])
    excerpt = report.splitlines()[-1].strip()
    assert len(excerpt) <= 301
    assert excerpt.endswith("…")


def test_deep_research_without_model_returns_clean_sources(monkeypatch):
    from app.core.config import settings

    class FakeSearch:
        def invoke(self, query):
            return format_sources(DOCS)

    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(settings, "EXA_API_KEY", "test")
    monkeypatch.setattr(deep_research, "retrieve_web_content", FakeSearch())

    result = run(deep_research.deep_research("hello", iteration_limit=3))

    assert result["search_queries"] == ["hello"]
    assert result["report"].startswith('I couldn\'t write a full report on "hello"')
    assert "I haven't seen you for months." in result["report"]


def test_final_report_uses_text_from_content_blocks(monkeypatch):
    from langchain_core.messages import AIMessage

    class ThinkingModel:
        async def ainvoke(self, messages):
            return AIMessage(content=[
                {"type": "thinking", "thinking": "hidden", "signature": "sig"},
                {"type": "text", "text": "# Hello\n\nA greeting."},
            ])

    monkeypatch.setattr(deep_research, "get_model", lambda: ThinkingModel())
    report = run(deep_research.generate_final_report("hello", ["context" + deep_research.SOURCE_SEPARATOR + "https://example.com"]))
    assert report == "# Hello\n\nA greeting."
