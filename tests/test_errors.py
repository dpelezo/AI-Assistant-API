"""Tests for the user-facing error messages."""

import anthropic
import httpx
import pytest

from app.services.errors import BUSY, GENERIC, UNAVAILABLE, user_facing_error

REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def api_error(cls, status: int, message: str = "error"):
    return cls(message, response=httpx.Response(status, request=REQUEST), body=None)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (api_error(anthropic.AuthenticationError, 401), UNAVAILABLE),
        (api_error(anthropic.PermissionDeniedError, 403), UNAVAILABLE),
        (api_error(anthropic.NotFoundError, 404, "model: claude-old"), UNAVAILABLE),
        (
            api_error(
                anthropic.BadRequestError,
                400,
                "Your credit balance is too low to access the Anthropic API.",
            ),
            UNAVAILABLE,
        ),
        (api_error(anthropic.RateLimitError, 429), BUSY),
        (api_error(anthropic.InternalServerError, 500), BUSY),
        (api_error(anthropic.APIStatusError, 529, "Overloaded"), BUSY),
        (anthropic.APIConnectionError(request=REQUEST), BUSY),
        (ValueError("ANTHROPIC_API_KEY is not configured"), UNAVAILABLE),
        (api_error(anthropic.BadRequestError, 400, "messages: too long"), GENERIC),
        (RuntimeError("boom"), GENERIC),
    ],
)
def test_user_facing_error(exc, expected):
    assert user_facing_error(exc) == expected


def test_messages_never_include_exception_text():
    secret = "sk-ant-secret-key"
    assert secret not in user_facing_error(RuntimeError(secret))
