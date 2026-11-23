"""Turn internal errors into short messages that are safe to show users.

The full exception is always logged where it is caught. Users never see
provider error text, API keys or stack traces.
"""

import anthropic

UNAVAILABLE = (
    "The assistant is temporarily unavailable. Please try again later."
)
BUSY = "The assistant is busy right now. Please try again in a minute."
GENERIC = "Something went wrong while preparing the answer. Please try again."

# Errors that will not fix themselves on retry: no credit, a bad or missing
# key, a retired model, or a missing setting.
_UNAVAILABLE_TYPES: tuple[type[BaseException], ...] = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
)

# Errors that usually clear up if the user waits.
_BUSY_TYPES: tuple[type[BaseException], ...] = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


def user_facing_error(exc: BaseException) -> str:
    """Return a friendly message for an exception raised while answering."""
    if isinstance(exc, _UNAVAILABLE_TYPES):
        return UNAVAILABLE
    if isinstance(exc, anthropic.BadRequestError) and "credit balance" in str(exc).lower():
        return UNAVAILABLE
    if isinstance(exc, _BUSY_TYPES):
        return BUSY
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code == 529:
        return BUSY  # Anthropic "overloaded"
    if isinstance(exc, ValueError) and "not configured" in str(exc):
        return UNAVAILABLE
    return GENERIC
