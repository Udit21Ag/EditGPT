"""Who a request is from.

Identity is decided here and nowhere else. Every route takes `PrincipalDep` and passes it
to the store; a route that leans on the store's default parameter behaves identically
today and leaks every job the moment a second user exists.

## Two modes, chosen by configuration

**Clerk configured** (`EDITGPT_CLERK_SECRET_KEY` present) — the session token is verified
and mapped to a row in `users`. An absent or invalid token is a **401**, never a fall back
to the shared account: a system that treats a bad credential as "no credential" turns any
expired token into a way in.

**Clerk absent** — every request is the anonymous sentinel. This is what a fresh checkout
and the test suite run as, so neither needs a credential. `GET /ready` says which mode is
live, because "authentication is off" is exactly the thing that must not be silent.

## Verification is networkless when it can be

`authenticate_request` fetches Clerk's JWKS over the network unless it is given the
instance's public key. Setting `EDITGPT_CLERK_JWT_KEY` makes verification pure local
arithmetic: no per-request latency, no dependency on Clerk being reachable, and no socket
for the test suite to block. Without it the SDK fetches and caches the key for five
minutes, which works but couples every cold start to a third party.

Only `session_token` is accepted. Clerk also issues API keys and machine tokens; this
service has no notion of either, and accepting a token type you do not model is how a
credential meant for something else becomes a login.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from editgpt_store import ANONYMOUS_USER_ID, User
from fastapi import Depends, HTTPException, Request

from editgpt_gateway.deps import Services, ServicesDep
from editgpt_gateway.settings import Settings

log = logging.getLogger(__name__)

SESSION_TOKEN = "session_token"


class NotAuthenticatedError(HTTPException):
    """A 401 that always carries `WWW-Authenticate`, because the RFC requires it.

    Without the header a browser has no idea what to do with the response, and a client
    library cannot tell "you are not signed in" from "this endpoint is broken".
    """

    def __init__(self, detail: str) -> None:
        super().__init__(401, detail, headers={"WWW-Authenticate": "Bearer"})


@dataclass(frozen=True, slots=True)
class Identity:
    """A verified caller: the subject Clerk vouched for, and our row for them."""

    user_id: UUID
    external_id: str | None
    """Clerk's `sub`. `None` for the anonymous sentinel, which no provider issued."""

    @property
    def is_anonymous(self) -> bool:
        return self.external_id is None


def verify(request: Request, settings: Settings) -> str:
    """Verify the request's session token and return Clerk's subject, or raise 401."""
    from clerk_backend_api.security.authenticaterequest import authenticate_request
    from clerk_backend_api.security.types import AuthenticateRequestOptions

    state = authenticate_request(
        request,
        AuthenticateRequestOptions(
            secret_key=settings.clerk_secret_key,
            jwt_key=settings.clerk_jwt_key or None,
            authorized_parties=list(settings.clerk_authorized_parties) or None,
            # Restricted deliberately — see the module docstring.
            accepts_token=[SESSION_TOKEN],
        ),
    )
    if not state.is_signed_in:
        # `state.reason` distinguishes "no token" from "expired" from "wrong party", and
        # it is safe to return: it describes the credential the caller sent, not anything
        # about the account it failed to reach.
        log.info("auth.rejected", extra={"reason": str(state.reason)})
        raise NotAuthenticatedError(state.message or "sign in to use this endpoint")

    subject = (state.payload or {}).get("sub")
    if not isinstance(subject, str) or not subject:
        # A signed-in state with no subject should be impossible. Failing loudly beats
        # inventing an identity for it.
        raise NotAuthenticatedError("the session token carries no subject")
    return subject


def user_for(services: Services, external_id: str) -> UUID:
    """Our `users.id` for a Clerk subject, creating the row on first sight.

    Clerk owns the account; this table owns the foreign key every other table points at.
    Provisioning on first authenticated request rather than through a webhook keeps the
    two from drifting: a user cannot reach this code without a row existing by the end
    of it, whereas a missed webhook leaves a signed-in user whose jobs cannot be stored.

    Idempotent under a race: two concurrent first requests both try to insert, the loser
    hits the unique constraint on `external_id`, and re-reads.
    """
    factory = getattr(services.jobs, "session_factory", None)
    if factory is None:
        # No database. The gateway is in its degraded in-memory mode, which `/ready`
        # already reports; there is nowhere to provision and nothing to point at.
        return ANONYMOUS_USER_ID

    with factory() as session:
        found = session.scalars(
            sa.select(User).where(User.external_id == external_id)
        ).one_or_none()
        if found is not None:
            return UUID(str(found.id))

        made = User(id=uuid4(), external_id=external_id)
        session.add(made)
        try:
            session.commit()
        except sa.exc.IntegrityError:
            session.rollback()
            raced = session.scalars(
                sa.select(User).where(User.external_id == external_id)
            ).one_or_none()
            if raced is None:
                raise
            return UUID(str(raced.id))
        log.info("auth.provisioned", extra={"user_id": str(made.id)})
        return made.id


def current_identity(request: Request, svc: ServicesDep) -> Identity:
    """The caller, verified if authentication is on and anonymous if it is not."""
    settings = svc.settings
    if not settings.uses_clerk:
        return Identity(user_id=ANONYMOUS_USER_ID, external_id=None)

    subject = verify(request, settings)
    return Identity(user_id=user_for(svc, subject), external_id=subject)


def current_principal(identity: Annotated[Identity, Depends(current_identity)]) -> UUID:
    """Just the owner id, which is all most routes need."""
    return identity.user_id


IdentityDep = Annotated[Identity, Depends(current_identity)]
PrincipalDep = Annotated[UUID, Depends(current_principal)]
"""Every route that touches a user's data takes one of these.

Passed explicitly to the store rather than relying on its default, so a route that forgets
is a visible omission instead of an invisible one.
"""


def describe(settings: Settings) -> dict[str, Any]:
    """What `/ready` reports about authentication. Never includes a key."""
    return {
        "provider": "clerk" if settings.uses_clerk else "none",
        "networkless_verification": bool(settings.clerk_jwt_key),
    }
