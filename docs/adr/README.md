# Decisions

One file per decision that closed off alternatives, with the evidence that closed them.
Write one with `/adr`. Never edit an accepted ADR's decision — supersede it with a new
one and link back, so the reasoning stays auditable.

| # | Decision | Status |
|---|---|---|
| [0001](0001-model-routing.md) | Model selection and local/remote routing | accepted, extended after Phase 1 |
| [0002](0002-text-grounding.md) | Grounding DINO replaces CLIPSeg as the text lane | accepted |
| [0003](0003-ask-when-unsure.md) | Offer candidates instead of guessing | accepted |
| [0004](0004-content-safety.md) | No content classifier; the generative provider is the control | accepted |

## When an ADR is warranted

- a model, provider or library was chosen over a named alternative
- a threshold was set from measurement
- a planned capability was cut or deferred
- an approach was tried and abandoned — **especially this one**; the next person will
  otherwise try it again

## When it is not

Anything reversible in an afternoon. An ADR for a variable rename devalues the ones that
matter.
