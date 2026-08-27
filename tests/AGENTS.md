# tests/

Integration tests that span more than one app.

Everything else lives beside the code it covers. These do not, because they import both
`editgpt_gateway` and `editgpt_worker`, and neither package depends on the other — that
one-way boundary is deliberate (the gateway must never pull in the worker's model stack)
and putting a test in either directory would quietly break it.

What belongs here: a flow that crosses a process boundary in production. What does not:
anything one package can prove on its own.
