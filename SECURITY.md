# Security policy

## Reporting a vulnerability

Use the repository's [Security tab](https://github.com/TrustifAI/typed_evals/security).
If **Report a vulnerability** is available, submit a private report there. Include
the affected package version, a minimal reproduction using synthetic data, the
expected behavior, and the impact. Do not include credentials or real user data.

If private vulnerability reporting is unavailable, open a
[GitHub issue](https://github.com/TrustifAI/typed_evals/issues/new) asking maintainers
for a private reporting channel. Keep that public issue free of exploit details
and sensitive data. Use public issues for ordinary bugs that do not expose a
security vulnerability.

## Scope and data handling

Security reports are welcome for guard bypasses, execution of arguments other
than those reviewed, unsafe artifact or dataset handling, and unintended exposure
of application evidence or credentials.

Evaluation sends fields selected by active metrics to your configured judge
provider. Evidence is not automatically redacted. Supply application-owned
authorization facts and keep secrets out of sample evidence. Review the
provider's data handling before sending sensitive material.

Runtime guards enforce configured judge decisions at explicit checkpoints.
Judgments can be wrong, and wrapping an agent entrypoint does not intercept hidden
tools. Keep deterministic permissions, schema validation, and current-state checks
in your application. A check after execution cannot reverse side effects. The
[runtime contract](docs/RUNTIME.md#policy-and-failure-semantics) describes these
boundaries; suspected implementation failures within that contract should be reported.

When reporting, specify whether the issue reproduces with the latest available
package or the current main branch. This project is alpha; no support schedule for
older versions or fixed response time is currently published.
