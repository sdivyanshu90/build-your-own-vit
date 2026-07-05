# Security

Security posture, threat model, and the controls implemented in this project.

## Threat model (serving)

| Threat | Vector | Mitigation |
|--------|--------|------------|
| Malicious/oversized upload | `POST /v1/predict` | Content-type allowlist (415), hard size cap with early rejection (413), safe decode via Pillow with error handling (400). |
| Decompression bomb | Crafted image | Bounded read (`VIT_MAX_UPLOAD_BYTES`) before decode; Pillow `load()` under try/except. |
| Unauthorized access | Public endpoint | Optional bearer token (`VIT_API_TOKEN`), constant-time compare. |
| Abuse / DoS | Request floods | Per-IP sliding-window rate limit + `Retry-After`; enforce at gateway for multi-replica. |
| Info leakage | Error messages/stack traces | Uniform error envelope; 500s log server-side, return a generic message; no traceback leakage. |
| Clickjacking / MIME sniffing | Browser context | `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, restrictive CSP. |
| SSRF | — | Service performs no outbound requests from user input. |
| Untrusted checkpoint | `torch.load` (pickle) | **Trust boundary** — only load checkpoints you produced/trust; keep them in access-controlled storage (see below). |

## OWASP mapping

- **A01 Broken Access Control** — bearer auth on the predict endpoint; health/
  metadata intentionally open for probes.
- **A02 Cryptographic Failures** — no secrets in code; TLS at ingress; tokens
  from env/secret store; constant-time token comparison.
- **A03 Injection** — no SQL; inputs are binary images validated before use.
- **A04 Insecure Design** — liveness/readiness split; fail-closed model loading.
- **A05 Security Misconfiguration** — secure headers by default; non-root
  container; least-privilege `securityContext` (drop caps, read-only rootfs).
- **A06 Vulnerable Components** — pinned deps, Dependabot-friendly layout, image
  scanning recommended in CI.
- **A08 Software & Data Integrity** — atomic checkpoints; supply-chain notes below.
- **A09 Logging & Monitoring** — structured JSON logs with request IDs; Prometheus
  metrics; see [operations.md](operations.md).
- **A10 SSRF** — no user-controlled outbound requests.

## Secrets management

- Never commit secrets. `.env` is git-ignored; only `.env.example` is tracked.
- In Kubernetes, inject `VIT_API_TOKEN` from a `Secret` (see the manifest).
- Rotate tokens by updating the secret and restarting pods (rolling, zero-downtime).

## The `torch.load` trust boundary

Checkpoints are Python pickles. Loading one is equivalent to running code from
its author. Controls:

1. Only load checkpoints your pipeline produced or that come from a trusted,
   integrity-verified source.
2. Store checkpoints in access-controlled, versioned object storage; verify a
   checksum before deploy.
3. Treat "checkpoint provenance" like "container image provenance".

The self-describing checkpoint uses only plain Python types plus tensors, but
the pickle trust boundary still applies.

## Container & supply-chain hardening

- **Non-root** runtime user; `allowPrivilegeEscalation: false`; `readOnlyRootFilesystem`;
  all Linux capabilities dropped; `seccompProfile: RuntimeDefault`.
- **Minimal base** (`python:slim`) with only required OS libs.
- **Pinned dependencies** via `pyproject.toml`; reproducible builds.
- **Recommended CI additions**: `pip-audit` for Python CVEs, Trivy/Grype for the
  image, and pinning images by digest. Sign artifacts with cosign if publishing.

## Reporting a vulnerability

Please open a private security advisory rather than a public issue. Include a
reproduction and affected versions.
