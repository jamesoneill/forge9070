# Security policy

Forge9070 is experimental. Security fixes are developed on `main`; older
snapshots do not have a promised maintenance window or response-time guarantee.
Check [VALIDATION.md](VALIDATION.md) for the limits of completed testing.

## Report a vulnerability privately

Use **Security → Report a vulnerability** on the repository when GitHub private
vulnerability reporting is enabled:

[Open a private security report](https://github.com/jamesoneill/forge9070/security/advisories/new).

Include the affected commit/version, a minimal reproduction, expected impact,
relevant platform details and a suggested fix if available. Use synthetic data;
do not attach live credentials, private model access tokens or personal prompts.

If the private reporting option is unavailable, open a minimal issue asking
@jamesoneill for a private reporting channel. **Do not include exploit details
or sensitive data in that public issue.** No separate security email address
or guaranteed response time is currently advertised.

## Deployment expectations

- Keep the service on its default loopback interface unless remote access is
  deliberately configured. Treat API access as access to local model management
  and inference, not as an untrusted public multi-tenant service.
- The native service accepts `--api-key`; binding beyond loopback requires at
  least 16 characters. The built-in HTTP server does not provide TLS. Use a
  properly configured authenticated TLS boundary for remote access.
- The UI shell can be public while API routes enforce the configured key.
  Saved native profiles exclude the API key. Protect credentials passed through
  launch commands, process inspection, shell history and logs.
- `HF_TOKEN` and `STRATA_API_KEY` belong to the compatibility path and have
  different purposes. They are not interchangeable with the native engine's
  `--api-key`. See the [README](README.md) and [native guide](docs/NATIVE-ENGINE.md).
- Do not publish local profiles, model paths containing personal information,
  prompt histories or unredacted diagnostic bundles.

## Model and client boundaries

GGUF structure checks are input validation, not a sandbox or proof that every
malformed file is harmless. Obtain model files from a source you trust and
retain their provenance. Model names and an “uncensored” label do not establish
compatibility, safety or output quality.

The API returns model-generated tool requests; it does not execute those tools.
Coding clients are responsible for authorization, filesystem access and user
approval before acting on model output. Generated text is untrusted input.

Potential security reports include authentication bypass, unintended file access,
unsafe handling of malformed models or requests, credential disclosure and
unexpected execution. Ordinary load failures, missing GPU support and performance
questions belong in the issue tracker with redacted details.
