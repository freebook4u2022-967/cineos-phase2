# Quality benchmarks

CINEOS 0.1 Alpha ships a versioned, deterministic benchmark contract. `cineos
benchmark list` identifies the built-in suite; run the CPU-safe subset with
`cineos benchmark run --suite alpha --mandatory-only --output-dir reports`.
The runner records whether fixture execution succeeded and labels every value as
measured, estimated, unsupported, unavailable, or manually reviewed. It never
presents unsupported renderer quality as a result.

Create an **unapproved** immutable baseline with `benchmark baseline-create`.
Approval requires the explicit Python `approve_baseline(path, approver)` call and
human review. Existing baseline files are never overwritten. Compare reports
with `benchmark compare`; blocking regressions fail the command. Real-renderer
quality and performance runs are optional and hardware-gated.
