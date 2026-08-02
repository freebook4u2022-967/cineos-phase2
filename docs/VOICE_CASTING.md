# Voice Casting

A `VoiceProfile` is approved identity metadata, not a model or embedding. It binds
one character to language, accent, range, cadence, timbre, emotional range,
approved references, a non-secret provider configuration reference, rights state,
and a deterministic content hash.

`VoiceCasting.assign` preserves one profile per character throughout every scene.
Conflicting assignments fail. A scene override is accepted only with an explicit
`approved_override=True`. Missing identity or rights data is rejected rather than
inferred or invented. Voice cloning adapters must perform the same rights check.
