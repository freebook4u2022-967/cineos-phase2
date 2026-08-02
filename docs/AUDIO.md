# Voice and Audio Production

CINEOS audio is a renderer-independent production layer. An `AudioProject` records
Film Package identity, language, timing, dialogue, ambience, effects, music,
subtitles, future-renderer lip-sync metadata, mix policy, and deliverables. Its
content hash excludes its UUID and is stable for identical production inputs.

Provider adapters implement capability-declared text-to-speech. The base system
requires no network or proprietary service; `fake-deterministic` writes offline
PCM tones for testing. Provider configuration in a project is a reference only:
never put credentials or API keys in project JSON.

## Safety

Only profiles with explicit rights approval and approved reference IDs can be
cast. CINEOS neither infers identity from recordings nor imitates a real person
without approval. Music cues describe user, licensed, library, or original assets;
the base implementation generates no music and never invents copyrighted audio.
Lip-sync export is timing metadata only, not visual lip-sync rendering.

## Outputs

Exports include mixed WAV (and FFmpeg-converted AAC/M4A when available), stems,
a cue sheet, subtitle timing, lip-sync metadata, a production report, and SHA-256
checksums. A valid silent WAV is the fallback when no source audio is available.
