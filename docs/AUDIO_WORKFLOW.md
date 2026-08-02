# Audio Workflow

```console
cineos audio plan project.json --output audio-project.json
cineos audio cast audio-project.json --dry-run
cineos audio synthesize audio-project.json --provider fake-deterministic --output-dir synthesis
cineos audio mix audio-project.json --output final-audio.wav
cineos audio export audio-project.json --output-dir audio-export
cineos audio inspect audio-project.json --json
```

Plan after NOVA direction and approved shots. Review dialogue/casting and cue
alignment, run dry-run preflight, synthesize through an adapter, mix, then export.
Dry-run reports casting, provider language support, assets, overlap state, FFmpeg,
and expected outputs without synthesis or mixing. FilmBuild attaches the mix or
records silent fallback; Studio exposes planning, provider selection, validation,
and export through its controller.
