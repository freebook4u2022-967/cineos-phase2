# Film build pipeline

The `cineos.film` package provides an end-to-end, backend-neutral short-film coordinator. A `FilmBuild` records every render attempt, validation decision, recovery action, selected file, checksum, warning, failure, and exported artifact.

The sequence is validation, compilation, timeline-ordered rendering, strict shot validation, bounded recovery, assembly, and export. Resume only reuses an approved file when its SHA-256 checksum matches. Identity rejection and corrupt or missing media are never accepted automatically.

Assembly invokes FFmpeg as an argument vector (never through a shell), normalizes video to H.264/yuv420p, and fails explicitly when FFmpeg is absent. Subtitles are sidecar SRT/WebVTT files by default. Audio is optional and an empty track list means an intentional silent film.

## CLI

```console
cineos film build project.json --renderer local-ai --output-dir build --dry-run
cineos film status build/build.json
cineos film resume build/build.json
cineos film cancel BUILD_UUID
cineos film export build/build.json --output final.mp4
```

Dry-run compiles and validates project/package inputs and records shot, asset, CineDNA, renderer, and expected output information without loading a model or submitting a render.
