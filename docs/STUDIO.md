# CINEOS Studio Alpha

Studio Alpha is the first native desktop shell for CINEOS. It uses PySide6 and the
existing Core, Compiler, Atlas Runtime, Renderer, Validation, Recovery, and Film
Build APIs. It is an alpha workflow surface, not a production nonlinear editor.

## Install and launch

Python 3.12 is required. Install the package and launch the registered command:

```console
python -m pip install -e .
cineos-studio
```

Qt supports Windows, Linux, and macOS. For headless smoke checks set
`QT_QPA_PLATFORM=offscreen`. Studio stores window geometry, recent paths, renderer
selection, and non-sensitive preferences with `QSettings`; it never stores tokens
or passwords.

## Workspace

The tabbed workspace exposes project metadata, asset/reference discovery,
scene/shot ordering, the timeline, renderer readiness, render queue, validation
review, recovery history, and exports. The build toolbar exposes validation,
FilmPackage compilation, conditioning, dry-run, selected-shot render, complete
build/resume/cancel, and export. Operations requiring an installed renderer remain
disabled by configuration rather than silently selecting a backend.

Project files use the versioned `cineos-project-1` JSON envelope. Compiler output
is a separate canonical FilmPackage. Domain validation and compilation are always
delegated to `ProjectValidator` and `FilmCompiler`.

## Safety and failures

Unsaved changes prompt before replacement or exit. Expensive operations use Qt's
thread pool, signals, and cooperative cancellation. Errors are shown in the UI and
cover missing FFmpeg/assets/references/models, unsupported hardware/renderers,
validation or render failure, exhausted recovery, corrupt output, and permissions.
