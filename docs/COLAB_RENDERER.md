# Colab renderer

Export with `cineos mission-one export-colab build/package.json --output mission-one.zip`. The archive contains canonical prompts, negative prompts, seeds, frame counts, settings, approved references, and optional dialogue audio—never credentials. The notebook checks CUDA/GPU compatibility, loads `THUDM/CogVideoX-2b`, renders shots sequentially, frees cache, assembles with FFmpeg, and emits results and verification JSON.

CogVideoX-2B text-to-video does **not** consume packaged character/environment references. They are retained for review and future compatible renderers, and reports explicitly mark reference conditioning unsupported.
