# Mission Zero

This fixture is deliberately limited to one character, one environment, one static
five-second shot, and deterministic seed `20260802`. The one selected backend is
`damo-vilab/text-to-video-ms-1.7b`, through the existing `local-ai` Atlas plugin.
Its supported output profile is 576x320 at 8 FPS; 24 FPS is not claimed for this
backend. The model path is explicit and local. CINEOS never downloads weights or
installs dependencies.

The repository's `hardware/hardware-report.json` is **not** accepted by this
workflow. Capture the actual workstation report at repository root:

```bash
cineos hardware-report --output hardware-report-local.json --verbose
cineos mission-zero preflight \
  --hardware hardware-report-local.json \
  --config examples/mission_zero/renderer-config.json
cineos mission-zero render \
  --project examples/mission_zero/project.json \
  --output-dir output/mission-zero
cineos mission-zero verify --output output/mission-zero/shot-001.mp4
```

## Explicit installation

Review versions for the workstation's NVIDIA driver, then run these commands
manually. They are documentation, not actions performed by CINEOS:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,hardware]'
python -m pip install 'torch>=2.2' 'diffusers>=0.30' 'transformers>=4.44' 'accelerate>=0.33' 'imageio-ffmpeg>=0.5'
sudo apt-get install ffmpeg
huggingface-cli download damo-vilab/text-to-video-ms-1.7b --local-dir models/text-to-video-ms-1.7b
```

Preflight requires measured Linux, NVIDIA GPU/VRAM, driver, CUDA, PyTorch CUDA,
available RAM/disk, FFmpeg, Python 3.12, and the local model index. If any field is
absent it fails rather than guessing. A failed preflight recommends at least an
8 GiB NVIDIA CUDA GPU, 16 GiB available system RAM, and 20 GiB disk. More headroom
is advisable. No three-shot film may be assembled until the generated MP4 passes
verification; identity consistency is not measured or claimed.
