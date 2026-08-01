# Local AI renderer setup

Nothing in CINEOS installs dependencies or downloads weights. On Linux, create
an isolated environment and deliberately run:

```bash
python -m pip install 'torch>=2.2' 'diffusers>=0.30' 'transformers>=4.44' 'accelerate>=0.33' 'imageio-ffmpeg>=0.5'
sudo apt-get install ffmpeg
huggingface-cli download damo-vilab/text-to-video-ms-1.7b \
  --local-dir models/text-to-video-ms-1.7b
cp examples/first_real_shot/renderer.local-ai.json .
cineos renderer validate local-ai
```

For CUDA, install the PyTorch build matching the installed NVIDIA driver/CUDA
runtime, set `device` to `cuda`, use `float16`, and validate again. One GPU only
is supported. `model_path` must contain `model_index.json`; remote identifiers
are deliberately rejected by validation.

```bash
cineos renderer list
cineos renderer inspect local-ai
cineos renderer validate local-ai --config renderer.local-ai.json
cineos renderer render film-package.json --shot shot-001 \
  --conditioning conditioning.json --output renders/shot-001.mp4 --dry-run
cineos renderer render film-package.json --shot shot-001 \
  --conditioning conditioning.json --output renders/shot-001.mp4
```

Dry-run validates package integrity, the shot, ConditioningPackage capabilities,
approved asset IDs, identity constraints, output settings, and backend limits
without loading the model. Run `validate` separately for the complete host check.
