# Mission One T4 rendering reliability

## What changed

The first Colab run completed diffusion, encoded three MP4 files, and assembled a film, but the individual files were effectively black. That evidence distinguishes successful inference *execution* from valid visual output. It does not prove a single root cause. The symptom is consistent with a faulty or incompatible render path under the initial Mission One T4 configuration.

A Tesla T4 has a constrained 16 GiB VRAM envelope. Mission One now loads `THUDM/CogVideoX-2b` in float16 with low-CPU-memory loading, sequential CPU offload, VAE tiling, and VAE slicing. The pipeline is never moved wholesale to CUDA. Seeds remain deterministic and use a CPU generator.

## Reproducible Colab environment

The notebook pins `diffusers==0.30.3`, `transformers==4.44.2`, `accelerate==0.34.2`, `sentencepiece==0.2.0`, `imageio[ffmpeg]==2.35.1`, and `safetensors==0.4.5`. It records the GPU, total/free VRAM, PyTorch, Diffusers, and CUDA runtime. Critically low free VRAM triggers garbage collection and one preflight retry before a clear failure.

## Smoke workflow

Set `MISSION_ONE_SMOKE_MODE = True` before uploading the package. Only shot 1 is rendered with conservative settings, validated, and then stopped without assembly. Use this inexpensive check whenever the Colab image or GPU changes. The production default is `False`.

## Content gate and controlled retry

After each encode, FFprobe confirms file size, duration, dimensions, and decoded frame count. FFmpeg samples three grayscale frames. Mean luminance and luminance variance identify near-black output; inter-frame pixel differences conservatively identify obviously frozen output. These measurements diagnose validity, not artistic quality. Static-camera footage is not itself rejected, though truly identical samples are conservatively flagged for review/failure.

A black result receives exactly one retry after memory cleanup using the documented safer profile (reduced frame count/resolution). The prompt and seed never change. Both original and retry settings and the reason are written to `render-results.json`. Other error types are not silently retried.

Assembly occurs only when every mandatory production shot has status `valid`. Otherwise no successful `final-film.mp4` is advertised, while render results, verification report, failed media, and a diagnostics archive remain downloadable.

## Remaining limitations

CogVideoX-2B text-to-video does not consume the packaged reference images in this backend. Lip synchronization remains approximate unless separately measured. CPU offload is slower, T4 availability and memory fragmentation vary between Colab sessions, and automated pixel checks cannot establish narrative or aesthetic quality; human review remains necessary after technical validation.
