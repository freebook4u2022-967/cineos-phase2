# First backend: text-to-video-ms-1.7b

## Hardware-driven selection

The latest report, `hardware/hardware-report.json`, was generated and inspected
on 2026-08-01 before implementation. It reports Linux x86_64, 19.25 GB RAM,
30.42 GB free disk, no GPU, no CUDA/driver, no PyTorch, and no FFmpeg. Therefore
we selected the Diffusers `damo-vilab/text-to-video-ms-1.7b` text-to-video
pipeline in **CPU mode**. Unlike CUDA-only alternatives, its PyTorch operations
can execute on this operating system without inventing GPU capabilities. The
repository host is not currently ready to infer: explicit Python dependencies,
FFmpeg, and local model files are intentionally absent.

## Honest limitations

The adapter supports text-to-video only: 576x320, 8 FPS, and at most four
seconds. CPU inference can take hours and 19 GB RAM leaves little headroom.
CUDA is optional, but when requested the validator requires a working NVIDIA
driver/PyTorch CUDA stack and at least 8 GiB VRAM. This model does **not** offer
image-reference, multi-reference, face identity, control-image, motion-reference,
or CineDNA identity conditioning. Such packages are rejected rather than
silently degrading identity. This integration makes no quality comparison with
commercial systems and does not claim Seedance-level results.

No weights are redistributed or automatically downloaded. Review the upstream
model license and model card before use.
