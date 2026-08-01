"""Explicit setup guidance; this module never installs software."""

SETUP_COMMANDS = (
    "python -m pip install 'torch>=2.2' 'diffusers>=0.30' "
    "'transformers>=4.44' 'accelerate>=0.33' 'imageio-ffmpeg>=0.5'",
    "sudo apt-get install ffmpeg",
    "huggingface-cli download damo-vilab/text-to-video-ms-1.7b "
    "--local-dir models/text-to-video-ms-1.7b",
)


def setup_commands() -> tuple[str, ...]:
    return SETUP_COMMANDS
