"""Config for the StyleTTS2 plugin."""
from nendo import NendoConfig
from pydantic import Field


class VoicegenStyletts2Config(NendoConfig):
    """Default settings for the StyleTTS2 plugin."""

    tts_model_link: str = Field(
        "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth?download=true"
    )
    asr_model_link: str = Field(
        "https://github.com/yl4579/StyleTTS2/raw/main/Utils/ASR/epoch_00080.pth"
    )
    f0_model_link: str = Field(
        "https://github.com/yl4579/StyleTTS2/raw/main/Utils/JDC/bst.t7"
    )
    bert_model_link: str = Field(
        "https://github.com/yl4579/StyleTTS2/raw/main/Utils/PLBERT/step_1000000.t7"
    )
    diffusion_steps: int = Field(15)
