"""A nendo plugin for text to speech based on StyleTTS2."""
from typing import Any, Optional

from nendo import Nendo, NendoGeneratePlugin, NendoConfig, NendoTrack, NendoError
from .config import VoicegenStyletts2Config
from .styletts2.api import StyleTTS2

settings = VoicegenStyletts2Config()


class VoicegenStyletts2(NendoGeneratePlugin):
    """A nendo plugin for text to speech based on StyleTTS2.

    Examples:
        ```python
        from nendo import Nendo, NendoConfig

        nendo = Nendo(config=NendoConfig(plugins=["nendo_plugin_voicegen_styletts2"]))
        generated_voice = nendo.plugins.musicgen(
            text="Hello from the nendo tts plugin!",
            voice="us-male-2",
        )
        generated_voice.play()
        ```
    """

    nendo_instance: Nendo = None
    config: NendoConfig = None
    model: StyleTTS2 = None

    def __init__(self, **data: Any):
        """Initialize the plugin with the StyleTTS2 model."""
        super().__init__(**data)
        self.model = StyleTTS2(
            tts2_download_path=settings.tts_model_link,
            asr_download_path=settings.asr_model_link,
            f0_download_path=settings.f0_model_link,
            bert_download_path=settings.bert_model_link,
        )

    @NendoGeneratePlugin.run_track
    def run_plugin(
        self,
        track: Optional[NendoTrack] = None,
        text: str = "Hello from nendo!",
        voice: str = "f-us-1",
        diffusion_steps: int = settings.diffusion_steps,
    ) -> NendoTrack:
        """Generate a voice track from text using StyleTTS2.

        Optionally can take in a track to use as a style reference for zero-shot voice cloning.

        Args:
            track (Optional[NendoTrack], optional): A track to use as a style reference. Defaults to None.
            text (str, optional): The text to synthesize. Defaults to "Hello from nendo!".
            voice (str, optional): The voice to use for synthesis. Defaults to "f-us-1".
            diffusion_steps (int, optional): The number of diffusion steps to use. Defaults to settings.diffusion_steps.

        Returns:
            NendoTrack: The generated voice track.
        """
        if track is not None:
            # use track as style reference
            sr, audio = self.model.clsynthesize(
                text, track.resource.src, diffusion_steps
            )
            return self.nendo_instance.library.add_related_track_from_signal(
                audio,
                sr,
                related_track_id=track.id,
                track_type="voice",
                meta={
                    "text": text,
                    "diffusion_steps": diffusion_steps,
                },
            )

        if voice not in self.model.voice_list:
            raise NendoError(
                f"Voice {voice} not found in voice list. Please use one of {self.model.voice_list}"
            )

        sr, audio = self.model.synthesize(text, voice, diffusion_steps)

        return self.nendo_instance.library.add_track_from_signal(
            audio,
            sr,
            track_type="voice",
            meta={
                "text": text,
                "voice": voice,
                "diffusion_steps": diffusion_steps,
            },
        )
