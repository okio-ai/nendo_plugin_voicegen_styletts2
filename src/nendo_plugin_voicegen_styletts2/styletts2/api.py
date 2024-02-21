import os
from typing import Tuple, List

import librosa
import nltk
import numpy as np
import phonemizer
import torch
import torchaudio
import yaml
from nltk.tokenize import word_tokenize
from tortoise.utils.text import split_and_recombine_text
from tqdm import tqdm

from nendo_plugin_voicegen_styletts2.styletts2.Utils.PLBERT.util import load_plbert
from nendo_plugin_voicegen_styletts2.styletts2.Modules.diffusion.sampler import (
    DiffusionSampler,
    ADPM2Sampler,
    KarrasSchedule,
)
from nendo_plugin_voicegen_styletts2.styletts2.models import (
    load_ASR_models,
    load_F0_models,
    build_model,
)
from nendo_plugin_voicegen_styletts2.styletts2.text_utils import TextCleaner
from nendo_plugin_voicegen_styletts2.utils import download_model
from .utils import recursive_munch

nltk.download("punkt")


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Create a mask from lengths."""
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def preprocess(wave: np.ndarray) -> torch.Tensor:
    """Preprocess the audio for the StyleTTS2 model."""
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300
    )(wave_tensor)
    mean, std = -4, 4
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def load_models(
    path_prefix: str,
    tts2_download_path: str,
    bert_download_path: str,
    f0_download_path: str,
    asr_download_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, List, DiffusionSampler]:
    """Load the StyleTTS2 model and its dependencies."""
    config = yaml.safe_load(open(os.path.join(path_prefix, "styletts2_config.yaml")))

    # load pretrained ASR model
    ASR_config = config.get("ASR_config", False)
    ASR_config = os.path.join(path_prefix, ASR_config)
    ASR_path = config.get("ASR_path", False)
    ASR_path = os.path.join(path_prefix, ASR_path)
    if not os.path.isfile(ASR_path):
        download_model(asr_download_path, ASR_path)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get("F0_path", False)
    F0_path = os.path.join(path_prefix, F0_path)
    if not os.path.isfile(F0_path):
        download_model(f0_download_path, F0_path)
    pitch_extractor = load_F0_models(F0_path)

    BERT_path = config.get("PLBERT_dir", False)
    BERT_dir = os.path.join(path_prefix, BERT_path)
    BERT_path = os.path.join(BERT_dir, "step_1000000.t7")

    if not os.path.isfile(BERT_path):
        download_model(bert_download_path, BERT_path)
    plbert = load_plbert(BERT_dir)

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    tts2_path = "models/epochs_2nd_00020.pth"
    os.makedirs("models", exist_ok=True)
    if not os.path.isfile(tts2_path):
        download_model(tts2_download_path, tts2_path)
    params_whole = torch.load(tts2_path, map_location="cpu")
    params = params_whole["net"]

    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict

                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    #             except:
    #                 _load(params[key], model[key])
    _ = [model[key].eval() for key in model]

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(
            sigma_min=0.0001, sigma_max=3.0, rho=9.0
        ),  # empirical parameters
        clamp=False,
    )
    return model, model_params, sampler


class StyleTTS2:
    """StyleTTS2 model for voice generation wrapped in a simpler API compared to the original repo."""

    def __init__(
        self,
        tts2_download_path: str,
        bert_download_path: str,
        asr_download_path: str,
        f0_download_path: str,
    ):
        """Initialize the StyleTTS2 model."""
        self.path_prefix = os.path.dirname(__file__)
        self.textcleaner = TextCleaner()
        self.sr = 24000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        self.model, self.model_params, self.sampler = load_models(
            self.path_prefix,
            tts2_download_path=tts2_download_path,
            bert_download_path=bert_download_path,
            asr_download_path=asr_download_path,
            f0_download_path=f0_download_path,
            device=self.device,
        )
        self.pwd = os.path.dirname(os.path.realpath(__file__))

        self.voice_list = [
            "f-us-1",
            "f-us-2",
            "f-us-3",
            "f-us-4",
            "m-us-1",
            "m-us-2",
            "m-us-3",
            "m-us-4",
        ]
        self.voices = self._load_voices()

    def synthesize(
        self, text: str, voice: str, lngsteps: int
    ) -> Tuple[int, np.ndarray]:
        """Synthesize voice from text using the StyleTTS2 model.

        This method uses one of the 8 predefined voices for synthesis.

        Args:
            text (str): The text to synthesize.
            voice (str): The voice to use for synthesis.
            lngsteps (int): The number of diffusion steps to use for synthesis.

        Returns:
            Tuple[int, np.ndarray]: The sample rate and the synthesized audio.
        """
        texts = split_and_recombine_text(text)
        audios = []
        v = voice.lower()
        for t in tqdm(texts):
            audios.append(
                self._inference(
                    t,
                    self.voices[v],
                    alpha=0.3,
                    beta=0.7,
                    diffusion_steps=lngsteps,
                    embedding_scale=1,
                )
            )
        return self.sr, np.concatenate(audios)

    def clsynthesize(
        self, text: str, voice: np.ndarray, vcsteps: int
    ) -> Tuple[int, np.ndarray]:
        """Synthesize voice from text using the StyleTTS2 model.

        This method uses the voice from a track as the reference for synthesis.

        Args:
            text (str): The text to synthesize.
            voice (np.ndarray): The voice to use for synthesis.
            vcsteps (int): The number of diffusion steps to use for synthesis.

        Returns:
            Tuple[int, np.ndarray]: The sample rate and the synthesized audio.
        """

        texts = split_and_recombine_text(text)
        audios = []
        for t in tqdm(texts):
            audios.append(
                self._inference(
                    t,
                    self._compute_style(voice),
                    alpha=0.3,
                    beta=0.7,
                    diffusion_steps=vcsteps,
                    embedding_scale=1,
                )
            )
        return self.sr, np.concatenate(audios)

    def _load_voices(self):
        """Compute the style embeddings for the 8 builtin voices for the StyleTTS2 model."""
        voices = {}
        for v in self.voice_list:
            voices[v] = self._compute_style(
                os.path.join(self.path_prefix, f"voices/{v}.wav")
            )
        return voices

    def _compute_style(self, path: str) -> torch.Tensor:
        """Compute the style embedding for a voice file."""
        wave, sr = librosa.load(path, sr=self.sr)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != self.sr:
            audio = librosa.resample(audio, sr, self.sr)
        mel_tensor = preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def _inference(
        self,
        text: str,
        ref_s: torch.Tensor,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: int = 1,
        use_gruut: bool = False,
    ) -> np.ndarray:
        """Perform inference using the StyleTTS2 model."""
        text = text.strip()
        ps = self.phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)
        tokens = self.textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,  # reference from the same speaker as the embedding
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        result = (
            out.squeeze().cpu().numpy()[..., :-50]
        )  # weird pulse at the end of the model, need to be fixed later
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return result
