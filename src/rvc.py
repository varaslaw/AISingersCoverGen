from functools import lru_cache
from multiprocessing import cpu_count
from pathlib import Path

import torch
from scipy.io import wavfile
from transformers import AutoFeatureExtractor, HubertModel

from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio
from vc_infer_pipeline import VC

BASE_DIR = Path(__file__).resolve().parent.parent


_hubert_cache = {}


def _resolve_model_path(model_path: str):
    if Path(model_path).is_file():
        print(
            f"HuBERT path '{model_path}' is a checkpoint file; defaulting to 'facebook/hubert-base-ls960'."
        )
        return "facebook/hubert-base-ls960"
    return model_path


@lru_cache(maxsize=None)
def _load_feature_extractor(model_path: str):
    model_path = _resolve_model_path(model_path)
    return AutoFeatureExtractor.from_pretrained(model_path)


def _cached_hubert_model(model_path: str, device: str, is_half: bool):
    model_path = _resolve_model_path(model_path)
    cache_key = (model_path, device, is_half)
    if cache_key in _hubert_cache:
        return _hubert_cache[cache_key]

    hubert_model = HubertModel.from_pretrained(model_path)
    hubert_model = hubert_model.to(device)
    hubert_model = hubert_model.half() if is_half else hubert_model.float()
    hubert_model.eval()
    _hubert_cache[cache_key] = hubert_model
    return hubert_model


class TransformersHubertWrapper(torch.nn.Module):
    def __init__(self, model_path: str, device: str, is_half: bool):
        super().__init__()
        self.device = device
        self.is_half = is_half
        self.feature_extractor = _load_feature_extractor(model_path)
        self.model = _cached_hubert_model(model_path, device, is_half)
        self.final_proj = self._init_final_proj()

    def _init_final_proj(self):
        hidden_size = self.model.config.hidden_size
        final_dim = getattr(self.model.config, "final_dim", hidden_size)
        if final_dim == hidden_size:
            return torch.nn.Identity()

        projection = torch.nn.Linear(hidden_size, final_dim, bias=False)
        with torch.no_grad():
            projection.weight.zero_()
            diag = min(hidden_size, final_dim)
            projection.weight[:diag, :diag] = torch.eye(diag)
        projection = projection.to(self.device)
        if self.is_half:
            projection = projection.half()
        return projection

    def extract_features(self, source, padding_mask=None, output_layer=None):
        attention_mask = None
        if padding_mask is not None:
            attention_mask = (~padding_mask).long().to(self.device)

        # Transformers processors normalize inputs to match pretraining expectations.
        processed = self.feature_extractor(
            source.squeeze(0).cpu().float().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        input_values = processed.input_values.to(self.device)
        if attention_mask is None and "attention_mask" in processed:
            attention_mask = processed.attention_mask.to(self.device)
        if self.is_half:
            input_values = input_values.half()
        else:
            input_values = input_values.float()

        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states or ()

        if output_layer is not None and output_layer < len(hidden_states):
            features = hidden_states[output_layer]
        else:
            features = outputs.last_hidden_state

        return (features,)


class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "1060" in self.gpu_name
                    or "1070" in self.gpu_name
                    or "1080" in self.gpu_name
            ):
                print("16 series/10 series P40 forced single precision")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(BASE_DIR / "src" / "configs" / config_file, "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(BASE_DIR / "src" / "configs" / config_file, "w") as f:
                        f.write(strr)
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif self.device == "mps" and torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            if self.device != "cpu":
                print("No supported N-card found, use CPU for inference")
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


def load_hubert(device, is_half, model_path):
    return TransformersHubertWrapper(model_path, device, is_half)


def get_vc(device, is_half, config, model_path):
    cpt = torch.load(model_path, map_location='cpu')
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(f'Incorrect format for {model_path}. Use a voice model trained using RVC v2 instead.')

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)

    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc


def rvc_infer(index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, fcpe_hop_length, fcpe_threshold):
    audio = load_audio(input_path, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)
    audio_opt = vc.pipeline(hubert_model, net_g, 0, audio, input_path, times, pitch_change, f0_method, index_path, index_rate, if_f0, filter_radius, tgt_sr, 0, rms_mix_rate, version, protect, crepe_hop_length, fcpe_hop_length, fcpe_threshold)
    wavfile.write(output_path, tgt_sr, audio_opt)
