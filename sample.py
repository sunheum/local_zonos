import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

torch.cuda.empty_cache()
torch.cuda.synchronize()

CONFIG_PATH = "Zonos-v0.1-transformer/config.json"
MODEL_PATH = "Zonos-v0.1-transformer/model.safetensors"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Zonos.from_local(CONFIG_PATH, MODEL_PATH, device=device)

wav, sampling_rate = torchaudio.load("assets/sample2.wav")
speaker = model.make_speaker_embedding(wav, sampling_rate)

torch.manual_seed(421)

context="""
안녕하십니까?
행복은행 텔레뱅킹 서비스입니다.
잔액조회는 111번, 송금은 211번, 분실신고 등 기타 서비스를 원하시면 별표 버튼을 눌러주세요.
"""

cond_dict = make_cond_dict(text=context, speaker=speaker, language="ko")
conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

codes = model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=1,
        sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence, quad=quadratic),
        callback=update_progress,
    )

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample1.wav", wavs[0], model.autoencoder.sampling_rate)
