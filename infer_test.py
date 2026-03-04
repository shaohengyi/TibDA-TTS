import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import librosa

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols_tibetan import symbols
from text import text_to_sequence, cleaned_text_to_sequence
from utils import load_wav_to_torch, load_filepaths_and_text

from scipy.io.wavfile import write
import os

from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2

config = visqol_config_pb2.VisqolConfig()

mode = "speech"
if mode == "audio":
    config.audio.sample_rate = 48000
    config.options.use_speech_scoring = False
    svr_model_path = "libsvm_nu_svr_model.txt"
elif mode == "speech":
    config.audio.sample_rate = 16000
    config.options.use_speech_scoring = True
    svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
else:
    raise ValueError(f"Unrecognized mode: {mode}")

config.options.svr_model_path = os.path.join(
    os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

api = visqol_lib_py.VisqolApi()

api.Create(config)

def get_text(text, hps):
    text_norm = cleaned_text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/tibetan_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()
_ = utils.load_checkpoint("logs/tibetan_base/G_118000.pth", net_g, None)

label_txt_path = hps.data.test_files
audiopaths_and_text = load_filepaths_and_text(label_txt_path)
for audio_path, text in audiopaths_and_text:
    refer_audio, sampling_rate = librosa.core.load(audio_path, sr=hps.data.sampling_rate)
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
    audio = audio * 32767 / max(0.01, np.max(np.abs(audio))) * 0.6
    audio = audio.astype(np.int16)
    refer_audio = refer_audio * 32767 / max(0.01, np.max(np.abs(refer_audio))) * 0.6
    refer_audio = refer_audio.astype(np.int16)
    print(os.path.basename(audio_path))
    write('results/'+os.path.basename(audio_path)[:-4] + "_ref.wav", hps.data.sampling_rate, refer_audio)
    write('results/'+os.path.basename(audio_path), hps.data.sampling_rate, audio)
    similarity_result = api.Measure('results/'+os.path.basename(audio_path)[:-4] + "_ref.wav", 'results/'+os.path.basename(audio_path))
    
