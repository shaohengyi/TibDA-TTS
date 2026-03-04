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

from scipy.io.wavfile import write


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
_ = utils.load_checkpoint("/root/autodl-tmp/Tibetan_TTS/logs/tibetan_base/G_384000.pth", net_g, None)


#*************************************自己加入的****************************************#
with open('/root/autodl-tmp/Tibetan_TTS/filelists/test_uni.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
# 循环处理每一行
for line in lines:
    # 分割每一行，假设文件路径和藏文之间用 "|" 分隔
    parts = line.strip().split('|')
    if len(parts) == 2:
        audio_file = parts[0]  # 提取音频文件路径
        tibetan_text = parts[1]  # 提取藏文部分
        # 打印出来或进行其他处理
        print(f"音频文件: {audio_file}")
        print(f"藏文内容: {tibetan_text}")
    else:
        print("格式错误，无法解析:", line)

    tibetan_text = str(tibetan_text)
    full_path = "/root/autodl-tmp/Tibetan_TTS/synthetic_speech/" + audio_file.split('/')[-1]
    print(tibetan_text)
    print(full_path)

    # stn_tst = get_text("དགའ་སྣང་དང་ཡང་སེམས་འཚབ་ཀྱང་ཡོང་གི་ཡོད་ཅེས་བརྗོད་འདུག", hps)
    stn_tst = get_text(tibetan_text, hps)
    # print(stn_tst)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
    audio = audio * 32767 / max(0.01, np.max(np.abs(audio))) * 0.6
    audio = audio.astype(np.int16)
    write(full_path, hps.data.sampling_rate, audio)

    # audio, sr = librosa.load('result.wav', sr=None, mono=True)

# # plt画图
# f, ((ax11, ax12)) = plt.subplots(1, 2, sharex=False, sharey=False)
# # 01 左，信号
# ax11.set_title('Signal')
# ax11.set_xlabel('Time (samples)')
# ax11.set_ylabel('Amplitude')
# ax11.plot(audio)
# # 02 右，傅里叶变换
# n_fft = 2048
# ft = np.abs(librosa.stft(audio[:n_fft], hop_length=n_fft+1))
# ax12.set_title('Spectrum')
# ax12.set_xlabel('Frequency Bin')
# # ax12.set_ylabel('Amplitude')
# ax12.plot(ft)

# plt.savefig("vis.jpg")