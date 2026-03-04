import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch
import librosa

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

# 该函数用来加载模型和优化器的检查点（checkpoint）
def load_checkpoint(checkpoint_path, model, optimizer=None):
  # 确定checkpoint_path指向的文件存在
  assert os.path.isfile(checkpoint_path)
  # 使用 torch.load 加载检查点文件，并将其映射到 CPU
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  # 提取检查点信息：iteration、learning_rate、optimizer、model参数
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  # hasattr应该是内置函数，如果model里有module这个状态（大概率是分布式训练），state_dict就是当前的模型状态
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  
  # 把保存的模型状态赋值给new_state_dict，再加载到模型里
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))
  # 返回更新参数后的model，optimizer，learning_rate和iteration
  return model, optimizer, learning_rate, iteration

# 在指定的iteration时保存模型参数，包括model的state_dict，iteration，optimizer.state_dict()，还有learning_rate
def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)

# 将各种类型的数据（标量、直方图、图像和音频）记录到 TensorBoard 中，这个暂时用不到了，不用看
def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)

# 该函数用于查找指定目录中最新的检查点文件
def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x

# 将频谱图（spectrogram）转换为一个NumPy数组：这段代码将频谱图（spectrogram）绘制成图像，并将其转换为NumPy数组
def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path, sampling_rate):
  # 读取对应路径的wav文件，如果原始的采样率等于sampling_rate那最好，如果wav的初始采样率和sampling_rate不一致那就会把wav的采样率转换到sampling_rate,这里的data是numpy数组
  data, sampling_rate = librosa.core.load(full_path, sr=sampling_rate)
  #sampling_rate, data = read(full_path)
  #if len(data.shape) == 2 and data.shape[1] == 2:
  #  data = data.mean(axis=1)

  #将data从numpy数组转换成浮点数张量再返回
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate

#读取指定路径文件，根据|分割path和标签text得到数据集列表filepaths_and_text
def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text

#返回模型训练所需要的超参数：在执行train.py的时候要给出--config（tibetan_base.json）和--model（tibetan_base）的具体值
def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,help='Model name')
  args = parser.parse_args()
  model_dir = os.path.join("./logs", args.model)
  #确保logs/tibetan_base文件存在
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  #把config里要用的配置文件转存到logs/tibetan_base里
  config_path = args.config
  config_save_path = os.path.join(model_dir, "config.json")
  if init:
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)
  #加载超参数并返回
  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams

#check_git_hash(model_dir) 函数的主要作用是确保代码版本的一致性。它通过比较当前代码的 Git 提交哈希值和之前保存的哈希值，检测代码是否有变化。（和我的代码运行关系不大）
def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      logger.warn("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)

#返回一个日志记录器
def get_logger(model_dir, filename="train.log"):
  # logger为全局变量，在函数内部定义的logger在函数外部依然可以访问
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))#创建一个新的日志记录器实例
  logger.setLevel(logging.DEBUG)#将日志记录器的日志级别设置为DEBUG
  #创建一个日志格式化器formatter，指定日志消息的格式：包含时间戳、记录器名称、日志级别和消息内容，以制表符分隔。
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))#创建一个文件处理器h，将日志记录到指定目录下的文件中
  h.setLevel(logging.DEBUG)#将文件处理器的日志级别设置为DEBUG
  h.setFormatter(formatter)#为文件处理器设置之前创建的格式化器
  logger.addHandler(h)#将文件处理器添加到日志记录器logger中
  return logger

#将获得的参数都换成具体的数值以供调用
class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
