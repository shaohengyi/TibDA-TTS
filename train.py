# -*- coding: utf-8 -*-
import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols_tibetan import symbols


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."#检查是否可以使用GPU进行训练。如果CUDA不可用，程序将抛出AssertionError
  n_gpus = torch.cuda.device_count()#返回当前系统中可用的GPU数量
  # print(n_gpus)#1
  os.environ['MASTER_ADDR'] = 'localhost'#表示主节点的地址是本地主机
  os.environ['MASTER_PORT'] = '8000'#表示主节点使用的端口号是8000
  hps = utils.get_hparams()#调用utils.get_hparams()函数，获取训练所需的超参数
  #使用 torch.distributed.launch的mp.spawn函数来启动多进程训练。run是用于执行的训练逻辑。nprocs=n_gpus是使用的GPU数量。args是传递给run函数的参数，包括GPU数量和超参数hps
  print('准备进入分布式训练：')
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))#mp.spawn会分出来n_gpus个进程，还会自动赋值rank（值为0到n_gpus-1）给run函数

def run(rank, n_gpus, hps):
  print('rank:', rank)#0
  print('ngpus:', n_gpus)#1
  print('hps.model_dir:', hps.model_dir)#./logs/tibetan_base
  global global_step
  #在分布式训练中，rank 为 0 的进程是主进程，负责处理一些全局的初始化和记录工作
  if rank == 0:
    #hps.model_dir是/home/brain/shy/藏语语音合成/logs/tibetan_base，返回一个全局变量日志记录器logger
    logger = utils.get_logger(hps.model_dir)
    #打印超参数
    logger.info(hps)
    # #检查哈希值（不重要）
    # utils.check_git_hash(hps.model_dir)
    #创建Tensorboard记录器，一个存放在hps.model_dir里，一个存放在hps.model_dir/eval
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  #初始化分布式进程组
  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  #设置随机种子,方便复现结果
  torch.manual_seed(hps.train.seed)
  #设置当前进程使用的GPU
  torch.cuda.set_device(rank)
  
  #自己的数据集，5000多个sample组成，每一个sample是由（text, spec, wav）,三个都是单维度tensor序列，text是id序列，spec是shape为[梅尔频谱数量，时间步长]的tensor，wav是单维度tensor序列
  train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
 
  #自己定义的采样器，保证每个批次的样本长度都差不多，在每个边界范围内的样本各自按照batch_size生成batch；使用for batch_indices in train_sampler: print(batch_indices)来查看得到结果，
  #每个batch的长度是64，print的结果是长度是64的索引序号列表，这些索引序号用来在train_dataset里找到对应的样本(text，spec，wav)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
 
  # 自己定义的批处理函数，将样本列表转换成批次数据
  collate_fn = TextAudioCollate()
 
  # 生成train_loader
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)

  # print(len(train_loader))#77个batch
  # for batch in train_loader: #一个batch的内容包括，64个text堆叠，对应长度，64个spec堆叠，对应的wav长度，64个wav堆叠，对应的长度，所以是6
  #   print(batch)#一个batch有64个样本堆叠而成
  #   print(batch[0].shape)#torch.Size([64, 173])
  #   print(batch[1].shape)#torch.Size([64])
  #   print(batch[2].shape)#torch.Size([64, 513, 297])
  #   print(batch[3].shape)#torch.Size([64])
  #   print(batch[4].shape)#torch.Size([64, 1, 76240])
  #   print(batch[5].shape)#torch.Size([64])
  #   break  # 打印第一个批次后退出循环，以避免过多输出

  # 如果是主进程，生成eval_dataloader
  if rank == 0:
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

  #SynthesizerTrn：是要训练的语音合成的模型
  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  #多周期鉴别器：用来判别合成器生成的音频的真实性
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  # 语音合成模型的Adam优化器
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  # 多周期鉴别器的adam优化器
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  # 将语音合成的模型和多周期鉴别器利用DDP分布式部署到多个GPU上进行并行训练，rank表示对应的GPU
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])

  print('预处理数据部分结束')
  # 尝试从最新的检查点文件中加载模型参数和优化器状态，以实现断点的续训和恢复：就是说有可用的检查点，就在他的基础上开始训练
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  # 如果没有可用的检查点就从头开始训练
  except:
    epoch_str = 1
    global_step = 0
  # 定义各自的学习率调度器
  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  # 使用 PyTorch 的 GradScaler 类来支持混合精度训练 (fp16_run)，以提高训练效率和减少内存占用
  scaler = GradScaler(enabled=hps.train.fp16_run)
  print('开始训练')
  # 训练循环
  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      #train_and_evaluate()函数跑的时候有问题
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    # 一个epoch训练结束需要更新学习率
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers
  # 分布式训练中的每个epoch都需要手动设置，所以要自己set_epoch
  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  # 索引，（文本数据，文本数据长度，频谱数据，频谱数据长度，音频数据，音频数据长度）
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
    # 把数据转移到指定的GPU上，rank是指定的GPU的序号
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    # hps.train.fp16_run=true，开启混合精度训练
    with autocast(enabled=hps.train.fp16_run):
      # y_hat是生成的音频信号
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths)
      # 输入频谱图spec，得到处理后的shape为[n_mel_channels，T]的梅尔频谱图数据
      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      # 将mel频谱进行裁剪，这部分是真实音频的频谱片段
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      # 将模型输出的y_hat计算出梅尔频谱图
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )
      # 裁剪真实音频信号片段
      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # Discriminator返回判别器对于真实音频信号 y 的输出 y_d_hat_r，以及对于生成器生成的音频信号 y_hat 的输出 y_d_hat_g
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      # 使用混合精度，计算判别器在真实音频和生成音频上的损失
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    # 梯度清零
    optim_d.zero_grad()
    # 反向传播
    scaler.scale(loss_disc_all).backward()
    # 在应用梯度更新之前，将梯度缩放回全精度
    scaler.unscale_(optim_d)
    # 裁剪梯度以防止梯度爆炸
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    # 更新判别器的参数
    scaler.step(optim_d)

    # 生成器根据损失进行训练，参数修改
    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    optim_g.zero_grad()
    # print(loss_gen_all.is_complex())
    # print(f'loss_gen_all dtype: {loss_gen_all.dtype}')
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()
    #记录和可视化训练过程
    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        break
      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
