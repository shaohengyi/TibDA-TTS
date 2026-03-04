"""
基于集成学习的合成语音质量评估模块
用于藏语语音数据增强项目的质量评估
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 导入必要的库
try:
    import pesq
    PESQ_AVAILABLE = True
except ImportError:
    print("警告: PESQ库未安装，将使用替代方案")
    PESQ_AVAILABLE = False
    pesq = None

try:
    import jiwer
    WER_AVAILABLE = True
except ImportError:
    print("警告: jiwer库未安装，将使用简单编辑距离计算")
    WER_AVAILABLE = False
    jiwer = None

try:
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    import torch
    ASR_AVAILABLE = True
except ImportError:
    print("警告: transformers库未安装，无法使用ASR模型")
    ASR_AVAILABLE = False


@dataclass
class AudioSample:
    """音频样本数据类"""
    audio_path: str
    audio_data: Optional[np.ndarray] = None
    sr: int = 16000
    reference_text: Optional[str] = None
    synthesized_text: Optional[str] = None
    features: Optional[Dict] = None


class SpeechQualityEvaluator:
    """
    基于集成学习的语音质量评估器
    结合多种指标进行综合评价
    """
    
    def __init__(self, 
                 sr: int = 16000,
                 pesq_mode: str = 'wb',  # 'wb' for wideband, 'nb' for narrowband
                 weights: Optional[Dict[str, float]] = None,
                 dynamic_threshold: float = 0.7,
                 threshold_adjustment_rate: float = 0.1,
                 asr_model_name: str = "facebook/wav2vec2-base-960h"):
        """
        初始化评估器
        
        Args:
            sr: 采样率
            pesq_mode: PESQ模式 ('wb' 宽带或 'nb' 窄带)
            weights: 各指标权重字典
            dynamic_threshold: 初始动态阈值
            threshold_adjustment_rate: 阈值调整速率
            asr_model_name: ASR模型名称
        """
        self.sr = sr
        self.pesq_mode = pesq_mode
        self.dynamic_threshold = dynamic_threshold
        self.threshold_adjustment_rate = threshold_adjustment_rate
        
        # 默认权重配置
        self.weights = weights or {
            'pesq': 0.35,
            'cosine_similarity': 0.25,
            'mse': 0.20,
            'wer': 0.20
        }
        
        # 验证权重和为1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"权重总和必须为1，当前为{total_weight}")
        
        # 初始化ASR模型（如果需要）
        self.asr_model = None
        self.asr_processor = None
        if ASR_AVAILABLE and asr_model_name:
            self._init_asr_model(asr_model_name)
        
        # 历史分数记录
        self.history_scores = []
        self.threshold_history = [dynamic_threshold]
    
    def _init_asr_model(self, model_name: str):
        """初始化ASR模型用于WER计算"""
        try:
            self.asr_processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.asr_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            print(f"ASR模型 {model_name} 加载成功")
        except Exception as e:
            print(f"ASR模型加载失败: {e}")
            self.asr_model = None
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频数据数组
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr)
            return audio
        except Exception as e:
            print(f"加载音频失败 {audio_path}: {e}")
            raise
    
    def compute_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """
        计算MFCC特征
        
        Args:
            audio: 音频数据
            
        Returns:
            MFCC特征矩阵
        """
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        return mfcc
    
    def compute_pesq_score(self, ref_audio: np.ndarray, deg_audio: np.ndarray) -> float:
        """
        计算PESQ分数
        
        Args:
            ref_audio: 参考音频
            deg_audio: 待评估音频
            
        Returns:
            PESQ分数
        """
        if not PESQ_AVAILABLE:
            # 如果没有PESQ库，使用替代的感知质量评估
            return self._alternative_quality_score(ref_audio, deg_audio)
        
        try:
            # 确保音频长度一致
            min_len = min(len(ref_audio), len(deg_audio))
            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]
            
            # 计算PESQ
            pesq_score = pesq.pesq(self.sr, ref_audio, deg_audio, self.pesq_mode)
            return pesq_score
        except Exception as e:
            print(f"PESQ计算失败: {e}")
            return 1.0  # 返回最低分数
    
    def _alternative_quality_score(self, ref_audio: np.ndarray, deg_audio: np.ndarray) -> float:
        """
        PESQ替代方案：基于频谱特征的感知质量评估
        
        Args:
            ref_audio: 参考音频
            deg_audio: 待评估音频
            
        Returns:
            替代质量分数 (0-4.5)
        """
        # 确保音频长度一致
        min_len = min(len(ref_audio), len(deg_audio))
        ref_audio = ref_audio[:min_len]
        deg_audio = deg_audio[:min_len]
        
        # 计算STFT
        ref_stft = librosa.stft(ref_audio)
        deg_stft = librosa.stft(deg_audio)
        
        # 计算幅度谱
        ref_mag = np.abs(ref_stft)
        deg_mag = np.abs(deg_stft)
        
        # 计算频谱差异
        spectral_diff = np.mean(np.abs(ref_mag - deg_mag))
        
        # 归一化到0-4.5范围
        max_diff = np.mean(ref_mag)  # 使用参考音频平均幅度作为最大差异估计
        
        if max_diff > 0:
            quality_score = 4.5 * (1 - min(1.0, spectral_diff / max_diff))
        else:
            quality_score = 2.0  # 默认中等分数
        
        return quality_score
    
    def compute_cosine_similarity(self, ref_audio: np.ndarray, deg_audio: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            ref_audio: 参考音频
            deg_audio: 待评估音频
            
        Returns:
            余弦相似度 (0-1)
        """
        # 提取MFCC特征
        ref_mfcc = self.compute_mfcc_features(ref_audio)
        deg_mfcc = self.compute_mfcc_features(deg_audio)
        
        # 确保特征维度一致
        min_frames = min(ref_mfcc.shape[1], deg_mfcc.shape[1])
        ref_mfcc = ref_mfcc[:, :min_frames]
        deg_mfcc = deg_mfcc[:, :min_frames]
        
        # 展平特征向量
        ref_vector = ref_mfcc.flatten()
        deg_vector = deg_mfcc.flatten()
        
        # 计算余弦相似度
        dot_product = np.dot(ref_vector, deg_vector)
        norm_ref = np.linalg.norm(ref_vector)
        norm_deg = np.linalg.norm(deg_vector)
        
        if norm_ref > 0 and norm_deg > 0:
            cosine_sim = dot_product / (norm_ref * norm_deg)
            # 确保在[-1, 1]范围内
            cosine_sim = max(-1.0, min(1.0, cosine_sim))
            # 转换到[0, 1]范围
            return (cosine_sim + 1) / 2
        else:
            return 0.0
    
    def compute_mse(self, ref_audio: np.ndarray, deg_audio: np.ndarray) -> float:
        """
        计算均方误差
        
        Args:
            ref_audio: 参考音频
            deg_audio: 待评估音频
            
        Returns:
            归一化的MSE分数 (0-1, 0表示最好)
        """
        # 确保音频长度一致
        min_len = min(len(ref_audio), len(deg_audio))
        ref_audio = ref_audio[:min_len]
        deg_audio = deg_audio[:min_len]
        
        # 计算MSE
        mse = np.mean((ref_audio - deg_audio) ** 2)
        
        # 归一化处理
        # 使用参考音频的方差作为归一化因子
        ref_variance = np.var(ref_audio)
        
        if ref_variance > 0:
            normalized_mse = mse / ref_variance
            # 转换为质量分数 (1为最好，0为最差)
            mse_score = 1.0 / (1.0 + normalized_mse)
        else:
            mse_score = 0.5  # 默认分数
        
        return mse_score
    
    def transcribe_audio(self, audio: np.ndarray) -> str:
        """
        使用ASR模型转录音频
        
        Args:
            audio: 音频数据
            
        Returns:
            转录文本
        """
        if not ASR_AVAILABLE or self.asr_model is None:
            return ""
        
        try:
            # 预处理音频
            input_values = self.asr_processor(
                audio, 
                sampling_rate=self.sr, 
                return_tensors="pt"
            ).input_values
            
            # 进行推理
            with torch.no_grad():
                logits = self.asr_model(input_values).logits
            
            # 解码
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.asr_processor.batch_decode(predicted_ids)[0]
            
            return transcription
        except Exception as e:
            print(f"音频转录失败: {e}")
            return ""
    
    def compute_wer(self, reference_text: str, hypothesis_text: str) -> float:
        """
        计算词错误率
        
        Args:
            reference_text: 参考文本
            hypothesis_text: 假设文本
            
        Returns:
            归一化的WER分数 (0-1, 1表示最好)
        """
        if not reference_text or not hypothesis_text:
            return 0.5  # 无法计算时返回中等分数
        
        if WER_AVAILABLE:
            try:
                # 使用jiwer计算WER
                wer_score = jiwer.wer(reference_text, hypothesis_text)
                # 转换为质量分数 (1 - WER)
                return 1.0 - min(1.0, wer_score)
            except:
                # 回退到简单编辑距离
                return self._simple_wer(reference_text, hypothesis_text)
        else:
            # 使用简单编辑距离
            return self._simple_wer(reference_text, hypothesis_text)
    
    def _simple_wer(self, ref: str, hyp: str) -> float:
        """
        基于编辑距离的简单WER计算
        
        Args:
            ref: 参考文本
            hyp: 假设文本
            
        Returns:
            归一化的WER分数
        """
        # 分割为单词
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        m, n = len(ref_words), len(hyp_words)
        
        # 初始化编辑距离矩阵
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 计算编辑距离
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],    # 删除
                                      dp[i][j-1],    # 插入
                                      dp[i-1][j-1])  # 替换
        
        # 计算WER
        wer = dp[m][n] / max(1, m)
        
        # 转换为质量分数
        return 1.0 - min(1.0, wer)
    
    def compute_all_metrics(self, reference_sample: AudioSample, 
                           synthesized_sample: AudioSample) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            reference_sample: 参考音频样本
            synthesized_sample: 合成音频样本
            
        Returns:
            包含所有指标分数的字典
        """
        # 加载音频数据
        if reference_sample.audio_data is None:
            ref_audio = self.load_audio(reference_sample.audio_path)
        else:
            ref_audio = reference_sample.audio_data
        
        if synthesized_sample.audio_data is None:
            syn_audio = self.load_audio(synthesized_sample.audio_path)
        else:
            syn_audio = synthesized_sample.audio_data
        
        # 确保采样率一致
        if len(ref_audio) != len(syn_audio):
            min_len = min(len(ref_audio), len(syn_audio))
            ref_audio = ref_audio[:min_len]
            syn_audio = syn_audio[:min_len]
        
        # 计算PESQ分数
        pesq_score = self.compute_pesq_score(ref_audio, syn_audio)
        
        # 计算余弦相似度
        cosine_score = self.compute_cosine_similarity(ref_audio, syn_audio)
        
        # 计算MSE分数
        mse_score = self.compute_mse(ref_audio, syn_audio)
        
        # 计算WER分数
        if reference_sample.reference_text and synthesized_sample.synthesized_text:
            # 如果提供了文本，直接使用
            wer_score = self.compute_wer(
                reference_sample.reference_text, 
                synthesized_sample.synthesized_text
            )
        else:
            # 否则，使用ASR模型转录
            if self.asr_model is not None:
                ref_text = self.transcribe_audio(ref_audio)
                syn_text = self.transcribe_audio(syn_audio)
                wer_score = self.compute_wer(ref_text, syn_text)
            else:
                wer_score = 0.5  # 无法计算时使用默认值
        
        return {
            'pesq': pesq_score,
            'cosine_similarity': cosine_score,
            'mse': mse_score,
            'wer': wer_score
        }
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        归一化分数到[0, 1]范围
        
        Args:
            scores: 原始分数字典
            
        Returns:
            归一化后的分数字典
        """
        # PESQ: 原始范围约[1.0, 4.5]，归一化到[0, 1]
        pesq_norm = (scores['pesq'] - 1.0) / 3.5 if scores['pesq'] > 1.0 else 0.0
        pesq_norm = max(0.0, min(1.0, pesq_norm))
        
        # 余弦相似度: 已在[0, 1]范围内
        cosine_norm = max(0.0, min(1.0, scores['cosine_similarity']))
        
        # MSE: 已在[0, 1]范围内
        mse_norm = max(0.0, min(1.0, scores['mse']))
        
        # WER: 已在[0, 1]范围内
        wer_norm = max(0.0, min(1.0, scores['wer']))
        
        return {
            'pesq': pesq_norm,
            'cosine_similarity': cosine_norm,
            'mse': mse_norm,
            'wer': wer_norm
        }
    
    def compute_composite_score(self, normalized_scores: Dict[str, float]) -> float:
        """
        计算综合评分
        
        Args:
            normalized_scores: 归一化后的分数字典
            
        Returns:
            综合评分
        """
        composite_score = 0.0
        for metric, score in normalized_scores.items():
            composite_score += score * self.weights.get(metric, 0.0)
        
        return composite_score
    
    def update_dynamic_threshold(self, new_score: float) -> None:
        """
        更新动态阈值
        
        Args:
            new_score: 新的综合评分
        """
        # 添加新分数到历史记录
        self.history_scores.append(new_score)
        
        # 计算历史分数的移动平均
        if len(self.history_scores) > 1:
            # 使用指数移动平均更新阈值
            alpha = self.threshold_adjustment_rate
            self.dynamic_threshold = (alpha * new_score + 
                                     (1 - alpha) * self.dynamic_threshold)
        else:
            # 第一个分数，直接使用
            self.dynamic_threshold = new_score
        
        # 记录阈值历史
        self.threshold_history.append(self.dynamic_threshold)
    
    def evaluate_sample(self, reference_sample: AudioSample, 
                       synthesized_sample: AudioSample) -> Dict[str, Union[float, bool]]:
        """
        评估单个样本
        
        Args:
            reference_sample: 参考音频样本
            synthesized_sample: 合成音频样本
            
        Returns:
            评估结果字典
        """
        # 计算所有指标
        raw_scores = self.compute_all_metrics(reference_sample, synthesized_sample)
        
        # 归一化分数
        normalized_scores = self.normalize_scores(raw_scores)
        
        # 计算综合评分
        composite_score = self.compute_composite_score(normalized_scores)
        
        # 判断是否通过质量检查
        passes_quality_check = composite_score >= self.dynamic_threshold
        
        # 更新动态阈值
        self.update_dynamic_threshold(composite_score)
        
        return {
            'raw_scores': raw_scores,
            'normalized_scores': normalized_scores,
            'composite_score': composite_score,
            'passes_quality_check': passes_quality_check,
            'current_threshold': self.dynamic_threshold
        }
    
    def evaluate_batch(self, reference_samples: List[AudioSample], 
                      synthesized_samples: List[AudioSample]) -> Dict[str, any]:
        """
        批量评估样本
        
        Args:
            reference_samples: 参考音频样本列表
            synthesized_samples: 合成音频样本列表
            
        Returns:
            批量评估结果
        """
        if len(reference_samples) != len(synthesized_samples):
            raise ValueError("参考样本和合成样本数量必须相同")
        
        results = []
        passed_count = 0
        composite_scores = []
        
        for ref, syn in zip(reference_samples, synthesized_samples):
            result = self.evaluate_sample(ref, syn)
            results.append(result)
            
            if result['passes_quality_check']:
                passed_count += 1
            
            composite_scores.append(result['composite_score'])
        
        # 计算统计信息
        total_count = len(reference_samples)
        pass_rate = passed_count / total_count if total_count > 0 else 0.0
        avg_composite_score = np.mean(composite_scores) if composite_scores else 0.0
        std_composite_score = np.std(composite_scores) if len(composite_scores) > 1 else 0.0
        
        return {
            'individual_results': results,
            'statistics': {
                'total_samples': total_count,
                'passed_samples': passed_count,
                'pass_rate': pass_rate,
                'avg_composite_score': avg_composite_score,
                'std_composite_score': std_composite_score,
                'current_threshold': self.dynamic_threshold
            }
        }
    
    def save_evaluation_report(self, results: Dict[str, any], filepath: str) -> None:
        """
        保存评估报告
        
        Args:
            results: 评估结果
            filepath: 文件保存路径
        """
        import json
        import datetime
        
        # 添加元数据
        report = {
            'evaluation_date': datetime.datetime.now().isoformat(),
            'evaluation_config': {
                'sr': self.sr,
                'pesq_mode': self.pesq_mode,
                'weights': self.weights,
                'initial_threshold': self.threshold_history[0],
                'final_threshold': self.dynamic_threshold
            },
            'results': results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"评估报告已保存到: {filepath}")


class DynamicThresholdOptimizer:
    """
    动态阈值优化器
    根据样本分布自动调整阈值
    """
    
    def __init__(self, 
                 initial_threshold: float = 0.7,
                 min_threshold: float = 0.3,
                 max_threshold: float = 0.9,
                 adaptation_rate: float = 0.1,
                 target_pass_rate: float = 0.7,
                 window_size: int = 100):
        """
        初始化阈值优化器
        
        Args:
            initial_threshold: 初始阈值
            min_threshold: 最小阈值
            max_threshold: 最大阈值
            adaptation_rate: 适应速率
            target_pass_rate: 目标通过率
            window_size: 滑动窗口大小
        """
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_rate = adaptation_rate
        self.target_pass_rate = target_pass_rate
        self.window_size = window_size
        
        self.pass_rates = []
        self.score_history = []
        self.threshold_history = [initial_threshold]
    
    def update(self, scores: List[float], passed_flags: List[bool]) -> float:
        """
        更新阈值
        
        Args:
            scores: 分数列表
            passed_flags: 通过标志列表
            
        Returns:
            更新后的阈值
        """
        if not scores:
            return self.threshold
        
        # 记录历史数据
        self.score_history.extend(scores)
        self.score_history = self.score_history[-self.window_size:]
        
        # 计算当前通过率
        if passed_flags:
            current_pass_rate = sum(passed_flags) / len(passed_flags)
            self.pass_rates.append(current_pass_rate)
            self.pass_rates = self.pass_rates[-self.window_size:]
            
            # 计算平均通过率
            avg_pass_rate = np.mean(self.pass_rates) if self.pass_rates else self.target_pass_rate
            
            # 调整阈值
            threshold_adjustment = self.adaptation_rate * (avg_pass_rate - self.target_pass_rate)
            new_threshold = self.threshold - threshold_adjustment
            
            # 确保阈值在合理范围内
            new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
            
            self.threshold = new_threshold
            self.threshold_history.append(new_threshold)
        
        return self.threshold
    
    def get_optimal_threshold(self) -> float:
        """获取当前最优阈值"""
        return self.threshold


# 使用示例
def main():
    """主函数：演示语音质量评估模块的使用"""
    
    # 示例音频文件路径
    REFERENCE_AUDIO_PATH = "path/to/reference.wav"
    SYNTHESIZED_AUDIO_PATH = "path/to/synthesized.wav"
    
    # 初始化评估器
    evaluator = SpeechQualityEvaluator(
        sr=16000,
        pesq_mode='wb',
        weights={
            'pesq': 0.35,
            'cosine_similarity': 0.25,
            'mse': 0.20,
            'wer': 0.20
        },
        dynamic_threshold=0.7
    )
    
    # 创建音频样本
    reference_sample = AudioSample(
        audio_path=REFERENCE_AUDIO_PATH,
        reference_text="དཔེ་གཞི་བོད་སྐད་དུ་བརྗོད་པའི་ཚིག་གི་དཔེ་བྱུང་།"  # 示例藏语文
    )
    
    synthesized_sample = AudioSample(
        audio_path=SYNTHESIZED_AUDIO_PATH,
        synthesized_text="དཔེ་གཞི་བོད་སྐད་དུ་བརྗོད་པའི་ཚིག་གི་དཔེ་བྱུང་།"  # 示例藏语文
    )
    
    # 评估单个样本
    result = evaluator.evaluate_sample(reference_sample, synthesized_sample)
    
    print("=" * 60)
    print("语音质量评估结果")
    print("=" * 60)
    
    print(f"\n原始分数:")
    for metric, score in result['raw_scores'].items():
        print(f"  {metric}: {score:.4f}")
    
    print(f"\n归一化分数:")
    for metric, score in result['normalized_scores'].items():
        print(f"  {metric}: {score:.4f}")
    
    print(f"\n综合评分: {result['composite_score']:.4f}")
    print(f"当前阈值: {result['current_threshold']:.4f}")
    print(f"质量检查: {'通过' if result['passes_quality_check'] else '不通过'}")
    
    # 批量评估示例
    print("\n" + "=" * 60)
    print("批量评估示例")
    print("=" * 60)
    
    # 创建批量样本
    batch_size = 5
    reference_batch = []
    synthesized_batch = []
    
    for i in range(batch_size):
        reference_batch.append(AudioSample(
            audio_path=f"path/to/reference_{i}.wav",
            reference_text="དཔེ་གཞི་བོད་སྐད་དུ་བརྗོད་པའི་ཚིག་གི་དཔེ་བྱུང་།"
        ))
        synthesized_batch.append(AudioSample(
            audio_path=f"path/to/synthesized_{i}.wav",
            synthesized_text="དཔེ་གཞི་བོད་སྐད་དུ་བརྗོད་པའི་ཚིག་གི་དཔེ་བྱུང་།"
        ))
    
    # 注意：这里只是示例，实际使用时需要替换为真实的音频文件路径
    print(f"批量评估 {batch_size} 个样本（示例）")
    
    # 保存评估报告
    # evaluator.save_evaluation_report(batch_results, "quality_evaluation_report.json")


if __name__ == "__main__":
    # 检查依赖
    print("检查依赖库...")
    if not PESQ_AVAILABLE:
        print("警告: 未找到PESQ库，将使用替代的质量评估方法")
        print("安装PESQ: pip install pesq")
    
    if not WER_AVAILABLE:
        print("警告: 未找到jiwer库，将使用简单的WER计算方法")
        print("安装jiwer: pip install jiwer")
    
    if not ASR_AVAILABLE:
        print("警告: 未找到transformers库，无法使用ASR功能")
        print("安装transformers: pip install transformers torch")
    
    print("\n依赖检查完成，可以运行示例代码")
    
    # 运行示例
    # main()