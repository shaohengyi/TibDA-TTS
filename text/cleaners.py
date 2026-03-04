""" from https://github.com/keithito/tacotron """

'''
清理器是对输入文本在训练和评估时进行转换的工具。

可以通过传递以逗号分隔的清理器名称列表作为 "cleaners" 超参数来选择清理器。有些清理器是专门针对英语的。通常你会使用以下几种清理器：

对于英语文本，使用 "english_cleaners"。
对于可以使用 Unidecode 库（https://pypi.python.org/pypi/Unidecode）转写为 ASCII 的非英语文本，使用 "transliteration_cleaners"。
如果你不想进行转写，使用 "basic_cleaners"（在这种情况下，你还应该更新 symbols.py 中的符号以匹配你的数据）。

清理器是什么？清理器是一种工具，它会在训练和评估模型时对输入文本进行转换。这些转换可以包括去除特殊字符、标准化文本格式等。
'''

import re
from unidecode import unidecode
from phonemizer import phonemize


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

#检索文本里的_abbreviations前面的简写，全部替换成后面的全称
def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text

#对文本做normalize处理，这里的normalize_numbers好像没有定义
def expand_numbers(text):
  return normalize_numbers(text)

#text文本全部小写
def lowercase(text):
  return text.lower()

#把_whitespace_re检索到的连续的空格符号，制表符号等全部替换成单个空格符号
def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)

#使用unidecode函数将输入文本转换成ASCII表示
def convert_to_ascii(text):
  return unidecode(text)

#basic_cleaners：小写，替换连续空格符号或者制表符号
def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

#transliteration_cleaners：先转换成ASCII表示，再执行小写+替换连续空格符号和制表符号
def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

#先转换成ASCII表示，再小写，再替换缩写，再转成音素，再去掉多余的空格符号，制表符号
def english_cleaners(text):
  '''Pipeline for English text, including abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes

# 比上面多了保留转换出来的音素的重音和标点符号
def english_cleaners2(text):
  '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes
