""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols_tibetan import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)} # 字符转序号
_id_to_symbol = {i: s for i, s in enumerate(symbols)} # 序号转字符

# 先清理文本再把文本转成序号列表
def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  clean_text = _clean_text(text, cleaner_names)
  for symbol in clean_text:
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence

# # 清理过的cleaned_text转变为编号序列
def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  for symbol in cleaned_text:
    try:
      sequence.append(_symbol_to_id[symbol])
    except:
      continue
  return sequence

# 编号序列转变为文字text
def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result

# 接受需要清理的文本text和清理器名称的列表cleaner_names，返回清理后的文本（这里的text往往经过多个不同的cleaner清理）
def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
