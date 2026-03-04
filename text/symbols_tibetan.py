""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
#标点符号
_punctuation = [
    " ", ".", ",", ":", ";", "!", "?", "\"", "'", "(", ")", "[", "]", "{", "}", "-", "_", "+", "=", "*", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "`", "~", "<", ">", 
    "·", "，", "。", "、", "；", "：", "？", "！", "“", "”", "‘", "’", "（", "）", "【", "】", "《", "》", "「", "」", "『", "』", "﹃", "﹄", "〔", "〕", "——", "……", "—", "－", "【", "】"
]
#字符
_letters = 'ༀ	༁	༂	༃	༄	༅	༆	༇	༈	༉	༊	་	༌	།	༎	༏	༐	༑	༒	༓	༔	༕	༖	༗	༘	༙	༚	༛	༜	༝	༞	༟	༠	༡	༢	༣	༤	༥	༦	༧	༨	༩	༪	༫	༬	༭	༮	༯	༰	༱	༲	༳	༴	༵	༶	༷	༸	༹	༺	༻	༼	༽	༾	༿	ཀ	ཁ	ག	གྷ	ང	ཅ	ཆ	ཇ	ཉ	ཊ	ཋ	ཌ	ཌྷ	ཎ	ཏ	ཐ	ད	དྷ	ན	པ	ཕ	བ	བྷ	མ	ཙ	ཚ	ཛ	ཛྷ	ཝ	ཞ	ཟ	འ	ཡ	ར	ལ	ཤ	ཥ	ས	ཧ	ཨ	ཀྵ	ཪ	ཫ	ཬ	ཱ	ི	ཱི	ུ	ཱུ	ྲྀ	ཷ	ླྀ	ཹ	ེ	ཻ	ོ	ཽ	ཾ	ཿ	ྀ	ཱྀ	ྂ	ྃ	྄	྅	྆	྇	ྈ	ྉ	ྊ	ྋ	ྌ	ྍ	ྎ	ྏ	ྐ	ྑ	ྒ	ྒྷ	ྔ	ྕ	ྖ	ྗ	ྙ	ྚ	ྛ	ྜ	ྜྷ	ྞ	ྟ	ྠ	ྡ	ྡྷ	ྣ	ྤ	ྥ	ྦ	ྦྷ	ྨ	ྩ	ྪ	ྫ	ྫྷ	ྭ	ྮ	ྯ	ྰ	ྱ	ྲ	ླ	ྴ	ྵ	ྶ	ྷ	ྸ	ྐྵ	ྺ	ྻ	ྼ	྾	྿	࿀	࿁	࿂	࿃	࿄	࿅	࿆	࿇	࿈	࿉	࿊	࿋	࿌	࿎	࿏	࿐	࿑	࿒	࿓	࿔	࿙	࿚	'
#数字
_number = "0123456789"
#英文字母
_english = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters = _letters.replace("	", "")

# Export all symbols:
symbols = [_pad] + list(_letters) + list(_number) + list(_english)

