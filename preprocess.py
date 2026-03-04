import argparse
import text
from utils import load_filepaths_and_text
#处理filelists里的txt文件，使用cleaner，生成对应的.cleaned后缀文件
if __name__ == '__main__':
  # 定义并解析命令行参数
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")#filelists文件夹里的文件，定义输出文件的扩展名.cleaned
  parser.add_argument("--text_index", default=1, type=int)#指定文本在文件列表中的索引
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])#要处理的文件列表
  parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])#指定用于清理文本的清理器列表

  args = parser.parse_args()
    

  for filelist in args.filelists:
    print("START:", filelist)
    # 分出path和text列表
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in range(len(filepaths_and_text)):
      original_text = filepaths_and_text[i][args.text_index]
      #对文本使用指定的cleaner进行清理
      cleaned_text = text._clean_text(original_text, args.text_cleaners)
      filepaths_and_text[i][args.text_index] = cleaned_text

    new_filelist = filelist + "." + args.out_extension#新的文件名，加了.cleaned后缀
    with open(new_filelist, "w", encoding="utf-8") as f:
      f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
