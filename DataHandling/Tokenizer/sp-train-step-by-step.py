import sentencepiece as spm
import glob

# spm.SentencePieceTrainer.Train(
#     "--input=../../data/botchan.txt --model_prefix=../../sentence-piece/m --vocab_size=78 --user_defined_symbols=<bos>,<eos>,<pad>"
# )

text_files = glob.glob("../../data/raw/*.txt")

input_files = ",".join(text_files)

spm.SentencePieceTrainer.Train(
    f"--input={input_files} --model_prefix=../../sentence-piece/m --vocab_size=10000 --user_defined_symbols=<bos>,<eos>,<pad>"
)
