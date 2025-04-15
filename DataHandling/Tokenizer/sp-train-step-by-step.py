import sentencepiece as spm
import glob
import os
from dotenv import load_dotenv

load_dotenv("../../.env")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", 10000))

text_files = glob.glob("../../data/raw/*.txt")

input_files = ",".join(text_files)

spm.SentencePieceTrainer.Train(
    f"--input={input_files} --model_prefix=../../sentence-piece/small --vocab_size={VOCAB_SIZE} --user_defined_symbols=<bos>,<eos>,<pad>"
)
