import sentencepiece as spm

# Train the SentencePiece model with properly formatted arguments
spm.SentencePieceTrainer.Train(
    "--input=data/botchan.txt --model_prefix=sentence-piece/m --vocab_size=2000 "
    "--control_symbols=<bos>,<eos>,<pad>"
)