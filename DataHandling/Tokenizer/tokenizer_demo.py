import sentencepiece as spm

sp = spm.SentencePieceProcessor()
proto = open("./sentence-piece/big.model", 'rb').read()
sp.LoadFromSerializedProto(proto)

for i in range(sp.GetPieceSize()):
    print(f"ID {i}: {sp.IdToPiece(i)}")