import sentencepiece as spm

# Assumes that m.model is stored in non-Posix file system.
serialized_model_proto = open("sentence-piece/m.model", 'rb').read()

sp = spm.SentencePieceProcessor()
sp.LoadFromSerializedProto(serialized_model_proto)

# Get the IDs of <bos> and <eos>
bos_id = sp.PieceToId('<bos>')
eos_id = sp.PieceToId('<eos>')
pad_id = sp.PieceToId('<pad>')

assert bos_id != None
assert eos_id != None
assert pad_id != None
