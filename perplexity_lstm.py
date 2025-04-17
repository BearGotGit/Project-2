import math
from DataHandling import save_score
from DataHandling.Utils import load_losses

file_base = "losses_lstm-04-16-2025_09-54pm"
_, validation_losses = load_losses(f"./results/training-metrics/{file_base}.json")
avg_cross_entropy_loss = sum(validation_losses) / len(validation_losses)
perplexity = math.exp(avg_cross_entropy_loss)
print(f"Perplexity: {perplexity:.4f}")

save_score(file_base=file_base, score=perplexity, score_type="perplexity")
