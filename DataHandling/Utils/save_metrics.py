import json
import os
from typing import List, Tuple, Literal


def save_losses(model, training_losses: List[float], validation_losses: List[float]):
    loss_data = {
        "training_losses": training_losses,
        "validation_losses": validation_losses
    }
    os.makedirs("results/training-metrics", exist_ok=True)
    file_name = f"results/training-metrics/losses_{model.identifier()}.json"
    with open(file_name, "w") as f:
        json.dump(loss_data, f, indent=2)

    print(f"Saved training/validation losses to {file_name}.")

def load_losses(file_name) -> Tuple[List[float], List[float]]:
    with open(file_name, "r") as f:
        loss_data = json.load(f)

    training_losses = loss_data.get("training_losses", [])
    validation_losses = loss_data.get("validation_losses", [])

    return training_losses, validation_losses

def save_score(file_base: str, score: float, score_type: Literal["bleu", "perplexity"] = "bleu"):
    data = {
        "score": score
    }
    os.makedirs(f"results/{score_type}", exist_ok=True)
    file_name = f"results/{score_type}/{file_base}.json"
    with open(file_name, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {score_type} score to {file_name}.")

def load_score(file_base, score_type: Literal["bleu", "perplexity"] = "bleu") -> float:
    file_name = f"results/{score_type}/{file_base}.json"
    with open(file_name, "r") as f:
        bleu_data = json.load(f)
    return bleu_data.get("score", -1)
