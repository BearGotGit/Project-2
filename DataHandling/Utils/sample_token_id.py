import torch

def sample_token_id_from_probability_distribution(prob_dist: torch.Tensor, p_sample_threshold=None) -> int:
    """
    Sample from a probability distribution with optional top-p (nucleus) sampling.
    If p_sample_threshold is None, falls back to greedy (argmax).

    :param prob_dist: Tensor, representing the probability distribution to sample from
    :param p_sample_threshold: Sample only from the top choices whose sum is less than p_sample_threshold (always chooses at least one, though)
    :return:
    """
    if p_sample_threshold is None:
        return torch.argmax(prob_dist, dim=-1).item()

    prob_dist = prob_dist / prob_dist.sum()
    sorted_probs, sorted_indices = torch.sort(prob_dist, descending=True)
    # It keeps dim
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    cutoff_mask = cumulative_probs <= p_sample_threshold
    # If first token were higher than p_sample_threshold, would be false by mask. Force true in all cases
    cutoff_mask[0] = True
    filtered_probs = sorted_probs[cutoff_mask]
    filtered_indices = sorted_indices[cutoff_mask]
    sampled_index = torch.multinomial(filtered_probs, 1).item()

    return filtered_indices[sampled_index].item()
