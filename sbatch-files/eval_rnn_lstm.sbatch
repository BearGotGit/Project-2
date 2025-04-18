#!/bin/bash
#SBATCH --job-name=eval_rnn_lstm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load cuda/11.8.0
module load python/3.12.7-anaconda

# Activate Conda
source ~/.bashrc
conda activate berend-grandt-csc-4700-ghawaly-project-2

# Only install if needed, remove -r (invalid)
# conda install --yes --file requirements.txt

cd /work/berend/Project-2

echo "Evaluating RNN..."
python3 bleu_rnn.py
python3 perplexity_rnn.py

echo "Evaluating LSTM..."
python3 bleu_lstm.py
python3 perplexity_lstm.py

# echo "Starting Transformer training..."
# python train_transformer.py

echo "Completed BLEU and Perplexity evaluations."

