#!/bin/bash
#SBATCH --job-name=train_rnn_lstm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --time=05:00:00
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

echo "Training RNN..."
python3 train_rnn.py

echo "Training LSTM..."
python3 train_lstm.py

# echo "Training Transformer..."
# python train_transformer.py

echo "All training runs completed."

