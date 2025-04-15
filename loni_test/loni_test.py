import torch
import os

print("Hello world")

# Read and print lines from a text file
with open("./dir/something.txt") as f:
    line = f.readline()
    while line:
        print("Printing to standard console? ", line.strip())
        line = f.readline()

# Check for CUDA and set device
is_cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda_available else "cpu")
print("CUDA available:", is_cuda_available)

# Simulate slow operation: large matrix multiplication
print("Performing matrix multiplication on", device)
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)
c = torch.matmul(a, b)

# Store result in .prf format (PyTorch binary format)
os.makedirs(".loni_test/dir", exist_ok=True)
output_path = ".loni_test/dir/test-matrix-mul.prf"
torch.save(c, output_path)

print(f"Saved result to {output_path}")
