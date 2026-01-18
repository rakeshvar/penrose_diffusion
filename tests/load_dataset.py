from pathlib import Path

from code.data.load import MyDataset
from code.utils import npz_stats

for path in Path(".").rglob("*.npz"):
    print("#"* 50, "\nOpening: ", path)
    try:
        dataset = MyDataset(path)
        print(dataset)
        npz_stats(path)
    except Exception as e:
        print(f"Error opening file {path}:\n\t", e)
