from pathlib import Path
import sys
import torch

from code.data.load import MyDataset
from code.models import get_model_class
from code.utils.advanced import xyac_to_svgs
from code.utils.basic import print_config
from code.utils.lossy import hex_lattice_loss_quadratic

def main():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} checkpoint.pt data.npz")
        sys.exit()

    cp_path = Path(sys.argv[1])
    assert cp_path.exists(), f"Checkpoint {cp_path} not found."
    data_path = Path(sys.argv[2])
    assert data_path.exists(), f"Data {data_path} not found."
    dataset = MyDataset(data_path)

    # 2. Load Checkpoint
    print(f"Loading {cp_path}...")
    cp = torch.load(cp_path, map_location=device)
    config = cp['config']
    print_config(config)

    # 3. Initialize Models
    Model = get_model_class(config['model']['model'])
    model = Model(config['model']).to(device)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()

    # 4. Extract Metadata
    symmetry = cp['symmetry']
    side = cp['side']
    class_lookup = cp['class_lookup']


    # 5. Interactive Loop
    i = 0
    TEN = 10

    print("\n--- Interactive Sampler ---")
    print("Press Enter to use random class, type a number/name to select, or 'q' to quit.")

    while True:
        xya, colors, labels = dataset[i:i+TEN]
        names = [class_lookup[c] for c in labels]

        paths = [f"library/samples/{cp_path.stem}_i{i:02d}_{sample_name}.svg"
                 for (i, sample_name) in zip(range(TEN), names)]

        samples = model.passthrough(xya, colors, labels)

        # stack orignial xya, colors, then append them to samples
        xyac = torch.cat([xya, 2+colors.unsqueeze(-1)], dim=-1)
        samples = torch.cat((xyac, samples), dim=1)

        xyac_to_svgs(samples, symmetry, side, paths, True) # type: ignore

        lattice_loss = hex_lattice_loss_quadratic(samples, side)
        print(f"Lattice loss: {lattice_loss:.4f}")

        i += TEN

        input("Press Enter to continue...")

if __name__ == "__main__":
    main()