python -m tests.diffuser datasets/hex_t096_c16_u18.npz
read -p "Press Enter to continue..."
python -m tests.load_dataset
read -p "Press Enter to continue..."
python  -m tests.pen_rhombus_reparametrize
read -p "Press Enter to continue..."
python  -m tests.pen_shapes
read -p "Press Enter to continue..."
python  -m tests.pregen_hex
read -p "Press Enter to continue..."
python  -m tests.pregen_pen
read -p "Press Enter to continue..."
python -m tests.analyze_diffuser_variance datasets/hex_t096_c16_u18.npz
read -p "Press Enter to continue..."
python -m tests.research.compare_losses
read -p "Press Enter to continue..."
python -m scripts.create_dataset 6 24 1
read -p "Press Enter to continue..."
python -m scripts.sample checkpoints/cp0114_0100_t096_e099.pt
read -p "Press Enter to continue..."
python train.py
read -p "Press Enter to continue..."
python train.py toy datasets/hex_t096_c01_u18.npz
read -p "Press Enter to continue..."
python train.py checkpoints/cp0114_0100_t096_e099.pt
