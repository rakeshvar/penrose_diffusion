read -p "Press Enter to continue... analyze_diffuser_variance"
echo "###############################################      analyze_diffuser_variance"
python -m tests.analyze_diffuser_variance datasets/hexxy_t096_c96_u18.npz
read -p "Press Enter to continue... diffuser_test"
echo "###############################################      diffuser_test"
python -m tests.diffuser_test datasets/hexxy_t096_c16_u18.npz
read -p "Press Enter to continue... generator_test"
echo "###############################################      generator_test"
python -m tests.generator_test 128
read -p "Press Enter to continue... load_dataset"
echo "###############################################      load_dataset"
python -m tests.load_dataset
read -p "Press Enter to continue... pen_rhombus_reparametrize"
echo "###############################################      pen_rhombus_reparametrize"
python -m tests.pen_rhombus_reparametrize
read -p "Press Enter to continue... pen_shapes"
echo "###############################################      pen_shapes"
python -m tests.pen_shapes
read -p "Press Enter to continue... pregen_hex"
echo "###############################################      pregen_hex"
python -m tests.pregen_hex
read -p "Press Enter to continue... pregen_pen"
echo "###############################################      pregen_pen"
python -m tests.pregen_pen
