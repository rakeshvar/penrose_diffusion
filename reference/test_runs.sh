#!/bin/bash
# source from one directory above

# Fancy banner function with colored text and bold borders
banner() {
    local msg="$1"
    local color_code="$2"  # ANSI code like "1;32" (bold green) or "1;36" (bold cyan)
    local reset="\e[0m"
    local color="\e[${color_code}m"
    local len=${#msg}
    local border_length=$((len + 4))
    local border=$(printf '═%.0s' $(seq 1 $border_length))

    echo
    echo -e "\e[1m╔${border}╗${reset}"
    echo -e "\e[1m║${reset}  ${color}${msg}${reset}  \e[1m║${reset}"
    echo -e "\e[1m╚${border}╝${reset}"
    echo
}

# Local no saving samples
banner "Local – No Samples Saved" "1;32"  # Bold green
python train.py gs://penrose_diffusion/datasets/hex_t096_c16_u18.npz toy -w enable=False -t save_samples=False -o .

# Local saving samples
banner "Local – With Samples Saved" "1;32"  # Bold green
python train.py gs://penrose_diffusion/datasets/hex_t096_c16_u18.npz toy -w enable=False -t save_samples=True -o .

# GCS no saving samples
banner "GCS – No Samples Saved" "1;36"  # Bold cyan
python train.py gs://penrose_diffusion/datasets/hex_t096_c16_u18.npz toy -w enable=False -t save_samples=False -o gs://penrose_toy

# GCS saving samples
banner "GCS – With Samples Saved" "1;36"  # Bold cyan
python train.py gs://penrose_diffusion/datasets/hex_t096_c16_u18.npz toy -w enable=False -t save_samples=True -o gs://penrose_toy
