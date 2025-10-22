echo "--- Initializing Conda and accepting ToS ---"
source ~/miniconda3/bin/activate
conda init bash
conda config --set auto_activate_base false
conda tos accept --override-channels --channel pkgs/main
conda tos accept --override-channels --channel pkgs/r

# --- Source Conda hooks to make 'conda activate' available ---
# This is the key step to allow 'conda activate' to work inside the script
echo "--- Sourcing Conda configuration ---"
source ~/miniconda3/etc/profile.d/conda.sh

# --- Create and Activate the Environment ---
# Now we can use the 'conda' command directly
echo "--- Creating and activating the 'iris' environment ---"
conda create -n iris python=3.10 --yes
conda activate iris
conda install pip
conda install -c conda-forge libstdcxx-ng --yes
pip install torch==2.5.1
pip install flash_attn==2.7.2.post1 --no-build-isolation
pip install -r requirements.txt


cd src/t2i-r1/src/utils/LLaVA-NeXT
pip install -e ".[train]"

cd ../GroundingDINO
pip install -e .

git config core.fileMode false

cd ../../../../../reward_weight
bash download.sh