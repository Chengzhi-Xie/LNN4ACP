# ESM2
conda create -n esm2_cpu python=3.10 -y
conda activate esm2_cpu
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install fair-esm==2.0.0 numpy pandas scikit-learn

# ProtT5
conda create -n prott5-cpu python=3.10 -y
conda activate prott5-cpu
conda install pytorch cpuonly -c pytorch -y
pip install -U transformers sentencepiece accelerate numpy pandas scikit-learn
pip uninstall -y google
pip install -U protobuf>=4.25.0 sentencepiece>=0.1.99 transformers>=4.45 packaging accelerate
