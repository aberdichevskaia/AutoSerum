# for generation 
conda create -n carlini python=3.12 --override-channels -c conda-forge -c defaults -y
conda activate carlini
pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install transformers==4.55.4 tqdm==4.67.1 numpy==2.0.2 tokenizers==0.21.4 huggingface_hub==0.34.4 safetensors==0.6.2

# for dataset building
pip install --no-cache-dir "datasets==2.21.0"
pip install --no-cache-dir zstandard
pip install --no-cache-dir "fsspec[http]==2024.6.1" "aiohttp>=3.8,<4"