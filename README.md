# SPAR3D for Partial Input Point Clouds

Forked from [SPAR3D](https://github.com/Stability-AI/stable-point-aware-3d). Thank you very much!

### Requesting Access and Login

SPAR3D is gated at [Hugging Face](https://huggingface.co):
1. Log in to Hugging Face and request access [here](https://huggingface.co/stabilityai/stable-point-aware-3d).
2. Create an access token with read permissions [here](https://huggingface.co/settings/tokens).
3. Run `huggingface-cli login` in the environment and enter the token.


### Installation

```bash
conda create -n spar python=3.10
conda activate spar
pip install torch
pip install --no-build-isolation git+https://github.com/SunzeY/AlphaCLIP.git
pip install -r requirements.txt --no-build-isolation
pip install -r requirements-remesh.txt
pip install flet==0.21.2
pip install "pyglet<2"
```

### Manual Inference

```sh
python3 run.py demo_files/examples/fish.png --output-dir output/ --low-vram-mode
```

