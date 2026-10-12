# Environment setup

Step-by-step equivalent of the [Dockerfile](Dockerfile) (PyTorch 2.4.1 / CUDA 12.1,
Flash Attention 2.6.3 with its CUDA extensions, xFormers 0.0.28, lm-evaluation-harness 0.4.4).
`environment.yaml` holds the frozen environment of that image.

```sh
conda create -n dlm-scaling python=3.9
conda activate dlm-scaling

pip install torch torchvision torchaudio

# install flash-attention (the rotary / layer_norm / xentropy csrc extensions are required)
pip uninstall ninja -y && pip install ninja -U

git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install packaging
python setup.py install

cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../../.. && rm -r flash-attention

# install xformers
pip install -U xformers

# install TinyLlama requirements
git clone https://github.com/jzhang38/TinyLlama.git
cd TinyLlama
pip install -r requirements.txt tokenizers sentencepiece
cd .. && rm -r TinyLlama

# dependencies needed for evaluation
pip install lm-eval==0.4.4 numpy==1.25.0 bitsandbytes==0.43.1
```
