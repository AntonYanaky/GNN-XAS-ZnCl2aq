## Publication: 
Chuntian Cao et al., Deciphering the Solvation Structure of Aqueous ZnCl₂ Solutions from X-ray Absorption Spectra using Interpretable Graph Neural Network, The Journal of Physical Chemistry B (2026).
https://doi.org/10.1021/acs.jpcb.6c00464

## Dataset 
Zenodo: https://doi.org/10.5281/zenodo.19684914 

## Environment Setup and Installation

Two installation methods are provided:

1. Fully reproducible Conda environment `environment.yml`.

2. Manual installation. 

---

### 1. Conda environment installation 

For full reproducibility, a reference Conda environment is provided in `environment.yml`. 

```bash
conda env create -f environment.yml
conda activate ml-xas
```

#### GPU Support
This environment assumes CUDA 12.2 available as a system module
Example: **module load cuda/12.2**


### 2. Manual Installation 

Below are commands used to build the development environment. 


```bash

conda create -n ml-xas python=3.10 -y
conda activate ml-xas

conda install numpy==1.26.4 scipy matplotlib scikit-learn jupyterlab ipython -y

module load cuda/12.2

pip install torch==2.2.0 torchvision==0.17.0 torchaudio lightning --index-url https://download.pytorch.org/whl/cu121

pip install captum torch-summary 

pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install matgl==1.3.0

pip install umap-learn==0.5.9.post2 numba==0.61.2 llvmlite==0.44.0
pip install pymatgen ase "numpy<2"

pip install kaleido==0.2.1

python -m ipykernel install --user --name="jupyter_ml-xas"

```

## RPC service for LightshowAI

The `environment.yml` environment includes the packages required by the RPC server, including PyTorch, DGL, MatGL, pymatgen, FastAPI, and Uvicorn. Create it once from this repository in WSL:

```bash
conda env create -f environment.yml
```

If `ml-xas` already exists, skip environment creation. Start the server from the repository root:

```bash
conda activate ml-xas
python server/xas_rpc_server.py
```

The first start loads the model checkpoint and may take a short time. The server uses the GPU when PyTorch detects CUDA, otherwise it runs on the CPU. It listens only on `127.0.0.1:8000`.

In a second WSL terminal, check that the server is ready:

```bash
curl http://127.0.0.1:8000/health
```

The response should be `{"status":"ok"}`. You can also open `http://127.0.0.1:8000/docs` to inspect the request schema.

In LightshowAI, select **Zn VASP** and set its `ZN_XAS_RPC_URL` environment variable to:

```text
http://127.0.0.1:8000/v1/predict
```

Keep LightshowAI's pinned environment unchanged. LightshowAI must be able to reach the WSL localhost address. If LightshowAI runs in a separate container or on another machine, this loopback-only server setup will not be reachable as written.

- `GET /health`: readiness check.
- `POST /v1/predict`: send `{"structure": structure.as_dict()}`. Optional `site_indices` select original zero-based Zn indices. By default, every Zn is predicted separately and averaged. Responses preserve original indices and contain 200 points. LightshowAI adds the 705.3 eV display offset.

The server uses the existing config, graph code, and model checkpoint. Each selected absorber is moved to graph node zero internally. Only O, H, Zn, and Cl are supported. To force CPU use or choose another port, for example:

```bash
XAS_DEVICE=cpu XAS_PORT=8001 python server/xas_rpc_server.py
```

Update the LightshowAI URL to match the selected port. Press `Ctrl+C` in the server terminal to stop it.

