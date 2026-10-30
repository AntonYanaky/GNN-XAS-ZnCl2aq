"""Serve this repository's aqueous ZnCl2 model to LightshowAI."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictInt
from pymatgen.core import Structure
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from data import XASGraphDataset, collate_fn
from models import XASLightningModule

ENERGY_EV = np.linspace(8900.0, 9050.0, 1000)[333:733:2]
ELEMENTS = ("O", "H", "Zn", "Cl")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    structure: dict
    absorbing_site: Literal["Zn"] = "Zn"
    spectroscopy_type: Literal["GNN"] = "GNN"
    site_indices: list[StrictInt] | None = None


class Predictor:
    def __init__(self):
        with (ROOT / "configs/config.yaml").open() as stream:
            config = yaml.safe_load(stream)
        self.device = torch.device(os.getenv("XAS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        self.cutoff = config["gnn"]["cutoff"]
        self.model = XASLightningModule(config["gnn"], config["head"])
        self.model.load_state_dict(torch.load(ROOT / "model/full_model.pth", map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()

    def predict(self, structure, indices=None):
        if any(not site.is_ordered for site in structure):
            raise ValueError("Disordered sites are not supported")
        symbols = [site.specie.symbol for site in structure]
        if set(symbols) - set(ELEMENTS):
            raise ValueError("Only O, H, Zn, and Cl are supported")
        if not np.isfinite(structure.frac_coords).all() or not np.isfinite(structure.lattice.matrix).all() or structure.volume <= 0:
            raise ValueError("Structure requires finite coordinates and a nonzero cell volume")
        zinc = [i for i, symbol in enumerate(symbols) if symbol == "Zn"]
        indices = zinc if indices is None else indices
        if not indices or len(set(indices)) != len(indices) or any(i not in zinc for i in indices):
            raise ValueError("Select unique, zero-based Zn indices from the submitted structure")

        spectra = {}
        for start in range(0, len(indices), 8):
            selected = indices[start:start + 8]
            structures = []
            for index in selected:
                # The checkpoint reads node zero. Keep original indices in the response.
                order = [index] + [i for i in range(len(structure)) if i != index]
                structures.append(Structure.from_sites([structure[i] for i in order]))
            dataset = XASGraphDataset(structures, np.zeros((len(selected), len(ENERGY_EV))), cutoff=self.cutoff)
            graph, _, _ = collate_fn([dataset[i] for i in range(len(dataset))])
            with torch.inference_mode():
                values = self.model(graph.to(self.device)).cpu().numpy()
            if values.shape != (len(selected), len(ENERGY_EV)) or not np.isfinite(values).all():
                raise RuntimeError("Model produced invalid spectra")
            spectra.update(zip(selected, values.tolist()))
        return spectra


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = Predictor()
    yield


app = FastAPI(title="ZnCl2 XAS", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])


def spectrum_response(spectra):
    return {
        "energy_eV": ENERGY_EV.tolist(),
        "reference_energy_shift_eV": 705.3,
        "spectra_by_site": {str(i): values for i, values in spectra.items()},
        "mean_spectrum": np.mean(list(spectra.values()), axis=0).tolist(),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/predict")
def predict(body: PredictRequest, request: Request):
    try:
        spectra = request.app.state.predictor.predict(Structure.from_dict(body.structure), body.site_indices)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return spectrum_response(spectra)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("XAS_PORT", "8000")))
