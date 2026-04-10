import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd

from engine.data import get_data
from engine.utils import calculate_returns, z_score_normalize
from engine.decompose import slice_signal, create_labels
from engine.coherence import calculate_coherence
from engine.granger import analyze_causal_flow

load_dotenv()
app = FastAPI(title="FinSignal Suite API", description="Production DSP engine for technical analysis", version="1.0.0")

class AssetRequest(BaseModel):
    symbol: str

class PairRequest(BaseModel):
    first: str
    second: str
    
@app.get("/")
def check_health():
    return {"status": "ok", "message": "FinSignal API is up and running."}

@app.post("/api/v1/decompose")
def api_decompose(payload: AssetRequest):
    prices = get_data(payload.symbol)
    if prices is None:
        raise HTTPException(status_code=404, detail=f"No data found for {payload.symbol}")
        
    returns = calculate_returns(prices)
    norm_data = z_score_normalize(returns).dropna()
    
    depth = 5
    bands = slice_signal(norm_data.values, depth=depth)
    band_names = create_labels(depth)
    
    results = {}
    for i, name in enumerate(band_names):
        if i < len(bands):
            results[name] = bands[i].tolist()
            
    return {"symbol": payload.symbol, "components": results}

@app.post("/api/v1/coherence")
def api_coherence(payload: PairRequest):
    prices1 = get_data(payload.first)
    prices2 = get_data(payload.second)
    if prices1 is None or prices2 is None:
        raise HTTPException(status_code=404, detail="One or both asset symbols not found.")
        
    min_size = min(len(prices1), len(prices2))
    r1 = calculate_returns(prices1).tail(min_size)
    r2 = calculate_returns(prices2).tail(min_size)
    
    aligned_data = pd.concat([r1, r2], axis=1).dropna()
    y1 = z_score_normalize(aligned_data.iloc[:, 0]).tail(750).values
    y2 = z_score_normalize(aligned_data.iloc[:, 1]).tail(750).values
    
    res_map, phase, coi, freqs, sig = calculate_coherence(y1, y2)
    
    return {
        "assets": [payload.first, payload.second],
        "frequencies": freqs.tolist(),
        "map_shape": res_map.shape,
        "average_resonance": float(np.mean(res_map))
    }

@app.post("/api/v1/causality")
def api_causality(payload: PairRequest):
    prices1 = get_data(payload.first)
    prices2 = get_data(payload.second)
    if prices1 is None or prices2 is None:
        raise HTTPException(status_code=404, detail="Symbols not found.")
        
    min_size = min(len(prices1), len(prices2))
    r1 = calculate_returns(prices1).tail(min_size)
    r2 = calculate_returns(prices2).tail(min_size)
    
    aligned_data = pd.concat([r1, r2], axis=1).dropna()
    input_stack = aligned_data.tail(1000).values
    
    bins, push_yx, push_xy = analyze_causal_flow(input_stack, maxlag=5)
    
    return {
        "candidate": payload.first,
        "target": payload.second,
        "frequencies": bins.tolist(),
        "causal_strength_fwd": push_yx.tolist(),
        "causal_strength_bwd": push_xy.tolist()
    }

