from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import pandas as pd
from datetime import datetime

from core.data.DataHandler import DataHandler
from core.analytics.Utils import Utils

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize handlers with your data file
DATA_FILE_PATH = "backend/core/data/SPY_Options_log.txt"
data_handler = DataHandler()
utils = Utils()

# Load data once when starting the server
try:
    df = data_handler.parse_file(DATA_FILE_PATH)
    df = data_handler.get_basic_data(df)
except Exception as e:
    print(f"Error loading initial data: {str(e)}")
    df = None

@app.get("/api/health")
async def health_check():
    """Check if data is loaded and API is ready"""
    if df is not None:
        return {"status": "healthy", "data_loaded": True}
    return {"status": "unhealthy", "data_loaded": False}

@app.get("/api/timestamps")
async def get_timestamps():
    """Get all available timestamps for dropdown selection"""
    try:
        timestamps = sorted(df['timestamp'].unique().tolist())
        return {"timestamps": timestamps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/volatility-surface")
async def get_volatility_surface(timestamp: Optional[str] = None, interpolated: bool = False):
    """Get volatility surface data with optional interpolation"""
    try:
        data = df.copy()
        if timestamp:
            data = data[data['timestamp'] == timestamp]
        
        if interpolated:
            grid_x, grid_y, grid_z = utils.interpolate_surface(data)
            return {
                "x": grid_x.tolist(),
                "y": grid_y.tolist(),
                "z": grid_z.tolist()
            }
        else:
            return {
                "days_to_expiry": data['days_to_expiry'].tolist(),
                "log_strike": data['log_strike'].tolist(),
                "implied_volatility": data['implied_volatility'].tolist()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/volatility-heatmap")
async def get_volatility_heatmap():
    """Get binned data for heatmap visualization"""
    try:
        binned_data = utils.create_heatmap_data(df)
        return {"z_values": binned_data.values.tolist(), "x_labels": binned_data.columns.tolist(), "y_labels": binned_data.index.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/volatility-smile")
async def get_volatility_smile():
    """Get volatility smile data"""
    try:
        smile_data = utils.create_smile_data(df)
        return {"x_values": smile_data.index.tolist(), "y_values": smile_data.columns.tolist(), "z_values": smile_data.values.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))