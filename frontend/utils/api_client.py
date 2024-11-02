import httpx
from typing import Optional, Dict, Any

class VolatilityAPIClient:
    """Client for interacting with the volatility API"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def get_health(self) -> Dict[str, Any]:
        """Check API health status"""
        response = await self.client.get(f"{self.base_url}/health")
        return response.json()

    async def get_timestamps(self) -> Dict[str, list]:
        """Get available timestamps"""
        response = await self.client.get(f"{self.base_url}/timestamps")
        return response.json()

    async def get_volatility_surface(self, timestamp: Optional[str] = None, 
                                   interpolated: bool = False) -> Dict[str, list]:
        """Get volatility surface data"""
        params = {"timestamp": timestamp, "interpolated": interpolated}
        response = await self.client.get(f"{self.base_url}/volatility-surface", 
                                       params=params)
        return response.json()

    async def get_volatility_heatmap(self) -> Dict[str, list]:
        """Get volatility heatmap data"""
        response = await self.client.get(f"{self.base_url}/volatility-heatmap")
        return response.json()

    async def get_volatility_smile(self, timestamp: str) -> Dict[str, Dict]:
        """Get volatility smile data"""
        response = await self.client.get(f"{self.base_url}/volatility-smile/{timestamp}")
        return response.json()

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()