from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import timedelta
from app.services.model_dashboard import ModelDashboard

router = APIRouter()
dashboard = ModelDashboard()

@router.get("/dashboard/models/{model_name}")
async def get_model_dashboard(
    model_name: str,
    days: Optional[int] = 7
):
    """Get dashboard data for a specific model"""
    try:
        return await dashboard.get_dashboard_data(
            model_name=model_name,
            time_range=timedelta(days=days)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dashboard/models/{model_name}/versions/{version_id}/deploy")
async def deploy_model_version(
    model_name: str,
    version_id: str,
    traffic_percentage: float = 10.0,
    rollback_threshold: Optional[dict] = None
):
    """Deploy a new model version with canary deployment"""
    try:
        await dashboard.manage_deployment(
            version_id,
            traffic_percentage,
            rollback_threshold
        )
        return {"status": "success", "message": "Deployment initiated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 