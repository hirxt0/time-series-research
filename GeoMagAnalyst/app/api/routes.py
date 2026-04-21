from fastapi import APIRouter, UploadFile, File, Form
from app.services.pipeline import run_pipeline

router = APIRouter()


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    impute_model: str = Form(...),
    anomaly_model: str = Form(...)
):
    result = await run_pipeline(file, impute_model, anomaly_model)
    return result

