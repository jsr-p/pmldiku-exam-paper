from pathlib import Path

FP_MODELS = Path(__file__).parents[2] / "models"
Path.mkdir(FP_MODELS, exist_ok=True)
