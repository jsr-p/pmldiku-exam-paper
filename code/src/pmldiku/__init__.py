from pathlib import Path

FP_PROJ = Path(__file__).parents[2]
FP_MODELS = FP_PROJ / "models"
FP_FIGS = FP_PROJ / "figs"
Path.mkdir(FP_MODELS, exist_ok=True)
