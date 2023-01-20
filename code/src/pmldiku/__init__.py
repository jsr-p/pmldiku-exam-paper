from pathlib import Path

FP_PROJ = Path(__file__).parents[2]
FP_MODELS = FP_PROJ / "models"
FP_FIGS = FP_PROJ / "figs"
FP_OUTPUT = FP_PROJ / "output"

for fp in [FP_MODELS, FP_FIGS, FP_OUTPUT]:
    Path.mkdir(fp, exist_ok=True)
