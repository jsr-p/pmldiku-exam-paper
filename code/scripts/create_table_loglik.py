import pandas as pd
import pmldiku

FNAME = "marginalloglikvalues.txt"

if __name__ == "__main__":
    file = pmldiku.FP_OUTPUT / "loglik" / FNAME
    df = pd.read_csv(file, header=None, names=["model", r"$\log p(x)$"])
    df["model"] = [" ".join(x.split("-")[1:]) for x in df["model"]]
    df[r"$\log p(x)$"] = df[r"$\log p(x)$"].round(2)
    df = df.T
    table = df.to_latex(escape=False)
    print(df)
    with open(pmldiku.FP_OUTPUT / "loglik.tex", "w") as file:
        file.write(table)
