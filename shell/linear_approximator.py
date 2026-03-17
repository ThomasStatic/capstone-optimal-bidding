from typing import Any, Optional
import numpy as np
import numpy.typing as npt
import pandas as pd

PRICE_COL = "lmpPrice"
HIST_LOAD_COL = "load_MWH"
FORECAST_LOAD_COL = "forecast"
LMP_CSV_PATH = "lmp-hourly-FORMOSA_CC2-ERCOT-SPP.csv"

class Discretizer:
    """
    Evaluates multiple binning methods and automatically selects the optimal one based on sparsity + imbalance.
    """

    def __init__(self, col, n_bins=8):
        self.col = col
        self.n_bins = n_bins
        self.methods = ["uniform", "quantile", "sigma", "hybrid"]
        self.edges_: Optional[npt.NDArray[np.floating[Any]]] = None
        self.best_method_ = None
        self.metrics_ = {}

    # Fit the edges for each method optn

    def _fit_edges(self, s, method, sigma_k=2, high_q=0.9):
        s = s.dropna()

        if method == "uniform":
            return np.linspace(s.min(), s.max() + 1e-9, self.n_bins + 1)

        elif method == "quantile":
            qs = np.linspace(0, 1, self.n_bins + 1)
            edges = np.unique(s.quantile(qs).to_numpy())
            if len(edges) < 3:
                return np.linspace(s.min(), s.max() + 1e-9, self.n_bins + 1)
            return edges

        elif method == "sigma":
            mu, sigma = s.mean(), s.std()
            if sigma == 0:
                return np.linspace(s.min(), s.max() + 1e-9, 4)
            edges = [s.min()] + [mu + i*sigma for i in range(-sigma_k, sigma_k+1)] + [s.max()]
            return np.unique(np.array(sorted(edges)))

        elif method == "hybrid":
            mid_bins = self.n_bins - 2
            s_min, s_max = s.min(), s.max()
            q_high_val = s.quantile(high_q)
            s_mid = s[(s >= 0) & (s <= q_high_val)]

            if s_mid.empty:
                return np.linspace(s_min, s_max + 1e-9, self.n_bins + 1)

            qs_mid = np.linspace(0, 1, mid_bins + 1)
            mid_edges = np.unique(s_mid.quantile(qs_mid).to_numpy())
            if len(mid_edges) < mid_bins + 1:
                mid_edges = np.linspace(0, q_high_val, mid_bins + 1)

            mid_edges[0] = 0.0
            mid_edges[-1] = q_high_val

            return np.unique(np.concatenate([[s_min], mid_edges, [s_max]]))

        else:
            raise ValueError("unknown method")

    # Define the metrics: sparsity and imbalance

    def _compute_metrics(self, s, edges):
        counts, _ = np.histogram(s.dropna(), bins=edges)
        sparsity = (counts == 0).mean()
        nz = counts[counts > 0]
        imbalance = nz.max()/nz.min() if len(nz) > 1 else 1.0
        return counts, sparsity, imbalance

    # Fit the class

    def fit(self, df):
        s = df[self.col]

        # Evaluate all methods
        for m in self.methods:
            edges = self._fit_edges(s, m)
            counts, sparsity, imbalance = self._compute_metrics(s, edges)
            self.metrics_[m] = {
                "edges": edges,
                "counts": counts,
                "sparsity": sparsity,
                "imbalance": imbalance,
            }

        # Choose optimal method: minimize sparsity, then imbalance
        min_sparsity = min(v["sparsity"] for v in self.metrics_.values())
        sparsity_methods = [
            m for m in self.methods
            if abs(self.metrics_[m]["sparsity"] - min_sparsity) < 1e-9
        ]

        best = min(sparsity_methods, key=lambda m: self.metrics_[m]["imbalance"])

        # Prefer quantile if tied or close
        if "quantile" in sparsity_methods:
            q_imb = self.metrics_["quantile"]["imbalance"]
            best_imb = self.metrics_[best]["imbalance"]
            if q_imb <= best_imb + 1e-9:
                best = "quantile"

        self.best_method_ = best
        self.edges_ = self.metrics_[best]["edges"]

        return self

    # Transform a new dataframe

    def _transform(self, df):
        if self.edges_ is None:
            raise RuntimeError("Call fit() before transform().")
        s = df[self.col]
        idx = np.digitize(s.to_numpy(), self.edges_) - 1
        return np.clip(idx, 0, len(self.edges_) - 2)

    # Fit and transform 

    def fit_transform(self, df):
        self.fit(df)
        return self._transform(df)




# Example usage for LMP
if __name__ == "__main__":
    df = pd.read_csv(LMP_CSV_PATH)

    disc = Discretizer(col=PRICE_COL, n_bins=8)
    df["lmp_bin"] = disc.fit_transform(df)
