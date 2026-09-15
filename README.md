# PCA from Scratch (SVD-based)

PCA is not "a dimensionality reduction trick." It's a change of basis onto the
directions of maximum variance ,the eigenvectors of the covariance matrix.
This repo implements it from scratch using SVD (numerically stable, and what
`sklearn` does internally), verifies it against `sklearn.decomposition.PCA`,
and applies it to an embedding-reduction task with real measured results.

## Results

 Check                                                                                   Result 

 Match vs. `sklearn.decomposition.PCA` (sign-aligned)                   max abs diff `< 1e-10` (machine precision) 
 Embedding reduction                                                     128-dim → 12-dim, **86.3%** variance retained 
 Clustering quality (silhouette score)                                   0.607 → 0.864(improved — noise dims removed) 
 Inference throughput (20,000 queries, cluster assignment)               ~4–5x faster on reduced representation 
 Core implementation size                                                19 lines of NumPy (`fit` + `transform`) 

Exact numbers vary slightly run-to-run due to `KMeans` random initialization 
see the notebooks for the live values from the latest run.

## Why SVD, not eigendecomposition of the covariance matrix?

Computing `Σ = XᵀX` and eigendecomposing it squares the condition number of
the data, which amplifies floating-point error at high dimensions. Running
SVD directly on the centered data matrix `X = UΣVᵀ` avoids that step entirely
— principal components are just the rows of `Vᵀ`. This is why `sklearn` does
the same thing internally.

## Repo structure

```
pca-from-scratch/
├── README.md
├── pca_from_scratch_explained.ipynb   # full walkthrough with math + commentary
└── pca_from_scratch_code_only.ipynb   # clean code, no explanation
```

## How to run

 `.ipynb` in Jupyter/Colab and run all cells.

## Implementation

```python
class PCAScratch:
    def __init__(self, n_components):
        self.k = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[: self.k]
        self.explained_variance_ratio_ = (S[: self.k] ** 2) / np.sum(S ** 2)
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T
```

## Verification methodology

PCA components are unique only up to sign and ordering, so a naive
subtraction against `sklearn`'s output is the wrong test. Each component is
sign-aligned via correlation before comparing:

```python
corr = np.corrcoef(mine[:, i], sklearn_out[:, i])[0, 1]
sign = np.sign(corr)
diff = np.max(np.abs(mine[:, i] * sign - sklearn_out[:, i]))
```
