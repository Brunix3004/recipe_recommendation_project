# Technical Audit: Why Your K-Means Cannot See Clusters

**Project:** Food.com Recipe Data Visualization — Week 7 Clustering
**Scope:** Audit of `build_clustering_matrix.py` and its interaction with Week 5 PCA/SVD artifacts
**Date:** May 14, 2026

---

I read the Week 5 explanation and `build_clustering_matrix.py` end-to-end. The diagnosis is unambiguous: the Week 7 script preserves a **pathologically anisotropic geometry** that K-Means is mathematically incapable of separating. The Week 5 artifacts are largely fine; the Week 7 assembly is where the search space is being sabotaged. Below is a candid breakdown.

---

## 1. The Core Diagnosis (Short Version)

You are feeding K-Means a **208-dimensional matrix** in which:

- **200 of those dimensions** come from `TruncatedSVD` outputs whose **per-column variance follows the singular-value spectrum** (i.e. `var(col_k) ≈ σ_k² / n`). The first few columns dominate by 1–3 orders of magnitude over the tail.
- **8 of those dimensions** come from `PCA` that was **standardized** (`StandardScaler`) and then multiplied by `0.35`, giving them a per-column variance of `≈ 0.12`.
- The script's comment claims it "preserves semantic geometry." In practice it preserves **only the top 2–4 SVD axes**, which are dominated by `keyword__easy` and `category__dessert`. K-Means then partitions the dataset along that single axis and calls it a day.

K-Means with Euclidean distance assumes **isotropic, comparably-scaled features**. Your matrix violates that assumption on **two independent axes simultaneously** (within the SVD block, and across blocks). That is why the clusters overlap.

---

## 2. Specific Decisions in `build_clustering_matrix.py` Sabotaging the Geometry

### 2.1 Leaving the SVD block unscaled is the single biggest error

Lines 174–178 of `build_clustering_matrix.py`:

```python
# SVD is intentionally left unscaled:
# 1) TF-IDF normalization already happened upstream.
# 2) We preserve the semantic weighting structure learned by TruncatedSVD.
# 3) We keep the TruncatedSVD variance hierarchy intact.
X_content_final = (X_content_svd * CONTENT_WEIGHT).astype(np.float32, copy=False)
```

The justification is mathematically wrong for a distance-based algorithm:

- **"TF-IDF normalization already happened upstream"** is irrelevant. L2 normalization of TF-IDF rows does **not** propagate to the SVD output. After `TruncatedSVD`, column `k` has variance proportional to `σ_k²`. From your Week 5 metrics (200 components retain only 52.7% of energy with monotonic decay), `σ_1²` is at least 30–50× larger than `σ_200²`.
- **"Preserving the variance hierarchy"** is exactly what kills K-Means. K-Means computes `||x_i − μ_c||²`, which sums squared differences across all dimensions. If `var(col_1)` is 50× `var(col_50)`, then `col_1` contributes 50× more to every distance computation. The algorithm effectively clusters on **the first 3–5 SVD components only**.
- Your Week 5 docs literally describe SVD1 as the **`easy/dessert/basic ingredients`** axis and SVD3 as **`quick/easy category-heavy`**. K-Means is thus splitting the catalog into "easy generic stuff" vs "everything else." That is the overlap you're observing.

### 2.2 The numeric block contribution is effectively zero

```python
CONTENT_WEIGHT = 1.0
NUMERIC_WEIGHT = 0.35
```

Compute the **block-level Frobenius energy ratio** before clustering. After your transforms:

- Numeric block: 8 columns × `var ≈ 0.35² × 1.0 = 0.1225` ⇒ total ≈ **0.98**
- Content block (just SVD1 alone, conservative estimate using your "0.527 retained energy across 200 components" with strong front-loading): ≈ **5–20+** in the first few columns alone, plus another ~10–30 across the tail.

So the numeric block contributes roughly **2–5%** of total squared distance. Your scaling intent ("numeric refines but doesn't overpower") was reasonable; your implementation gave numeric features no voice at all. K-Means cannot use nutrition or time to separate clusters because those features are statistically invisible in the distance metric.

### 2.3 Standardizing the numeric PCA at this stage is also incorrect

```python
def scale_numeric_pca(X_numeric_pca: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_numeric_pca_scaled = scaler.fit_transform(X_numeric_pca).astype(np.float32)
    return X_numeric_pca_scaled, scaler
```

PCA components are **already orthogonal and ordered by variance**. By forcing each PC to unit variance you **destroy the variance hierarchy you preserved on the SVD side**. So the script applies *opposite* transformations to the two blocks: it whitens the numeric block (flattening its spectrum) while leaving the content block fully anisotropic. There is no consistent geometric philosophy here — it is the worst of both worlds.

If you keep the standardization, you must also standardize SVD. If you remove it, PC1 (overall nutritional richness) will dominate the numeric block, which is at least internally coherent.

### 2.4 The 200-dimensional SVD block triggers distance concentration

K-Means in 208 dimensions on 522,517 points suffers from the **concentration of distances** phenomenon: as `d → ∞`, `(max_dist − min_dist)/min_dist → 0`, so all points look equidistant from each centroid. With 200 noisy SVD dimensions (only 52.7% energy retained — nearly half is noise), the algorithm is being asked to find structure in a high-dimensional fog.

200 components were a defensible choice for **content-based recommendation** (retrieval is robust to dimensionality). It is a poor choice for **K-Means clustering**. These two downstream tasks should not share the same `k`.

### 2.5 No L2 row normalization before clustering

This is the standard LSA + clustering recipe (Schütze, Manning, etc.): when clustering TF-IDF/SVD vectors, you normalize each row to unit norm so that **Euclidean distance becomes a monotonic function of cosine distance** (`||a − b||² = 2 − 2 cos(a, b)` for unit vectors). Without this, recipes with many ingredient tokens have larger SVD norms and sit farther from the origin, dragging cluster centroids around independently of semantic content. Your 200-dim SVD block, applied raw, encodes *recipe verbosity* as much as recipe content.

### 2.6 Concatenation, not fusion

```python
def build_clustering_matrix(
    X_content_final: np.ndarray,
    X_numeric_final: np.ndarray,
) -> np.ndarray:
    X_clustering = np.hstack([X_content_final, X_numeric_final]).astype(np.float32, copy=False)
    return X_clustering
```

Naive `hstack` between heterogeneous representations (200-d sparse-derived semantic vs 8-d dense scaled numeric) without a shared subspace is a known anti-pattern. There is no joint manifold being learned — you are asking K-Means to do feature fusion *and* clustering simultaneously, which it cannot do.

---

## 3. Is the SVD/PCA Integration Mathematically Coherent for K-Means?

**No.** Concretely:

| Property | `X_numeric_pca` (after script) | `X_content_svd` (after script) |
|---|---|---|
| Per-column variance | ≈ `0.35²` (constant, whitened) | proportional to `σ_k²` (steep decay) |
| Row norm | bounded, comparable | varies with recipe content density |
| Semantic basis | linear comb. of standardized numeric features | linear comb. of TF-IDF tokens (sparse origin) |
| Implicit metric | Euclidean (well-defined) | cosine (not Euclidean) |

You are mixing two vectors that live in **different metric assumptions**. The numeric PCA respects Euclidean geometry by construction (PCA on standardized features is the Euclidean-optimal linear projection). The content SVD respects **cosine** geometry by construction (LSA convention). Concatenating them as if they shared a metric is the root incoherence.

For distance-based clustering you must either:

1. **Project content into a Euclidean-consistent space** (L2-normalize SVD rows + standardize columns), OR
2. **Switch to spherical K-Means / cosine K-Means** and treat the whole problem as an angular one (this is what most practitioners do for LSA-derived embeddings).

---

## 4. Why the Model Cannot See Boundaries

Three compounding effects, in order of impact:

1. **Anisotropy collapse.** The first 3 SVD components carry ~15–25% of total energy across only 1.5% of the columns. K-Means optima collapse onto these directions, producing 2–4 elongated, overlapping clusters split along the "easy generic" axis. Everything else is noise to the algorithm.
2. **Effective cluster signal washed out.** The features that *would* produce culinary boundaries — nutritional density (PC1), time/complexity (PC2), carb-vs-protein contrast (PC3), and the *tail* of SVD components encoding cuisine and ingredient combinations — contribute negligibly to the distance metric. They exist in the matrix but not in the geometry.
3. **High-D distance concentration.** With 200 noisy dimensions and >500k points, centroid–point distances concentrate around their mean, so cluster assignments become nearly random for points not close to a dominant SVD axis. Silhouette scores in this setup are typically in the 0.02–0.08 range — exactly the "blob with no separation" symptom you describe.

Additionally: the keyword `easy` appears in 276,838 of 522,517 recipes (53%). It alone almost certainly defines SVD1. Until you suppress generic tags, no clustering algorithm will produce meaningful structure on top of this content matrix.

---

## 5. Recommended Architectural Changes

I would restructure `build_clustering_matrix.py` along these lines. Below is a proposed replacement for the core transformation logic — the surrounding scaffolding (argparse, validation, save) can stay.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

def build_clustering_matrix_v2(
    X_content_svd: np.ndarray,   # (N, 200) raw SVD output
    X_numeric_pca: np.ndarray,   # (N, 8) raw PCA output
    *,
    n_content_kept: int = 50,
    n_joint_components: int = 30,
    content_weight: float = 1.0,
    numeric_weight: float = 1.0,
) -> np.ndarray:
    """
    Geometry-aware clustering matrix:
      1) Truncate SVD tail (keep top n_content_kept components only).
      2) Standardize each block column-wise (isotropic per-block geometry).
      3) L2-normalize SVD rows (cosine ~ Euclidean for the semantic block).
      4) Block-equalize: rescale so each block contributes equal Frobenius norm,
         then apply the user-facing weights with predictable meaning.
      5) Optional joint PCA to a low-D shared subspace where KMeans behaves.
    """
    X_content = X_content_svd[:, :n_content_kept].astype(np.float64, copy=True)
    X_numeric = X_numeric_pca.astype(np.float64, copy=True)

    X_content = StandardScaler().fit_transform(X_content)
    X_numeric = StandardScaler().fit_transform(X_numeric)

    X_content = normalize(X_content, norm="l2", axis=1)

    content_energy = np.linalg.norm(X_content, ord="fro")
    numeric_energy = np.linalg.norm(X_numeric, ord="fro")
    X_content *= (content_weight / content_energy)
    X_numeric *= (numeric_weight / numeric_energy)

    X_joint = np.hstack([X_content, X_numeric])
    X_joint = PCA(n_components=n_joint_components, whiten=True,
                  random_state=42).fit_transform(X_joint)

    return X_joint.astype(np.float32, copy=False)
```

### What each step buys you

| Step | Fixes |
|---|---|
| Truncate SVD to top ~30–50 components | Removes ~150 noisy dimensions; mitigates distance concentration; keeps the discriminative semantic axes. |
| `StandardScaler` on **both** blocks | Restores isotropy *within* each block. K-Means now treats SVD components equally instead of being captured by SVD1. |
| L2-normalize SVD rows | Aligns the semantic block with cosine geometry, removes recipe-verbosity bias. Equivalent to running spherical K-Means on the content side. |
| Frobenius-normalized block weights | Makes `content_weight` and `numeric_weight` *actually mean* the relative contribution they suggest. With both weights = 1.0 you get a true 50/50 split. |
| Joint PCA + whitening to ~30 dims | Forces a shared, isotropic subspace; eliminates the "two metrics in one vector" incoherence; produces a search space K-Means can reason about. |

### Companion changes I would also strongly recommend

1. **Use `MiniBatchKMeans`** instead of `KMeans` for 522k rows (10–50× faster, comparable quality). Run it for `k ∈ {5, 8, 10, 15, 20, 30}` and pick by silhouette + Davies–Bouldin on a 50k subsample.
2. **Add a Week 5b cleanup pass on the TF-IDF**: bump `max_df` down (e.g. `max_df=0.4` would have removed `easy`), and consider a curated stoplist for generic platform tags (`easy`, `inexpensive`, `beginner_cook`). These tags carry almost no culinary information and currently dominate SVD1.
3. **Run three parallel clusterings** for diagnostic comparison:
   - numeric-only (PCA, 8-d) — should produce clean nutritional/time clusters
   - content-only (SVD top-50, L2-normalized + standardized) — should produce cuisine/category clusters
   - combined (the v2 above) — should refine both

   If numeric-only also overlaps badly, the problem is deeper (e.g. clipping at p99.5 may be too aggressive — your skewness numbers post-`log1p` already look healthy, so this is unlikely the culprit).
4. **Consider replacing K-Means** for the combined matrix with **GaussianMixture** (handles non-spherical clusters) or **HDBSCAN** on a 100k sample (handles non-globular density and natively rejects noise). Food data is rarely composed of equal-variance spherical clusters.
5. **Always evaluate visually with UMAP** on a 30k subsample of the *clustering matrix you actually feed to K-Means*, not on the raw SVD. If UMAP doesn't show structure, no flat-clustering algorithm will find it.

---

## 6. Bottom Line

Your Week 5 pipeline is solid — the leakage discipline, time-derivation logic, sparse vs dense separation, and PCA/SVD split are all defensible.

Your Week 7 script then commits the canonical mistake of **concatenating two embeddings with incompatible metric assumptions**, leaves a 200-d anisotropic block dominated by 2–3 directions, and drowns the numeric signal under an unintended weight imbalance. K-Means is not failing because the data lacks structure; it is failing because the geometry you handed it has no usable structure for Euclidean partitioning.

Fix the per-block standardization, L2-normalize SVD rows, truncate the SVD tail, equalize block energies, add a joint PCA, and the same Week 5 artifacts will start producing clearly separated, interpretable clusters.
