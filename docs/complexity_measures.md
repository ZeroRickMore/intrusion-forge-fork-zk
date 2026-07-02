# Complexity Measures

All measures are computed by `compute_all_complexity_measures` under a **neutral
schema** (no `_class_` / `_cluster_` suffixes). The same function powers two
parallel analyses driven by `pipelines/compute_complexity.py`, distinguished
only by which partition label is passed as `y_cluster`:

| Analysis      | Partition  | Output                          |
|---------------|------------|---------------------------------|
| cluster-level | clusters   | `shared/complexity.json`        |
| class-level   | classes    | `shared/class_complexity.json`  |

Each output has an independent skip-marker, so the two analyses can be
regenerated separately by deleting the relevant file (or via
`complexity.force=true` to recompute both).

The pairwise families (F, N, ND) are aggregated against a **single scope**:
each partition (cluster or class) versus its **top-K nearest adversarial
partitions** — i.e., the top-K nearest partitions of a *different class* ranked
by centroid distance under `complexity.distance`.

For every metric the final value is the **min** (worst case), **mean**, and
**max** (best case) over the top-K populations.

The k-NN graph is built once per run using a **Gower-hybrid distance** (batched,
mixed numerical + categorical). The numerical contribution is configurable via
`complexity.distance`:

- `cosine` (default): cosine distance on RobustScaled numerical features
  (computed via L2-normalisation + Euclidean), combined with Hamming-style
  indicators on categoricals.
- `euclidean`: per-feature range-normalised Manhattan on RobustScaled
  numerical features, combined with Hamming-style indicators on categoricals.

Both modes normalise every per-feature contribution to $[0, 1]$, so the overall
distance stays in $[0, 1]$. Defaults: `complexity.k = 30`,
`complexity.top_k_clusters = 10`. Both are configurable via
[../configs/config.yaml](../configs/config.yaml).

The failure classifier (Step 3b) consumes **both** files: per cluster it
joins the cluster-level row and the class-level row of the cluster's class,
prefixing feature names as `cluster_*` and `class_*`.

---

## Family F — Feature-based

Measures how well individual features separate a partition from its top-K
adversarial partitions. Computed on RobustScaled numerical features only
(`X_num`). Categorical information is captured by the N/ND families.

### `f1_{min,mean,max}`

**Fisher discriminant ratio**

$$f_1(c, j) = \frac{1}{1 + \max_f \frac{(\mu_{c,f} - \mu_{j,f})^2}{\sigma^2_{c,f} + \sigma^2_{j,f}}}$$

| | |
|---|---|
| **Complexity** | $O(d)$ per pair |
| **Range** | $(0, 1]$ |
| **→ 0** | Large Fisher ratio on at least one feature → easy to separate |
| **→ 1** | All features have zero discriminative power → very hard to separate |

---

### `f2_{min,mean,max}`

**Bounding-box overlap ratio**

Mean fraction of per-feature range shared between partition $c$ and adversarial
population $j$.

$$f_2(c, j) = \frac{1}{d} \sum_f \frac{\max(0,\, \min(c_{\max,f}, j_{\max,f}) - \max(c_{\min,f}, j_{\min,f}))}{\max(c_{\max,f}, j_{\max,f}) - \min(c_{\min,f}, j_{\min,f})}$$

| | |
|---|---|
| **Complexity** | $O(nd)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | No feature-space overlap → easy to separate |
| **→ 1** | Complete overlap on every feature → very hard to separate |

---

### `f3_{min,mean,max}`

**Best single-feature separability (fraction in overlap region)**

For each feature, fraction of partition-$c$ samples that fall in the overlap
region with the adversarial population. Takes the **minimum over features**
(best discriminating feature).

$$f_3(c, j) = \min_f \frac{|\{x \in c : lo_f \le x_f \le hi_f\}|}{|c|}$$

where $lo_f = \max(c_{\min,f},\, j_{\min,f})$ and $hi_f = \min(c_{\max,f},\, j_{\max,f})$.

| | |
|---|---|
| **Complexity** | $O(nd)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | At least one feature fully separates partition $c$ from $j$ |
| **→ 1** | Every feature has all partition-$c$ samples inside the overlap region |

---

### `f4_{min,mean,max}`

**Joint-feature overlap fraction**

Fraction of partition-$c$ samples that fall inside the overlap region on **all
features simultaneously**.

$$f_4(c, j) = \frac{|\{x \in c : \forall f,\; lo_f \le x_f \le hi_f\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | $O(nd)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | No sample simultaneously overlaps on all features → clean joint separation |
| **→ 1** | All samples are jointly inside the overlap region on every feature |

---

## Family N — Neighborhood-based

Measures local overlap in the k-NN graph (Gower distance). Noise points
($y_{\text{partition}} = -1$) are excluded from partition membership but may
appear as neighbours.

### `n1_{min,mean,max}`

**MST boundary fraction**

Fraction of partition-$c$ samples that have at least one edge in the
approximate minimum spanning tree connecting them to a sample of the
adversarial population. The MST is built on the symmetrised k-NN graph and a
bridge edge is added per disconnected component to guarantee connectivity.

> **Approximation.** The MST is computed on the *sparse* k-NN graph, not on
> the full pairwise distance matrix, so it is an approximation of the true
> MST. When categorical features partition the space, the k-NN graph can be
> disconnected; connectivity is then restored by adding one bridge edge per
> component towards a reference node. Bridge edges are not MST-optimal and
> can introduce a small number of artificial adjacencies, so $n_1$ should be
> read as a boundary indicator on the neighbourhood graph rather than an
> exact MST statistic.

$$n_1(c, j) = \frac{|\{x \in c : \exists\, (x, y) \in \text{MST},\; y \in j\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | $O(n \log n)$ for MST; $O(|E|)$ vectorised boundary check |
| **Range** | $[0, 1]$ |
| **→ 0** | Partition $c$ is entirely interior, never adjacent to $j$ in the MST |
| **→ 1** | Every sample of partition $c$ borders $j$ in the MST |

---

### `n2_{min,mean,max}`

**Intra/inter nearest-neighbour distance ratio**

For each sample $x \in c$, ratio of nearest same-partition distance to the sum
of intra and inter distances:

$$n_2(x) = \frac{d_{\text{intra}}}{d_{\text{intra}} + d_{\text{inter}}}$$

Averaged over all samples in $c$ that have both kinds of neighbour within
the k-NN.

| | |
|---|---|
| **Complexity** | $O(|c| \cdot k)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | Intra-partition neighbours are far closer than $j$ neighbours → well-separated |
| **→ 0.5** | Intra and inter distances are equal → no local separation |
| **→ 1** | $j$ neighbours are much closer than same-partition neighbours → severe overlap |

---

### `n3_{min,mean,max}`

**1-NN error rate restricted to c ∪ j**

For each sample $x \in c$, find its nearest neighbour in the k-NN list that
belongs to $c \cup j$. Misclassified if that neighbour is in $j$.

$$n_3(c, j) = \frac{|\{x \in c : \mathrm{NN}_{c \cup j}(x) \in j\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | $O(|c| \cdot k)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | Nearest neighbour is always from the same partition → clean local boundary |
| **→ 1** | Nearest neighbour is always from $j$ → partition $c$ is embedded inside $j$ |

> Practical note: this metric is meaningful only if the k-NN list has enough
> reach to contain at least one $c \cup j$ neighbour for most samples. With
> the default `k = 30` this is true for the vast majority of samples on
> NB15/CIC-IDS-2018.

---

### `n4_{min,mean,max}`

**k-NN majority-vote error rate**

For each sample $x \in c$, count votes from same-partition vs $j$ neighbours.
Misclassified if $j$-votes > $c$-votes.

$$n_4(c, j) = \frac{|\{x \in c : \text{votes}_j(x) > \text{votes}_c(x)\}|}{|c|}$$

| | |
|---|---|
| **Complexity** | $O(|c| \cdot k)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | Majority of neighbours always from same partition |
| **→ 1** | Majority of neighbours always from $j$ |

---

## Family ND — Network density

### `network_density_{min,mean,max}`

**Cross-population k-NN density**

Fraction of k-NN edges from partition $c$ pointing to samples of the
adversarial population:

$$\text{density}(c, j) = \frac{\sum_{x \in c} |\{nb \in NN(x) : nb \in j\}|}{|c| \times k}$$

| | |
|---|---|
| **Complexity** | $O(|c| \cdot k)$ per pair |
| **Range** | $[0, 1]$ |
| **→ 0** | No k-NN edges cross to $j$ → isolated partition |
| **→ 1** | All k-NN edges from partition $c$ point to $j$ → fully intermixed |

---

### `cls_coef`

**Local clustering coefficient** within the partition (intra-partition
neighbour pairs that are themselves neighbours, averaged across partition
members).

### `hub`

**Mean in-degree** in the reverse k-NN graph (hubness proxy).

---

## Family T — Dimensionality

Per-partition dimensionality measures. Computed on the subset of samples
belonging to each partition.

### `t2`

**Feature-to-sample ratio**

$$T_2 = \frac{d_{\text{num}} + d_{\text{cat}}}{n_c}$$

### `t3`

**PCA components-to-sample ratio** — intrinsic dimensionality (PCA components
for 95% variance) relative to the partition size, per the Lorena et al. (2019)
canon.

$$T_3 = \frac{n_{\text{PCA}_{95\%}}}{n_c}$$

### `t4`

**PCA intrinsic dimensionality ratio** — fraction of numerical features
needed to explain 95% of variance within the partition.

$$T_4 = \frac{n_{\text{PCA}_{95\%}}}{d_{\text{num}}}$$

T-family is computed on numerical features only by design (categorical
information is captured by the N/ND families).

---

## Family G — Geometry

Per-partition geometric properties computed under the configured metric on
`X_num`. G-family is numerical-only by design.

> **Approximation.** Silhouette-based quantities (`p5_silhouette`,
> `frac_at_risk`) are estimated on a stratified subsample of at most 10 000
> points to bound the $O(n^2)$ pairwise cost. Samples outside the subsample
> have no silhouette value, so for partitions much larger than the cap the
> reported percentiles are estimated from a fraction of the members and
> should be treated as approximate.

### `max_dispersion`

Maximum distance from any sample in the partition to its centroid.

### `p95_dispersion`

95th percentile of the sample → centroid distances. Outlier-robust
counterpart to `max_dispersion`.

### `dist_to_nearest_centroid`

Minimum centroid-to-centroid distance from partition $c$ to any other
partition (under `complexity.distance`).

### `p5_silhouette`

5th percentile of silhouette scores (computed on non-noise points using
partition labels under the configured metric).

### `frac_at_risk`

Fraction of partition-$c$ samples with silhouette score $< 0$.
