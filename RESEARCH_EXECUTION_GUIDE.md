# RESEARCH EXECUTION GUIDE
## Practical Checklist for PhD Thesis Implementation

---

## 📋 PHASE 1-2: COMPLETED (Theory + Synthetic)

### ✅ Weeks 1-6 Deliverables (DONE)

- [x] **Week 1**: Topology foundations notebook
  - Metric spaces, compactness, ball neighborhoods
  - File: `phase1_topology/week1_topology_foundations.ipynb`

- [x] **Week 4**: GNN expressivity theory
  - Weisfeiler-Lehman iteration, bounds, receptive field
  - File: `phase2_approximation/week4_gnn_universal_approximation.ipynb`

- [x] **Week 5**: Synthetic experiments
  - Barabási-Albert graph, 5-hop dependency
  - GNN depths 1-6 on synthetic income task
  - File: `phase2_approximation/week5_synthetic_gnn_experiments.ipynb`

- [x] **Week 6**: Real Chicago experiments
  - Load 800 Census tracts
  - Train GNNs at all depths
  - Measure R² on income prediction
  - File: `phase2_approximation/week6_chicago_experiments.ipynb`

---

## 🔬 PHASE 3: IN PROGRESS (Weeks 7-9)

### Week 7: Decision Boundaries & Interpretability

**Goal:** Answer "WHAT does the network learn?"

**Checkpoints:**

- [ ] **Load trained models** (from Week 6)
  ```python
  # In new notebook: week7_decision_boundaries.ipynb
  import torch
  model_depth6 = load_model('models/chicago_gnn_depth6.pt')
  model_depth1 = load_model('models/chicago_gnn_depth1.pt')
  ```

- [ ] **Extract learned representations** (hidden layer outputs)
  ```python
  # Forward pass without output layer
  # Get 64-dim feature vectors for each tract
  reps_deep = model_depth6.convs[-1](x, edge_index)   # Before final Linear
  reps_shallow = model_depth1.convs[-1](x, edge_index)
  
  # Expected: deep repr clusters by income
  # Expected: shallow repr is less structured
  ```

- [ ] **Visualize with UMAP**
  ```python
  from umap import UMAP
  
  # UMAP projection to 2D
  mapper = UMAP(n_neighbors=15, metric='euclidean')
  reps_2d_deep = mapper.fit_transform(reps_deep.detach().numpy())
  
  # Plot: color by income
  plt.scatter(reps_2d_deep[:, 0], reps_2d_deep[:, 1], 
              c=y, cmap='viridis', s=50)
  # Expect: deep network shows income gradient
  # Expect: shallow network shows random scatter
  ```

- [ ] **Feature Importance (Integrated Gradients)**
  ```python
  # Which input features matter most?
  # Compute ∂R²/∂x_i for each feature
  
  def feature_importance(model, X, y, feature_idx):
      X.requires_grad_(True)
      out = model(X, edge_index)
      loss = mse(out, y)
      loss.backward()
      importance = X.grad.abs().mean(dim=0)
      return importance[feature_idx]
  
  # Compare: importance of transit features vs. OSM vs. Census
  # Expected: deep network values transit > shallow
  ```

- [ ] **Residual Analysis**
  ```python
  # Which tracts are hard to predict?
  predictions = model_depth6(X, edge_index)
  residuals = y - predictions.squeeze()
  
  # Analysis: do errors correlate with spatial isolation?
  # Plot: residual vs. nearest_stop_distance
  plt.scatter(df['nearest_stop_dist_km'], np.abs(residuals))
  # Expected: isolated tracts have larger errors
  # → Validates: network is learning transit importance
  ```

- [ ] **Statistical Test: R² by Depth**
  ```python
  # Paired t-test
  from scipy.stats import ttest_rel
  
  r2_depth1 = [0.42, 0.41, 0.43, ...]  # 10 runs
  r2_depth6 = [0.58, 0.59, 0.57, ...]
  
  t_stat, p_value = ttest_rel(r2_depth6, r2_depth1)
  print(f"t-stat = {t_stat:.2f}, p-value = {p_value:.6f}")
  # Expected: p < 0.001 (strongly significant)
  ```

**Output Files:**
- `figures/week7_umap_deep_vs_shallow.png` — Representation visualization
- `figures/week7_feature_importance.png` — Importance rankings
- `figures/week7_residuals_by_distance.png` — Residual analysis
- `data/audit/week7_analysis.csv` — Summary statistics

**Success Criterion:**
- [ ] Deep network UMAP shows clear income gradient
- [ ] Transit features rank top 5 in importance
- [ ] Residuals correlate with distance to transit (r > 0.3)

---

### Week 8: Topological Features (Optional, Advanced)

**Goal:** Quantify what makes deep representations different

**Checkpoints:**

- [ ] **Persistent Homology** (Optional; requires Ripser library)
  ```python
  from ripser import ripser
  
  # Compute topological features of deep representation
  result_deep = ripser(reps_deep)
  # H0: connected components, H1: loops, H2: voids
  
  # Compare to shallow representation
  result_shallow = ripser(reps_shallow)
  
  # Expected: deep representation has more complex topology
  ```

- [ ] **Representation Clustering**
  ```python
  from sklearn.cluster import KMeans
  
  # Do representations cluster by neighborhood type?
  kmeans = KMeans(n_clusters=5)
  clusters_deep = kmeans.fit_predict(reps_deep)
  clusters_shallow = kmeans.fit_predict(reps_shallow)
  
  # Measure: silhouette score, Davies-Bouldin index
  # Expected: deep network clusters more coherently
  ```

**Output Files:**
- `figures/week8_persistent_homology.png` — Barcodes
- `figures/week8_clustering.png` — Cluster assignments on map

**Success Criterion:**
- [ ] Deep representation has higher silhouette score (>0.4)
- [ ] Topological features differ significantly

---

### Week 9: Equity & Fairness Analysis

**Goal:** Ensure model is unbiased across demographic groups

**Checkpoints:**

- [ ] **Accuracy by Income Quartile**
  ```python
  # Split into 4 income groups
  q1 = df[df['median_household_income'] <= df['median_household_income'].quantile(0.25)]
  q2 = df[(df['median_household_income'] > 0.25) & ...]
  q3 = ...
  q4 = df[df['median_household_income'] > df['median_household_income'].quantile(0.75)]
  
  # Measure R² for each group
  r2_q1 = r2_score(q1['y_true'], q1['y_pred'])
  r2_q2 = r2_score(q2['y_true'], q2['y_pred'])
  r2_q3 = r2_score(q3['y_true'], q3['y_pred'])
  r2_q4 = r2_score(q4['y_true'], q4['y_pred'])
  
  disparity = max(r2_q1, r2_q2, r2_q3, r2_q4) - min(...)
  print(f"R² disparity across quartiles: {disparity:.4f}")
  # Expected: disparity < 0.05 (fair)
  # Concern: if disparity > 0.10 (unfair to one group)
  ```

- [ ] **Error Analysis by Neighborhood Type**
  ```python
  # Categorize tracts: downtown/residential/suburban/industrial
  df['neighborhood_type'] = categorize(df['geometry'], df['features'])
  
  # Measure: MAE by type
  for ntype in ['downtown', 'residential', 'suburban', 'industrial']:
      subset = df[df['neighborhood_type'] == ntype]
      mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
      print(f"{ntype}: MAE = ${mae:,.0f}")
  # Expected: MAE roughly uniform across types
  # Concern: if industrial/sparse areas have 2× higher error
  ```

- [ ] **Disparate Impact Test**
  ```python
  # Does model systematically over/under-predict certain groups?
  # Compute: avg residual by spatial area
  
  residuals_by_area = df.groupby('area').apply(
      lambda x: (x['y_pred'] - x['y_true']).mean()
  )
  
  # Test: are residuals zero-mean for each area?
  # H0: mean residual = 0 for all areas
  # Use Kruskal-Wallis test (nonparametric ANOVA)
  ```

**Output Files:**
- `figures/week9_fairness_by_quartile.png` — R² by income group
- `figures/week9_fairness_by_neighborhood.png` — Error by area type
- `data/audit/week9_fairness_metrics.csv` — Summary statistics

**Success Criterion:**
- [ ] R² disparity < 0.05 across income quartiles (PASS)
- [ ] MAE disparity < 20% across neighborhood types (PASS)
- [ ] Kruskal-Wallis p-value > 0.05 (residuals unbiased)

---

## 📜 PHASE 4: FINAL (Weeks 10-12)

### Week 10: Lower Bounds Proof

**Goal:** Formalize why shallow networks fail

**Task:** Write Theorem + Proof (LaTeX in `proofs/` folder)

```latex
\begin{theorem}[Depth Necessity for Multi-Scale Functions]
  Let G = (V, E) be a connected graph with diameter d.
  Let f: V → ℝ be a target function that depends on 
  all k-hop neighborhoods for some k ≤ d.
  
  Any k-1-layer GNN with finite hidden dimension h 
  cannot represent f exactly, regardless of h.
  
  Proof: By Weisfeiler-Lehman expressivity bounds...
\end{theorem}
```

**Deliverable:** `proofs/theorem_depth_necessity.pdf`

---

### Week 11: Depth-Width Tradeoff

**Goal:** Can we trade depth for width?

**Experiment:**
```python
# Train networks: (depth, width) pairs
configs = [
  (1, 256),  # Shallow + wide
  (2, 128),
  (3, 64),
  (4, 32),
  (6, 16),   # Deep + narrow
]

results = {}
for depth, width in configs:
    model = GNNRegressor(in_features=2056, hidden_dim=width, depth=depth)
    result = train_gnn(...)
    results[(depth, width)] = result['r2_normalized']

# Plot: heat map of R² by (depth, width)
```

**Expected finding:**
- Deep + narrow > Shallow + wide
- → Proves: cannot compensate for lack of depth with width

**Deliverable:** `phase2_approximation/week11_depth_width_tradeoff.ipynb`

---

### Week 12: Final Paper

**Structure:**

```
1. Abstract (150 words)
   "We prove that graph neural network depth is necessary 
   for learning multi-scale geographic patterns..."

2. Introduction (2 pages)
   - Problem: GNNs work, but why?
   - Gap: Weisfeiler-Lehman theory doesn't explain regression
   - Question: Can we prove depth is necessary?

3. Related Work (1.5 pages)
   - Graph neural networks (Kipf & Welling, Hamilton et al.)
   - Expressivity bounds (Morris et al., Maron et al.)
   - Spatial regression (Anselin, spatial econometrics)

4. Methods (2 pages)
   - Data selection & privacy
   - GNN architecture (GraphConv layers)
   - Evaluation protocol (train/val/test splits)

5. Results (3 pages)
   - Synthetic task: depth 1-4 fail, 5-6 succeed
   - Chicago data: R² improves with depth (p < 0.001)
   - Feature importance: transit > OSM > Census
   - Fairness: unbiased across groups

6. Discussion (2 pages)
   - Findings support Weisfeiler-Lehman theory
   - Practical implications (depth ≈ 2× spatial range needed)
   - Limitations (5-year estimates, Street View bias)

7. Conclusion (0.5 pages)
   - Depth matters for geographic learning
   - Practitioners should use depth proportional to spatial scale
   - Future: optimize depth for efficiency

8. References (2 pages)
   - ~50 citations
```

**Deliverable:** `writeup/thesis_draft.pdf`

---

## 📊 Statistical Summary Template

**Use this for every result:**

```
Finding: [What we discovered]

Data: [How many samples, what dataset]
  - Sample size: n = 800 tracts
  - Train/val/test: 480/160/160
  - Outcome variable: median_household_income

Method: [How we measured it]
  - Model: GNN with depth 6
  - Metric: R² (coefficient of determination)
  - Significance test: Paired t-test, α = 0.05

Result: [Numbers with confidence intervals]
  - R² = 0.58 [0.55, 0.61] (95% CI)
  - p-value = 0.0001 ***
  - Cohen's d = 1.2 (large effect)

Interpretation: [What it means]
  - We are 95% confident that true R² is 
    between 0.55 and 0.61
  - The effect (depth matters) is substantial
  - Result is highly statistically significant

Robustness: [How stable is this?]
  - Holds across 5 different random seeds
  - Holds across 3 different train/val/test splits
  - Holds with different feature normalizations
```

---

## ✅ Go/No-Go Decision Gates

### Gate 1 (End Week 6): Empirical Depth Effect
**PASS if:**
- [ ] R²(depth=6) - R²(depth=1) ≥ 0.10
- [ ] p-value < 0.05 (paired t-test)
- [ ] Effect size (Cohen's d) > 0.5

**If FAIL:**
- Review: Is graph structure weak? (fix: use stronger adjacency)
- Review: Are features informative? (fix: add transit features)
- Pivot: Reframe as "what's learnable with geographic structure?"

---

### Gate 2 (End Week 7): Interpretability
**PASS if:**
- [ ] Deep UMAP shows income gradient (visual inspection)
- [ ] Transit features rank in top 5 importance
- [ ] Residuals correlate with isolation (r > 0.3)

**If FAIL:**
- Review: Are learned representations meaningful? (check activation distributions)
- Pivot: Focus on representation quality, not just R² improvement

---

### Gate 3 (End Week 9): Fairness
**PASS if:**
- [ ] R² disparity < 0.05 across income quartiles
- [ ] No systematic bias in residuals (p > 0.05 Kruskal-Wallis)
- [ ] Error uniform across neighborhood types

**If FAIL:**
- Investigate: Is bias due to sparse data? (exclude sparse tracts)
- Investigate: Is bias due to model complexity? (simplify, regularize)
- Mitigate: Use class weights, stratified sampling

---

### Gate 4 (End Week 12): Publication Readiness
**PASS if:**
- [ ] Theorem statement is clear and novel
- [ ] All claims backed by data + statistics
- [ ] Paper is > 10 pages, < 20 pages (conference format)
- [ ] No major ethical concerns

**If FAIL:**
- Too incremental? → Emphasize novelty of theory
- Claims not supported? → Run additional experiments
- Ethics unclear? → Document all privacy measures

---

## 🛠️ Troubleshooting Guide

### Problem: R² is too low (< 0.3)
**Possible causes:**
1. Graph is too sparse (tracts are independent)
   - **Fix:** Use k-NN with k=10 instead of k=4
2. Features are noisy
   - **Fix:** Add more Street View images per tract
3. Income is driven by non-spatial factors (education, industry)
   - **Fix:** Include industry/job type data if available

### Problem: Results vary widely across seeds
**Possible causes:**
1. Training is unstable (high learning rate)
   - **Fix:** Reduce LR from 0.01 to 0.001
2. Validation set is too small (160 samples)
   - **Fix:** Use 5-fold cross-validation instead
3. Early stopping isn't working
   - **Fix:** Increase patience from 30 to 50 epochs

### Problem: Fairness disparity is high (> 0.10 R² difference)
**Possible causes:**
1. Sparse tracts are included (low-income areas have fewer images)
   - **Fix:** Exclude tracts with < 10 Street View images
2. Model is overfitting to wealthy neighborhoods
   - **Fix:** Use class weights: weight_low = 1.5, weight_high = 0.8
3. Features are biased (OSM is richer in wealthy areas)
   - **Fix:** Include `pct_missing_height` as a feature; model learns bias

---

## 📝 Reproducibility Checklist

Before finalizing any results:

- [ ] All random seeds fixed (numpy, torch, sklearn)
- [ ] Data split is deterministic (fixed split indices)
- [ ] Hyperparameters logged to `config.yaml`
- [ ] Results saved to `data/audit/` CSV
- [ ] Figures saved in high-DPI (300+ dpi)
- [ ] Notebook outputs cleared before commit
- [ ] Git log shows all changes

---

## 🎓 Writing for Your Committee

**How to present findings:**

**WEAK:** "Deep networks do better."
**STRONG:** "GNNs with depth 6 achieve R²=0.58 [0.55,0.61] on predicting Chicago median household income, compared to depth 1 at R²=0.42 [0.40,0.44]. This 0.16 difference is highly significant (paired t-test p=0.0001, Cohen's d=1.2), validating Weisfeiler-Lehman expressivity theory: k-layer networks can only propagate information k hops, thus cannot learn functions depending on >k-hop neighborhoods."

**Key elements:**
- Precise numbers with uncertainty (confidence intervals)
- Sample size and data source
- Statistical significance + effect size
- Connection to theory
- Practical interpretation

---

## 📚 References for Your Thesis

**Key papers to cite:**

1. **Weisfeiler-Lehman Expressivity:**
   - Morris et al. "Weisfeiler and Leman Go Neural" (2019)
   - Maron et al. "Invariant and Equivariant Graph Networks" (2019)

2. **GNN Architecture:**
   - Kipf & Welling "Semi-Supervised Classification with GCNs" (2016)
   - Hamilton et al. "GraphSAGE" (2017)

3. **Spatial Regression:**
   - Anselin "Spatial Econometrics: Methods and Models" (1988)
   - Fotheringham & Rogerson "Handbook of Spatial Analysis" (2014)

4. **Urban Geography:**
   - Florida "The Rise of the Creative Class" (2002)
   - Cervero & Kockelman "Travel Demand and the 3Ds" (1997)

---

## 🚀 Next Steps (After Phase 4)

1. **Conference Submission** (ICML, NeurIPS, ICLR)
   - Compress to 8-page paper format
   - Focus on: theorem + synthetic validation

2. **Journal Submission** (TPAMI, IEEE TNNLS)
   - Expand to 20-30 page article
   - Include: full proofs, geographic analysis, fairness

3. **Software Release**
   - Package code as: `gnn-expressivity` Python library
   - Add tutorials, examples, CLI tools
   - Host on GitHub, PyPI

4. **Engagement**
   - Present at urban planning conferences
   - Collaborate with geographers on real-world applications
   - Explore: generalization to other cities

---

**Last Updated:** 2026-04-10  
**Status:** EXECUTION READY
