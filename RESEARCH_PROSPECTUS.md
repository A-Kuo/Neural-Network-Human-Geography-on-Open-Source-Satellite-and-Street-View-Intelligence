# RESEARCH PROSPECTUS
## Topological Foundations of Neural Network Expressivity in Urban Geography

**Version:** 1.0 (Research Design Document)  
**Status:** Ready for Committee Review

---

## PART I: PROBLEM STATEMENT

### 1.1 Core Research Question

**Primary Question:**
> *Can we prove that graph neural network depth is fundamentally necessary for learning multi-scale spatial patterns in urban geography, and that shallow networks face insurmountable topological barriers?*

**Specific, Testable Hypotheses:**

1. **H1 (Topological Necessity)**: A k-layer GNN cannot learn spatial functions that depend on information >k hops away in the tract graph, regardless of training data or model width.
   - **Grounded in:** Weisfeiler-Lehman expressivity theory
   - **Testable by:** Synthetic task where income depends on 5-hop neighborhoods

2. **H2 (Multi-Scale Geography)**: Chicago neighborhood income is fundamentally a multi-scale phenomenon:
   - Local (1-2 hops): building density, street type
   - District (3-4 hops): transit cluster membership
   - City (5+ hops): proximity to economic core (Loop)
   - **Grounded in:** Urban geography literature (transit-oriented development)
   - **Testable by:** Real Chicago data with spatial graph

3. **H3 (Depth-Performance Tradeoff)**: Deeper networks achieve lower test error on income prediction than shallow networks, with statistically significant difference (p < 0.05).
   - **Grounded in:** Information theory (larger receptive field = more information)
   - **Testable by:** Hold-out test set, paired t-test across depths

### 1.2 Why This Question Matters

**Academic Significance:**
- **GNNs are black-box**: We know depth helps empirically, but WHY?
- **Theory-practice gap**: Weisfeiler-Lehman theory says k-layer GNN can distinguish k-WL-nonequivalent graphs, but what does this mean for regression on spatial data?
- **Geographic ML is under-theorized**: Most papers treat space as generic features, not as a topological structure

**Practical Significance:**
- If depth is necessary, practitioners should stop trying to fit performance with shallow wide networks
- If we can characterize the required depth, we can design efficient models
- Privacy angle: can we aggregate to larger regions (lower spatial resolution) and still get good predictions?

### 1.3 What We're NOT Asking

❌ "Do GNNs work better than MLPs on geographic data?" (Too broad, already known)  
❌ "Can we predict income?" (Possible with simple linear models)  
❌ "What is the best hyperparameter?" (Uninteresting, domain-specific)  

✅ "Why does depth matter, and can we prove it?" (This is the theoretical contribution)

---

## PART II: DATA SELECTION

### 2.1 Geographic Scope: Why Chicago?

**Selection Criteria:**
1. **Data availability** ✓
   - Google Street View: 90%+ coverage
   - OpenStreetMap: 99% building coverage
   - CTA/Metra: 100% route data (public GTFS)
   - Census: 100% tract data (official government source)

2. **Tractability** ✓
   - ~800 Census tracts (manageable for GNN, large enough for statistics)
   - Single urban area (no state boundary complications)
   - English language (simplifies Street View analysis)

3. **Spatial structure diversity** ✓
   - Downtown (Loop): dense, expensive, high transit
   - North Shore suburbs: wealthy, sprawling, car-dependent
   - South/West Side: lower income, mixed transit access
   - Industrial zones: sparse data (intentionally excluded)
   - This diversity validates whether our model works across contexts

4. **Temporal stability** ✓
   - Data from 2022 ACS (recent, stable)
   - CTA/Metra schedules current
   - Street View coverage consistent
   - No major infrastructure changes 2020-2022

**Why NOT other cities?**
- NYC: Too large (2000+ tracts), too expensive (Street View quota)
- Los Angeles: Car-dependent sprawl (transit signal weak)
- San Francisco: Steep terrain (causes projection artifacts)
- Austin: Rapid change (data quality varies)

### 2.2 Data Sources & Justification

#### Source 1: US Census Bureau (Outcome Variable)

**What:** American Community Survey (ACS) 5-year estimates, 2018-2022  
**Why this source:**
- Gold standard for neighborhood economics
- Public domain (no licensing issues)
- Tract-level aggregation (4,000+ people per tract, preserves privacy)
- Margins of error reported (enables confidence intervals)

**Variables Selected:**
- **Primary outcome**: `median_household_income` (B19013_001E)
  - Why median not mean? Robust to outliers, standard in literature
  - Range: ~$35k–$200k (realistic for Chicago)
  - Why not individual income? Privacy + reduces to tract aggregate anyway

- **Confounders** (to validate model learns right patterns):
  - `total_population` (B01003_001E) — larger tracts more stable income
  - `poverty_rate` (ratio of B17001_002E / B17001_001E) — structural poverty
  - `pct_college_plus` (ratio of B15003_{022,023,025}E / B15003_001E) — education proxy

**Limitations acknowledged:**
- 5-year estimates have ±15% margin of error (see ETHICS.md)
- No individual data (aggregated by design)
- No race/ethnicity (prevents redlining, added separately if needed)

#### Source 2: OpenStreetMap Buildings (Proxy for Local Development)

**What:** Building footprints + height tags from Overpass API  
**Why this source:**
- Captures FORM (density, size, type) which affects neighborhood character
- OSM is free + open-source (no API keys, reproducible)
- Height tag → building age proxy (older buildings often missing)

**Variables Extracted:**
- `building_density_km2` — buildings per km² (Scale 1: local)
- `pct_residential` — % of buildings residential vs. commercial
- `median_building_area_m2` — typical building footprint
- `pct_missing_height` — data quality / mapping effort bias (see below)

**Known bias:**
- Wealthy neighborhoods have more volunteer OSM editors
- Richer areas have more complete height tagging
- **How we handle it:** 
  - Include `pct_missing_height` as a feature (let model learn bias)
  - Document correlation: `pct_missing_height` ↔ `income` in audit (see data/audit/osm_coverage_audit.csv)
  - Exclude tracts with >80% missing height from training (if correlation strong)

#### Source 3: GTFS Transit Data (Multi-Scale Accessibility)

**What:** Official schedules from CTA (buses + L) and Metra (commuter rail)  
**Why this source:**
- Encodes city structure: transit hubs cluster development
- Deterministic (schedules are fixed rules, not observed behavior)
- Scale 2 (district): 500m-1km catchments
- Scale 3 (city): travel time to Downtown Loop

**Variables Extracted:**
- `nearest_stop_dist_km` — distance to closest transit stop
- `n_stops_500m` — stop density (walkability)
- `median_travel_time_loop_min` — fastest connection to Loop (financial core)
- `transit_score` — composite 0-100 score (40 pts proximity + 30 pts density + 30 pts travel time)

**Theoretical rationale:**
- Transit accessibility is THE spatial organizing principle in Chicago
- Predicts income *because* transit hubs attract jobs + wealthy professionals
- Creates spatial hierarchy: CBD → transit corridors → isolated neighborhoods

**Limitations:**
- Assumes transit schedules = real service quality (not always true)
- Doesn't capture bus frequency (CTA cuts disproportionately hurt poor areas)
- Documented in ETHICS.md as "transit coverage bias"

#### Source 4: Google Street View (Visual Features)

**What:** Street-level imagery (640×640 JPEGs) → ResNet-152 embeddings  
**Why this source:**
- Captures visual character (street infrastructure, building façade quality, trees)
- State-of-the-art computer vision models (ResNet-152, ImageNet pretrained)
- Covers ~90% of Chicago streets
- Extract only 2048-dim embeddings (no images stored — privacy)

**Variables Extracted:**
- 2048-dim ResNet feature vector per tract (aggregated from ~30 images per tract)
- Captures visual proxy for neighborhood quality, investment, maintenance

**Known bias:**
- Street View coverage lower in South/West Side (historical undersampling)
- Seasonal variation (some streets photographed in different seasons)
- Daytime only (9 AM–4 PM; misses nightlife, evening activity)
- **How we handle it:**
  - Flag tracts with <10 images as `sparse_imagery` (exclude from training)
  - Document: data/audit/streetview_coverage.csv (lists all sparse tracts)
  - Train on n_images ≥ 10 only (prevents extreme class imbalance)

### 2.3 Feature Engineering: Building the Graph

**Node Features (per Census tract):**
- 8 handcrafted features:
  - 1 OSM building feature (building_density_km2)
  - 3 transit features (nearest_stop_dist, n_stops_500m, median_travel_time_loop)
  - 2 Census features (total_population, pct_college_plus)
  - 2048 ResNet features (averaged across tract images)
  - **Total: 2056 features per node**

**Why these features?**
- **Redundancy by design**: We expect GNN to learn which features matter through graph propagation
- **Multi-scale representation**: Local (OSM), intermediate (transit), global (Census)
- **Robustness**: If one source fails, others compensate

**Edge Construction (Graph Topology):**

**Method 1: Spatial Adjacency** (If geometries available)
- Two tracts share an edge if they touch or intersect
- Creates "natural" geographic neighborhoods
- ~2-3 edges per node on average (sparse)

**Method 2: k-Nearest Neighbors** (Fallback + Supplement)
- Connect each tract to k=4 nearest neighbors by feature distance
- Captures functional similarity (tracts "like" each other)
- ~4 edges per node (denser than spatial)

**Combined:**
- Use both methods simultaneously (union of edges)
- Results in ~6-7 edges per node
- Undirected (if A→B, then B→A)

**Why union, not just spatial?**
- Spatial alone is too sparse (some suburban tracts barely touch)
- k-NN alone misses geographic constraints
- Together: captures both spatial + feature similarity

---

## PART III: MODEL SELECTION

### 3.1 Why Graph Neural Networks?

**Alternative 1: Ordinary MLP (Feedforward Network)**
- ❌ Ignores spatial structure entirely
- ❌ Treats each tract independently
- ✗ Violates geographic principle: nearby places are similar
- Baseline for comparison

**Alternative 2: Convolutional Neural Nets (CNNs)**
- ❌ Assumes grid structure (lat/lon grid)
- ❌ Chicago tracts don't form a regular grid
- ❌ Hard to incorporate non-spatial features (transit, Census)

**Alternative 3: Spatial Regression (e.g., Spatial Autoregressive Model)**
- ✓ Handles spatial correlation
- ❌ Linear (cannot learn nonlinear patterns)
- ❌ No proven way to scale to multi-scale spatial structure
- Baseline for comparison

**Selected: Graph Neural Networks (GNNs)** ✓
- ✓ Handles irregular graphs (tract adjacency)
- ✓ Propagates information k hops (captures multi-scale)
- ✓ Nonlinear (can learn complex patterns)
- ✓ Theoretically analyzable (Weisfeiler-Lehman expressivity)
- ✓ Scalable to 800 nodes

### 3.2 GNN Architecture: GraphConv Layers

**Layer Type: GraphConv (Graph Convolution)**

Mathematical form:
```
x_i^{(k+1)} = W^{(k)} * (x_i^{(k)} + Σ_{j ∈ N(i)} x_j^{(k)})
```

Where:
- `x_i^{(k)}` = feature vector for node i at layer k
- `W^{(k)}` = learnable weight matrix (shared across all nodes)
- `N(i)` = neighbors of node i
- Σ = sum aggregation

**Why GraphConv?**
- ✓ Simple + well-understood (Kipf & Welling 2016)
- ✓ Proven to work on geographic tasks
- ✓ Each layer increases receptive field by 1 hop
- ✓ Depth control is transparent (1 layer = 1 hop, 6 layers = 6 hops)

**Why NOT other GNN layers?**
- GAT (Graph Attention): Adds complexity; for this task, simple aggregation sufficient
- GraphSAGE: Sampling-based; not needed for 800-node graphs
- GIN (Graph Isomorphism): More expressive but overkill; harder to interpret

### 3.3 Network Architecture (Detailed)

```
Input: x ∈ ℝ^{n × 2056}  (n=800 tracts, 2056 features)

Layer 1: GraphConv(2056 → 64)
  - Aggregates neighbor features, applies 2056×64 weight matrix
  - Output: ℝ^{800 × 64}
  - ReLU activation
  - Dropout(0.3)

Layers 2-k: GraphConv(64 → 64) [repeated k-1 times]
  - Same operation, maintains 64 channels
  - Output: ℝ^{800 × 64}
  - ReLU activation
  - Dropout(0.3)

Output Head: Linear(64 → 1)
  - Single output: predicted income per tract
  - No activation (regression)
  - Output: ℝ^{800 × 1}

Loss: MSE(predictions, true_income)
```

**Why 64 hidden dimensions?**
- Standard choice (not too big, not too small)
- Ablation planned for Phase 4 (could try 32, 128)
- Proportional to graph size: 64 ≈ 800^0.25 (common heuristic)

**Why ReLU + Dropout?**
- ReLU: Standard activation, prevents vanishing gradients
- Dropout(0.3): Regularization, prevents overfitting on 800 samples
- No activation on output: Regression task (can predict negative, but won't)

---

## PART IV: PARAMETER SELECTION & JUSTIFICATION

### 4.1 Training Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Optimizer** | Adam | Standard for deep learning; handles varying gradient scales |
| **Learning Rate** | 0.01 | Medium: fast enough to converge, stable enough to avoid oscillation |
| **Weight Decay (L2)** | 1e-4 | Mild regularization (prevents extreme weights) |
| **Batch Size** | Full batch (800) | GNN on small graphs uses full-batch; larger batches = more stable |
| **Epochs** | 200 | Enough for convergence; early stopping prevents overfitting |
| **Early Stopping** | Patience=30 | Stop if val loss doesn't improve for 30 epochs |
| **Dropout** | 0.3 | Moderate (0.2-0.5 typical); prevents co-adaptation |

### 4.2 Data Split

**Train / Val / Test: 60 / 20 / 20**
- Train: 480 tracts (learn patterns)
- Val: 160 tracts (tune hyperparameters, early stopping)
- Test: 160 tracts (final evaluation, held-out)

**Why this split?**
- 60/20/20 is standard for medium-sized datasets
- Not 80/10/10 (too much training data, small test set unreliable)
- Stratified by income quantiles (ensure all income ranges in each split)
- Fixed seed (42) for reproducibility

### 4.3 Depth Specification: Why 1-6 Layers?

**Hypothesis space:**
- **Depth 1**: Sees only immediate neighbors (1 hop)
  - Can learn: "nearby rich neighborhoods are rich"
  - Cannot learn: "5-hop transit is important"
- **Depth 2-3**: Sees district level (2-3 km)
  - Can start learning multi-scale patterns
- **Depth 4-5**: Sees city-wide patterns (4-5 km)
  - Should capture transit corridor effects
- **Depth 6**: Sees most of Chicago (6+ km in tract graph)
  - Should capture Loop effect

**Why not deeper?**
- Diminishing returns: Chicago only ~800 tracts, diameter ~8 hops max
- Depth 6 is sufficient for 6-hop dependencies (our synthetic task)
- Deeper = risk of over-smoothing (all nodes become similar)

**Why test all 6?**
- To observe the learning curve: does performance plateau?
- If R² improves consistently to depth 6, infer: deeper needed
- If R² plateaus at depth 3: infer: multi-scale structure weaker than expected

### 4.4 Synthetic Task Parameters (Week 5)

**Graph:** Barabási-Albert (50 nodes, attachment m=2)
- Why BA? Realistic: has hubs (like Chicago has downtown), sparse edges
- Why not random? Random doesn't have structure; defeats the purpose

**Income Function Formula:**
```
income[i] = 0.3*scale*local_feature[i] 
          + 0.5*scale*transit_5hop[i] 
          + 0.2*scale*global_component[i] 
          + noise(std=5000)
```

**Why these weights?**
- 0.5 on 5-hop: DOMINATES → shallow nets MUST fail
- 0.3 local: Nonzero → can't ignore immediate features
- 0.2 global: Cluster effect → encourages learning structure
- noise=5000: ~5% of scale (realistic signal-to-noise ratio)

**Why this function?**
- Ground truth: k=5 hops are necessary
- Tests hypothesis: depth 1-4 fail, depth 5-6 succeed
- Controlled experiment (unlike real data where we don't know ground truth)

---

## PART V: CONFIDENCE & STATISTICS

### 5.1 Evaluation Metrics

**Primary Metric: R² Score (Coefficient of Determination)**

Formula: 
```
R² = 1 - (SS_res / SS_tot)
where SS_res = Σ(y_true - y_pred)²
      SS_tot = Σ(y_true - mean(y_true))²
```

**Interpretation:**
- R² = 0.5 means model explains 50% of income variance
- R² = 0.8 is excellent (hard to predict income <0.8 with any model)
- R² < 0.3 means model worse than predicting mean (useless)

**Why R²?**
- Scale-invariant (works whether income is in dollars or thousands)
- Interpretable (% of variance explained)
- Comparable across datasets
- Problematic if: outliers present (use robust R² = median instead)

**Secondary Metrics:**
- **MAE (Mean Absolute Error)** in dollars
  - E.g., MAE=$8,000 means off by $8k on average
  - Interpretable: directly tells practitioners the expected error
  
- **RMSE (Root Mean Squared Error)**
  - Penalizes large errors more
  - Comparable to standard deviation of income

### 5.2 Statistical Significance Testing

**Test 1: Paired t-test across depths**

```
H0: R²(depth=1) = R²(depth=6)  [null: no difference]
H1: R²(depth=6) > R²(depth=1)   [alt: deeper is better]

Procedure:
1. Train GNN at each depth 5 times (different random seeds)
2. Record R² for each run
3. Compute mean ± std for each depth
4. Paired t-test: does depth 6 significantly beat depth 1?
   t-stat = (mean_6 - mean_1) / SE(diff)
   p-value = P(t > t-stat | H0 true)
5. Accept H1 if p < 0.05
```

**Why paired t-test?**
- Same data split across depths (paired)
- Accounts for variance due to data vs. variance due to depth
- Standard in ML (common practice in AutoML papers)

**Example result:**
```
Depth 1: R² = 0.42 ± 0.03
Depth 6: R² = 0.58 ± 0.02
t-stat = 6.2, p-value = 0.0001 ***
→ Strongly significant: deep networks better
```

**Test 2: Confidence intervals on R²**

```
Procedure:
1. Train GNN 10 times with different seeds
2. Record R² values: [r2_1, r2_2, ..., r2_10]
3. Compute 95% CI: [quantile(0.025), quantile(0.975)]
   OR bootstrap CI: resample with replacement, recompute R²

Example:
Depth 6 R²: 0.58 [0.55, 0.61] (95% CI)
→ Likely true R² is between 0.55 and 0.61
→ Precise estimate (width = 0.06)
```

**Why important:**
- Single R² value is misleading (depends on data split)
- Confidence intervals show uncertainty
- Narrow CI = stable model; wide CI = unstable

### 5.3 Effect Size Quantification

**Metric: Cohen's d** (standardized difference)

Formula:
```
d = (R²_deep - R²_shallow) / pooled_std
```

**Interpretation:**
- d > 0.2: small effect
- d > 0.5: medium effect  
- d > 0.8: large effect

**Example:**
```
d = (0.58 - 0.42) / 0.05 = 3.2
→ Very large effect: depth has huge impact
```

### 5.4 Confidence in Synthetic Task Results

**Question: Do depth 1-4 networks FAIL on synthetic task?**

**Passing criterion:**
- MAE < 0.1 × income_range on test set
- E.g., income range = $180k, so MAE < $18k
- If MAE > $18k, consider it a FAILURE (network didn't learn)

**Example results:**
```
Synthetic task (5-hop dependency):
Depth 1: MAE = $62k ✗ (FAILED - can't see 5 hops)
Depth 2: MAE = $44k ✗ (FAILED - insufficient reach)
Depth 3: MAE = $28k ✗ (FAILED - still too shallow)
Depth 4: MAE = $19k ✗ (FAILED - barely short)
Depth 5: MAE = $8k  ✓ (PASSED - can see 5 hops)
Depth 6: MAE = $6k  ✓ (PASSED - can see 5+ hops)
```

**Confidence claim:**
> "We demonstrate that GNN depth is fundamentally necessary 
> for the synthetic 5-hop task. Networks with depth < 5 fail 
> (MAE > threshold), while depth ≥ 5 succeed, validating 
> Weisfeiler-Lehman expressivity theory."

---

## PART VI: CONCLUSIONS JUSTIFICATION

### 6.1 Claim: "Depth Matters"

**To claim this, we need ALL of:**

1. ✓ **Synthetic task success** (Week 5)
   - Depths 1-4 fail, depths 5-6 succeed
   - p < 0.01 significance test

2. ✓ **Real data validation** (Week 6)
   - Chicago: R² improves with depth
   - Improvement is statistically significant (p < 0.05)
   - Effect size is large (Cohen's d > 0.5)

3. ✓ **Theory alignment**
   - Weisfeiler-Lehman bounds explain why depth is necessary
   - Real Chicago graph diameter supports this (6+ hops needed)

4. ✓ **Robustness checks**
   - Results hold across multiple data splits
   - Results hold with different random seeds
   - Confidence intervals narrow (stable estimate)

**Example conclusion:**
> "Across synthetic and real datasets, we demonstrate that 
> graph neural network depth is necessary for learning 
> multi-scale geographic patterns. On our synthetic 5-hop 
> task, networks with depth < 5 achieve MAE > $62k, while 
> depth ≥ 5 networks achieve MAE < $10k (p < 0.001). 
> On real Chicago data, deeper networks (depth 6) achieve 
> R² = 0.58 vs. shallow networks (depth 1) achieving R² = 0.42 
> (paired t-test p = 0.0001, Cohen's d = 3.2). These results 
> validate Weisfeiler-Lehman expressivity theory: a k-layer 
> GNN cannot propagate information beyond k hops, thus cannot 
> learn spatial functions depending on >k-hop neighborhoods. 
> We conclude that for geographic tasks with multi-scale 
> structure, practitioners should use networks with depth 
> proportional to the required spatial range."

### 6.2 Claim: "Chicago Income is Multi-Scale"

**Evidence needed:**

1. ✓ **Feature importance analysis** (Week 7)
   - Local features (OSM): matter for baseline
   - Transit features: matter more than local
   - Scale matters: 5+ hop transit effects stronger than 1-hop

2. ✓ **Visualization of learned patterns** (Week 7)
   - t-SNE/UMAP of learned representations
   - Deep network representations cluster by income
   - Shallow network representations don't

3. ✓ **Residual analysis** (Week 7)
   - Which tracts does the model mispredict?
   - Isolated tracts (far from transit): high error (confirms multi-scale)
   - Loop-proximate tracts (close to CBD): low error

**Example conclusion:**
> "Chicago's neighborhood income exhibits multi-scale structure. 
> Our deep GNN learns a representation where income clusters 
> strongly with transit accessibility (r² = 0.62 between learned 
> features and transit_score), but shallow networks fail to capture 
> this (r² = 0.18). Through feature importance analysis, we find 
> that 5+-hop neighborhood features contribute 40% of the variance 
> explained, demonstrating that income depends on city-scale patterns 
> (proximity to downtown), not just local characteristics."

### 6.3 Claim: "We Can Do This Ethically"

**Evidence needed:**

1. ✓ **Privacy by design** (already documented)
   - No lat/lon + income stored together
   - Images deleted after feature extraction
   - Census tract aggregation (4,000+ people per unit)

2. ✓ **Bias audit** (Week 7)
   - Street View coverage audit (sparse tracts listed)
   - OSM coverage audit (building completeness × income correlation)
   - Sparse tracts excluded from training

3. ✓ **Fairness analysis** (Week 9)
   - Does model accuracy vary by income level? (Should be uniform)
   - Does model accuracy vary by neighborhood demographic? (Should be uniform)
   - If disparities exist, document them

**Example conclusion:**
> "We implement privacy-by-design throughout the pipeline. 
> (1) No individual-level data is fetched or stored; all analysis 
> is at Census tract level (minimum 4,000 people). 
> (2) Street View images are deleted after ResNet feature extraction; 
> only 2048-dim embeddings are retained. 
> (3) Final dataset contains no latitude/longitude, preventing 
> reverse-geocoding of income. 
> We audit known biases: Street View coverage is 15% sparse in 
> South/West Side tracts (documented in data/audit/streetview_coverage.csv); 
> OSM building completeness correlates with income (r = 0.32, p < 0.001), 
> documented in data/audit/osm_coverage_audit.csv. 
> We exclude sparse tracts (n < 10 images) from training. 
> Our model achieves uniform accuracy across income quartiles 
> (R² differences < 0.05), suggesting fairness."

---

## PART VII: EXPERIMENTAL TIMELINE & DELIVERABLES

### Week 1-3: Phase 1 (COMPLETE ✓)
- Week 1: Topology foundations (metric spaces, compactness)
- Deliverable: `phase1_topology/week1_topology_foundations.ipynb`

### Week 4-6: Phase 2 (COMPLETE ✓)
- Week 4: GNN expressivity theory (Weisfeiler-Lehman)
- Week 5: Synthetic experiments
- Week 6: Real Chicago experiments
- Deliverables: `week4_...`, `week5_...`, `week6_chicago_experiments.ipynb`

### Week 7-9: Phase 3 (TO DO)
- Week 7: Decision boundaries (UMAP, feature importance, residual analysis)
- Week 8: Topological features (persistent homology, learned representations)
- Week 9: Equity synthesis (disparate impact analysis, fairness)
- **Key statistic:** Accuracy disparity across groups (should be < 0.05 R²)

### Week 10-12: Phase 4 (TO DO)
- Week 10: Lower bounds (Baire category theorem formalization)
- Week 11: Depth-width tradeoff (can we trade depth for width?)
- Week 12: Final paper (publication-ready manuscript)
- **Key deliverable:** Theorem statement + proof

---

## PART VIII: SUCCESS CRITERIA (Go/No-Go Gates)

**Gate 1 (End of Week 6): Does depth matter empirically?**
- ✓ PASS if: R²(depth 6) - R²(depth 1) > 0.10 AND p < 0.05
- ✗ FAIL if: difference < 0.05 or p > 0.05
- Action if FAIL: Re-examine graph construction, feature quality, data splits

**Gate 2 (End of Week 7): Can we explain WHAT the network learns?**
- ✓ PASS if: Feature importance clearly shows transit > local features
- ✗ FAIL if: All features equally important (model is black box)
- Action if FAIL: Add stronger geographic constraints, use saliency maps

**Gate 3 (End of Week 9): Is the model fair?**
- ✓ PASS if: R² disparity across income quartiles < 0.05
- ✗ FAIL if: Model biased toward/against certain groups
- Action if FAIL: Retrain with class weights, oversample underrepresented groups

**Gate 4 (End of Week 12): Can we write a compelling paper?**
- ✓ PASS if: Clear theorems, empirical validation, novel insights
- ✗ FAIL if: Results are incremental, not publishable
- Action if FAIL: Pivot to different research question

---

## PART IX: CONTINGENCIES

**If Street View quota exhausted:**
- Use existing 80% coverage (don't refetch)
- Exclude sparse tracts (≥20% missing coverage) from training
- Use Census + OSM + transit features only (ResNet features optional)

**If Chicago income distribution is linear (R² plateaus at 0.4):**
- Implies: insufficient multi-scale structure
- Alternative hypothesis: income driven by non-spatial factors (education, industry)
- Pivot: reframe as "can GNNs capture what's learnable?"

**If synthetic task fails (depth doesn't matter):**
- Implies: Weisfeiler-Lehman theory doesn't apply to regression
- Alternative: test on classification (k-hop node label prediction)
- Pivot: investigate whether graph structure matters at all

---

## SUMMARY: Research Design Checklist

- ✅ Clear, testable research question (H1, H2, H3)
- ✅ Justified data sources (4 public, low-cost)
- ✅ Explicit model selection (GNNs > alternatives)
- ✅ Transparent hyperparameters (learning rate, depth, splits)
- ✅ Statistical testing plan (paired t-test, CI, effect size)
- ✅ Specific evaluation metrics (R², MAE, fairness disparity)
- ✅ Confidence criteria (what counts as "success"?)
- ✅ Reproducible setup (fixed seeds, logged transforms)
- ✅ Ethical safeguards (privacy audit, bias documentation)
- ✅ Go/no-go gates (when to pivot if results disappointing)

**This prospectus is:**
- Detailed enough for a committee to evaluate
- Flexible enough to adapt if unexpected results occur
- Rigorous enough to support publication claims
- Clear enough that someone else could reproduce it
