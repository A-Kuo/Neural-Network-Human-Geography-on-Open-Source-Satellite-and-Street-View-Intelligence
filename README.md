# Topological Foundations of Neural Network Expressivity in Human Geographic Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch) ![PyG](https://img.shields.io/badge/PyTorch%20Geometric-GNNs-orange) ![Docker](https://img.shields.io/badge/Docker-reproducible-blue?logo=docker) ![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

> **Maps socioeconomic structure from open satellite/street-view imagery, OSM, and transit data — and uses it to test *why* deep networks beat shallow ones on multi-scale geography.**

## Problem

Socioeconomic mapping from imagery usually stops at "the CNN predicts income, R² = X." This project asks the harder question underneath: geographic signal lives at multiple spatial scales (block → neighborhood → city), and there are theoretical reasons (Weisfeiler-Lehman bounds) to believe shallow GNNs *cannot* integrate across them. Chicago — with its sharp transit-accessibility and income gradients — is the testbed, built entirely from open data with tract-level privacy enforced in code.

## Approach

```
Street View imagery ─► ResNet-152 embeddings (originals deleted — ethics)
OpenStreetMap       ─► building footprints/height → tract aggregates
CTA/Metra GTFS      ─► multi-scale transit accessibility features
US Census ACS       ─► income/poverty/education targets
        │
        ▼
Privacy-validated tract-level dataset (n≈800 Chicago tracts, no lat/lon)
        │
        ▼
Shallow (1–2 layer) vs. deep (6–8 layer) GNN/CNN comparison
+ topological analysis of what each depth can express
```


**Research Question:** How does neural network depth affect learning of multi-scale geographic patterns in Chicago transit infrastructure and neighborhood economics? Can we prove that shallow networks fail to capture hierarchical spatial structure that deep networks learn?

**Dataset:** Chicago Census tracts (n≈800) with Street View imagery, OpenStreetMap buildings, CTA/Metra transit network (GTFS), and US Census income data.

**Timeline:** 12 weeks (4 phases)

---

## Project Structure

```
.
├── data_pipeline/              # Data fetching, cleaning, feature extraction
│   ├── fetch_streetview.py     # Google Street View API (rate-limited, privacy-aware)
│   ├── fetch_osm.py            # OpenStreetMap buildings + streets (Overpass API)
│   ├── fetch_gtfs.py           # CTA + Metra transit schedules (public GTFS)
│   ├── fetch_census.py         # US Census Bureau ACS data (Census API)
│   ├── extract_features.py     # ResNet-152 embeddings → delete originals (ethics)
│   ├── clean_osm.py            # Building aggregation to tract level
│   ├── compute_transit.py      # Transit accessibility metrics (multi-scale)
│   └── build_dataset.py        # Privacy-validated final join
│
├── phase1_topology/            # Real analysis + geographic metric spaces
│   └── week1_topology_foundations.ipynb
│
├── phase2_approximation/       # Stone-Weierstrass + GNN expressivity theory
├── phase3_geographic_learning/ # CNN + GNN on real data, decision boundaries
├── phase4_expressivity_bounds/ # Depth vs. width, lower bounds, empirical validation
│
├── src/                        # Core modules
│   ├── data_loading.py         # Load/normalize final dataset
│   ├── graph_construction.py   # Build GNNs from geographic graphs
│   └── model_training.py       # Train shallow vs. deep architectures
│
├── visualizations/             # Maps, embeddings, decision boundaries
├── notebooks/                  # Executable analyses
├── tests/                      # Unit + integration tests
├── figures/                    # Publication-quality outputs
├── proofs/                     # LaTeX/markdown formal proofs
├── writeup/                    # Final paper
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Reproducible environment
├── ETHICS.md                   # Data provenance, bias documentation
└── README.md                   # This file
```

---

## Quick Start

### 1. Setup

```bash
# Clone repo and activate environment
git clone https://github.com/a-kuo/...
cd Neural-Network-Human-Geography-...
pip install -r requirements.txt

# OR use Docker
docker build -t geographic-ml .
docker run -p 8888:8888 -v $(pwd):/project geographic-ml
```

### 2. Fetch Data (requires API keys)

```bash
# Set API keys (see ETHICS.md for how to obtain)
export GOOGLE_SV_API_KEY="..."      # Street View Static API key
export CENSUS_API_KEY="..."         # Census Bureau API key

# Fetch all layers (each is resumable/idempotent)
cd data_pipeline

python fetch_census.py --api-key $CENSUS_API_KEY --output ../data/raw/census/
python fetch_osm.py --output ../data/raw/
python fetch_gtfs.py --output ../data/raw/gtfs/
python fetch_streetview.py --api-key $GOOGLE_SV_API_KEY \
  --tracts ../data/processed/tract_centroids.geojson \
  --output ../data/raw/streetview/
```

### 3. Process & Extract Features

```bash
# ResNet feature extraction (GPU recommended)
python extract_features.py --device cuda --output-dir ../data/processed/image_features/

# Aggregate to tract level
python clean_osm.py --buildings ../data/raw/osm_buildings.geojson \
  --tracts ../data/raw/census/cook_county_tracts_2022.geojson \
  --output ../data/processed/tract_building_stats.parquet

python compute_transit.py \
  --stops ../data/raw/gtfs/all_transit_stops.geojson \
  --tracts ../data/raw/census/cook_county_tracts_2022.geojson \
  --output ../data/processed/tract_transit_features.parquet

# Final join (privacy-validated)
python build_dataset.py
```

### 4. Explore Week 1 Topology

```bash
jupyter lab notebooks/
# Open: phase1_topology/week1_topology_foundations.ipynb
```

---

## Key Research Claim

**Hypothesis:** Chicago's transit network has a multi-scale hierarchical structure:
- **Scale 1 (local):** Blocks and immediate neighborhoods (0-500m)
- **Scale 2 (neighborhood):** Districts and accessibility corridors (500m-2km)  
- **Scale 3 (city):** Downtown proximity + travel time to economic core (2km+)

This hierarchy explains **why depth matters**:
- **Shallow networks** (1-2 GNN layers) can only see Scale 1 — they learn local building/street characteristics but miss the broader economic structure
- **Deep networks** (6-8 layers) can propagate signals across scales — they discover that transit accessibility (Scale 3) predicts income (Census data)

We will prove this using:
1. **Topological theory** (Weisfeiler-Lehman bounds on shallow GNNs)
2. **Empirical experiments** (shallow networks plateau on synthetic + real tasks where deep networks succeed)
3. **Geographic insights** (visualization of learned representations reveals transit-income alignment in deep GNN features)

---

## Data Pipeline & Ethics

### Privacy Rules (Enforced in Code)

1. **Minimum aggregation unit:** Census tract (≥4,000 people)
   - No individual addresses, households, or persons stored
   - All joins are at tract level; reverse-geocoding impossible

2. **Image deletion (mandatory)**
   - `extract_features.py` deletes original Street View JPEGs after ResNet embedding
   - Final dataset stores only 2048-dim feature vectors + tract ID
   - Original images not retained anywhere

3. **No lat/lon + income join**
   - `build_dataset.py` enforces: final dataset has NO latitude/longitude columns
   - This prevents reverse-lookup of which addresses are wealthy/poor
   - Coordinates stay in raw data only; processed dataset has tract_id only

4. **Bias documentation**
   - Every data layer has a coverage audit (see `data/audit/`)
   - Street View: which tracts have sparse imagery?
   - OSM buildings: is wealthy-neighborhood data richer?
   - Transit: are poor neighborhoods underserved?
   - All documented in `ETHICS.md`

See **ETHICS.md** for detailed limitations + policy implications.

---

## Data Sources (All Public/Free)

| Source | License | Coverage | Key Variables |
|--------|---------|----------|----------------|
| **Google Street View Static API** | Free tier (25k/mo) | ~90% of Chicago streets | Pixel-level street imagery → ResNet features |
| **OpenStreetMap** | ODbL | ~99% building coverage | Footprints, height, type |
| **CTA/Metra GTFS** | Public domain | 100% of transit routes | Stop locations, schedules, travel times |
| **US Census ACS** | Public domain | 100% of Census tracts | Median income, poverty, education, population |

All data is open-source and reproducible. No private datasets required.

---

## Reproducibility

Every transformation is logged:
- `data/audit/streetview_coverage.csv` — which tracts have sparse imagery?
- `data/audit/feature_extraction_log.csv` — extraction status per tract
- `data/audit/build_dataset_transform_log.csv` — every join operation with row counts
- All scripts are version-controlled and idempotent (safe to rerun)

To regenerate the dataset:
```bash
git log --oneline data_pipeline/  # See what changed
git show HEAD:data_pipeline/build_dataset.py  # Inspect exact transformation
```

---

## Phase Breakdown

### **Phase 1 (Weeks 1-3): Topology Foundations**
- Metric spaces on geographic data
- Connectedness of settlement networks  
- Compactness of bounded territories
- **Deliverable:** `phase1_topology/week1_topology_foundations.ipynb` (executable, visualizations)

### **Phase 2 (Weeks 4-6): Approximation Theory**
- Stone-Weierstrass theorem on spatial functions
- Graph neural network universal approximation
- Synthetic task: compare shallow vs. deep GNNs
- **Deliverable:** Proof that depth is necessary; empirical comparison

### **Phase 3 (Weeks 7-9): Geographic Learning**
- Train GNNs on real Chicago data
- Decision boundaries: what do networks learn?
- Shallow networks fail on complex multi-scale patterns
- **Deliverable:** Visualizations, learned embeddings, quantitative comparison

### **Phase 4 (Weeks 10-12): Expressivity Bounds**
- Formal lower bounds: why shallow networks fail
- Baire category theorem applied to graphs
- Full experimental validation
- **Deliverable:** Paper-ready proofs + results

---

## Key Files for Contributors

- **Starting point:** `phase1_topology/week1_topology_foundations.ipynb`
- **Data pipeline docs:** Each `data_pipeline/*.py` file has detailed docstrings
- **Ethics:** Read `ETHICS.md` before analyzing sensitive data
- **Dependencies:** `requirements.txt` (install with `pip install -r requirements.txt`)

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Citation

If you use this dataset or code, please cite:
```bibtex
@misc{kuo2026geographic,
  title={Topological Foundations of Neural Network Expressivity in Human Geographic Analysis},
  author={Kuo, A.},
  year={2026},
  url={https://github.com/a-kuo/neural-network-human-geography-...}
}
```

---

## Contact

For questions on data, ethics, or research direction, see `ETHICS.md` for contact details and discussion of potential misuses.

---

**Last updated:** 2026-04-10  
**Reproducible with:** Python 3.11+, Docker


---

## Evaluation Design

Research in progress — numbers land as Phases 2–3 complete. The comparisons are fixed in advance so results can't be cherry-picked:

| Question | Test | Metric |
|----------|------|--------|
| Does depth matter on synthetic multi-scale tasks? | shallow vs. deep GNN on constructed hierarchies (Phase 2) | accuracy plateau gap |
| Does depth matter on real geography? | 1–2 layer vs. 6–8 layer GNN predicting tract income (Phase 3) | R² / MAE vs. tract-feature baseline |
| Is the mechanism transit accessibility? | representation probing of learned embeddings | transit–income alignment in deep features |
| Are the theory claims sound? | WL-bound and Baire-category arguments (Phase 4) | formal proofs in `proofs/` |

## Tech Stack

- **ML**: PyTorch, PyTorch Geometric (GNNs), ResNet-152 feature extraction
- **Geo/Data**: GeoPandas, OSMnx/Overpass, GTFS parsing, Census API, pandas/pyarrow
- **Rigor**: pytest pipeline tests, per-tract coverage audits, Docker environment, ETHICS.md

## Status

🔬 Research prototype. Data pipeline and Phase 1 (topology foundations) complete; Phase 2 (approximation theory + synthetic depth experiments) in progress. Every data layer has a coverage/bias audit before modeling conclusions are drawn.

## Author

**Austin Kuo** | [GitHub](https://github.com/A-Kuo) | ML Engineer & Data Engineer
