# Ethics, Data Provenance & Limitations

**Purpose:** This document details what this dataset can and cannot be used for, documents known biases, and specifies the non-negotiable privacy rules enforced throughout the pipeline.

**Intended Use:** Academic research on neural network expressivity, urban geography, and public health equity. NOT for targeting individuals or neighborhoods.

---

## Data Provenance & Sources

All data is open-source, publicly available, and reproducible.

| Source | License | Obtained | Coverage | Key Limitation |
|--------|---------|----------|----------|-----------------|
| Google Street View Static API | Free tier (25,000 req/month) | Publicly available | ~90% of Chicago streets | Rate-limited; some neighborhoods less photographed |
| OpenStreetMap (buildings) | Open Data Commons ODbL | Overpass API (public) | ~99% coverage | Wealthy areas may have more detailed tagging |
| CTA/Metra GTFS | Public domain | Official transit agency sites | 100% route coverage | Schedules ≠ actual service |
| US Census Bureau ACS | Public domain | Census.gov API | 100% of Census tracts | 5-year estimates (not current-year precision) |

**None of these sources contain confidential information.**

---

## Privacy Rules (Non-Negotiable)

The pipeline enforces these rules in code. If any rule is violated, the build fails.

### Rule 1: Aggregation Threshold
- **Minimum unit:** Census tract (typically 4,000+ people)
- **Never store together:** [latitude, longitude, individual_income]
- **Never store:** Addresses, street names, or individual household IDs
- **Enforcement:** `build_dataset.py` raises `ValueError` if any PII column pattern is detected

### Rule 2: Image Handling
- **Original images:** Deleted immediately after ResNet feature extraction
- **Stored only:** Tract ID + 2048-dim feature vectors (anonymized embeddings)
- **No reverse-lookup possible:** Feature vector alone cannot identify buildings or streets
- **Enforcement:** `extract_features.py` calls `os.remove()` on each processed JPEG; audit log documents deletion count

### Rule 3: Transit Data
- **Individual commutes:** Never inferred or stored
- **Aggregated only:** Mean/median/percentile accessibility metrics per tract
- **No person-to-place mapping:** Cannot link any individual to their travel time

### Rule 4: Income Variable
- **Usage:** Census tract aggregate (published publicly by Census Bureau)
- **Framing:** "Neighborhood economic indicator," not "individual wealth"
- **Never claim:** "This model predicts [person X]'s income"
- **Never use for:** Redlining, targeted lending discrimination, or individual targeting

### Rule 5: Bias Documentation
- **Coverage audits:** Included for every data layer (see `data/audit/`)
- **Tracts with sparse data:** Flagged and excluded from ML training
- **Reproducible:** Every transformation logged to CSV (see `build_dataset_transform_log.csv`)

### Rule 6: Auditability
- **Git history:** Every data transformation is version-controlled
- **Transformation logs:** Saved to `data/audit/` and committed
- **Reproducibility:** Entire pipeline is deterministic (same inputs → same outputs)

---

## Known Biases & Coverage Gaps

### Street View Imagery Bias

**Issue:** Google Street View coverage is not uniform.

| Issue | Impact | Documentation |
|-------|--------|-----------------|
| **South Side undersampling** | Historically lower Street View coverage in South Side neighborhoods | `data/audit/streetview_coverage.csv`: list sparse tracts |
| **Seasonal variation** | Winter images from different year than summer | Metadata included; consider temporal artifacts |
| **Daytime only** | All images taken 9 AM–4 PM | Night-time street characteristics missing |
| **Algorithm bias in imagery selection** | Google's algorithm may avoid certain areas | Unknown; document if discovered |

**Mitigation:** 
- Minimum image threshold: tracts with <10 images are flagged `sparse` and excluded from training
- Bias audit: `data/audit/streetview_coverage.csv` lists all sparse/missing tracts
- Transparent reporting: Final paper lists which neighborhoods lack data

### OpenStreetMap Bias

**Issue:** OSM building data quality correlates with area wealth.

| Issue | Data | Bias |
|-------|------|------|
| **Wealthy neighborhoods** | More volunteer contributors | More detailed tagging (height, material, age) |
| **Poorer neighborhoods** | Fewer OSM editors | Less complete data (missing height, type) |
| **Downtown Chicago** | Heavily mapped | Building footprints very detailed |
| **Suburban Cook County** | Sparser mapping | May miss small structures |

**Detected correlation:** `data/audit/osm_coverage_audit.csv` includes r-value between building_density and income.

**Mitigation:**
- Flag missing values: `pct_missing_height` column documents data completeness
- Do NOT use OSM density as a proxy for actual building density (it may reflect mapping effort, not reality)
- Report: "X% of tracts have height data; poorer neighborhoods have Y% data sparsity"

### Transit Coverage Bias

**Issue:** Metra (commuter rail) serves suburbs; CTA (buses + L) serves city.

| Mode | Coverage | Equity Issue |
|------|----------|--------------|
| **CTA L (rapid transit)** | Well-distributed | Serves lower-income areas more than wealthy suburbs |
| **CTA buses** | City-wide | Subject to service cuts in lower-income areas (historical) |
| **Metra rails** | Suburban (mostly) | Mostly serves wealthier suburbs |

**Impact:** Transit accessibility metric may conflate "close to a bus stop" with "good transit" — but bus service quality varies by neighborhood wealth.

**Mitigation:**
- Use only GTFS schedule data (stops exist; frequency/reliability ≠ perfect)
- Document: "Transit score reflects accessibility, not service quality"
- Do NOT use as a policy tool without human review of actual service

### Census Data Bias

**Issue:** ACS 5-year estimates use small-area estimation (statistical uncertainty).

| Tract Population | ACS Data Type | Reliability |
|------------------|---------------|------------|
| >65,000 | 1-year estimates | High |
| 20,000–65,000 | 3-year estimates | Medium |
| <20,000 | 5-year estimates | Low (noisy) |

**For Chicago:** Most tracts have 5,000–15,000 people → ACS estimates have ±15% margin of error.

**Impact:** Small differences in income between neighborhoods may be noise, not signal.

**Mitigation:** 
- Use median (not mean) for robustness
- Report confidence intervals (Census publishes margins of error)
- Avoid over-interpreting fine-grained differences

---

## What This Dataset Is NOT For

### ❌ DO NOT use for:

1. **Individual targeting:** "This person lives in ZIP code X, so they probably earn Y"
2. **Redlining or lending discrimination:** "This tract has low transit_score, so deny mortgage"
3. **Police deployment:** "This neighborhood has feature profile Z, so patrol more"
4. **Eviction prediction:** "These buildings have certain visual characteristics, so evict residents"
5. **Discriminatory marketing:** "Show ads only to people in high-income areas"
6. **Any application that treats model predictions as certain truth** (they are not)

### ⚠️ USE WITH CAUTION:

- **Policy planning:** Model insights can inform discussion, but require human review and ground truth
- **Hypothesis generation:** Use to *ask* questions, not to *conclude* answers
- **Resource allocation:** Consider multiple data sources; never rely on this alone

---

## What This Dataset IS Good For

### ✅ Recommended uses:

1. **Understanding neural network expressivity:** Why do deep networks outperform shallow ones on geographic data?
2. **Validating geographic theory:** Does multi-scale structure affect learning?
3. **Improving urban equity research:** Can we transparently document biases in geographic ML?
4. **Pedagogy:** Teaching privacy-preserving data science to students
5. **Methodological contribution:** Demonstrating how to build privacy-aware geographic datasets

---

## Missing Data & Gaps

### Completely Absent:

- **Race/ethnicity data:** Not included (Census data is available separately; not joined here)
- **Individual income:** Only tract medians (as required by privacy rules)
- **Employment/occupation:** Not available at tract level without additional data
- **Historical data:** Current snapshot only; no temporal evolution
- **Non-English neighborhoods:** Street View labels in English; linguistic bias possible

### Sparse Coverage:

| Tract Type | Data Quality |
|-----------|--------------|
| **Downtown/Loop** | Excellent (heavy Street View coverage) |
| **North Shore suburbs** | Good (good GTFS, decent OSM) |
| **South/West Side neighborhoods** | Fair to poor (gaps in Street View, OSM density varies) |
| **Industrial zones** | Sparse (few images, buildings underrepresented in OSM) |
| **Parks, airports, cemeteries** | Excluded (not residential tracts) |

**See:** `data/audit/streetview_coverage.csv` for exact tract-by-tract status.

---

## Fairness & Equity Implications

### The Research Question Itself Has Equity Implications

Our question is: "Can deep networks learn multi-scale patterns that shallow networks miss?"

**Equity angle:** Multi-scale geographic structure is *partly* a proxy for infrastructure inequality. Deep networks might learn:
- ✅ That transit accessibility predicts income (descriptive)
- ✅ Historical patterns of segregation (informative for remedy)
- ❌ That poor neighborhoods should remain poor (dangerous conclusion)

**Our responsibility:**
1. **Clearly separate description from prescription**
2. **Never present learned patterns as natural or inevitable**
3. **Highlight that patterns reflect historical policy choices, not geographic destiny**
4. **Recommend how findings could inform equitable policy** (e.g., "If transit access predicts outcome, increase transit investment in underserved areas")

### Potential Harms

| Harm | How It Could Happen | Mitigation |
|------|-------------------|-----------|
| **Reinforcing stereotypes** | "Poor neighborhoods have feature profile X" reported without context | Always contextualize with historical policy + acknowledge limitations |
| **Justifying underinvestment** | "This model shows why area Y won't improve" | Explicitly reject determinism; emphasize policy levers |
| **Targeting vulnerable groups** | Using model for gentrification prediction | Do NOT use for real-estate targeting; publish limitations prominently |
| **Privacy leakage** | Reverse-engineering individual homes from features | Aggregation to tract level mitigates this; audit privacy rules |

---

## How to Report Misuses

If you discover this dataset being used in ways contrary to ETHICS.md (e.g., for individual targeting, discrimination, or privacy violation):

1. **Document the misuse** (screenshot, URL, description)
2. **Contact the authors** (file GitHub issue with `[ETHICS]` label)
3. **Report to relevant authority** (university IRB, if applicable)

---

## Audit Checklist for Users

Before using this dataset, verify:

- [ ] I have read and understood ETHICS.md
- [ ] My use case is for research, not targeting individuals
- [ ] I will aggregate results at tract level, never below
- [ ] I will document data limitations in my paper/report
- [ ] I will contextualize findings (not present patterns as inevitable)
- [ ] I understand Census Bureau confidentiality rules (not applicable for published ACS, but good practice)
- [ ] I will not attempt to reverse-geocode or link features back to individuals

---

## Data Dictionary & Definitions

### tract_id
- **Type:** String (11-digit Census FIPS)
- **Format:** SSCCCTTTTTT (state + county + tract)
- **Example:** "17031234500" = Illinois (17) + Cook County (031) + Tract 2345.00
- **Use for:** Joins only; never for geographic identification

### n_images
- **Type:** Integer
- **Definition:** Number of Street View images used for ResNet extraction in this tract
- **Threshold:** Tracts with n_images < 10 are flagged ml_ineligible
- **Bias:** High counts indicate well-photographed areas (downtown > periphery)

### img_feat_0 to img_feat_2047
- **Type:** Float32
- **Definition:** ResNet-152 average embedding (mean across all images in tract)
- **Interpretation:** High-dimensional representation of street-level visual characteristics
- **Caution:** Embeddings are opaque; do not over-interpret individual features

### building_count
- **Type:** Integer
- **Definition:** Number of building footprints in OSM within tract
- **Bias:** Reflects OSM completeness, not actual building count
- **Use:** For normalized density (building_density_km2) only

### building_density_km2
- **Type:** Float
- **Definition:** building_count / land_area_km2
- **Interpretation:** Buildings per square km of land area
- **Caution:** May correlate with mapping effort, not actual density

### median_height_m
- **Type:** Float
- **Definition:** Median estimated building height (in meters)
- **Missing:** NaN if <50% of buildings have height data
- **Bias:** Sparse in less-mapped areas; better in downtown

### nearest_stop_dist_km
- **Type:** Float
- **Definition:** Distance (km) from tract centroid to nearest transit stop
- **Radius:** Typically 0.5–2.5 km across Chicago
- **Interpretation:** Walking distance to transit (5 min = ~400m; 10 min = ~800m)

### n_stops_500m
- **Type:** Integer
- **Definition:** Number of CTA/Metra stops within 500m walking radius
- **Interpretation:** Local transit accessibility (0–20 typical)
- **Note:** Count, not frequency; does not account for service level

### n_stops_1km
- **Type:** Integer
- **Definition:** Number of stops within 1 km
- **Interpretation:** Neighborhood-scale transit access

### median_travel_time_loop
- **Type:** Float (minutes)
- **Definition:** Median GTFS-based travel time from nearest accessible stop to Chicago Loop
- **Interpretation:** Commute time estimate (15–90 min typical)
- **Caution:** GTFS ≠ actual service; schedules change; no real-time delays

### transit_score
- **Type:** Float (0–100)
- **Definition:** Composite accessibility score (see compute_transit.py for formula)
- **Interpretation:** Higher = better transit access
- **Use:** Not as a policy label; input feature only

### median_household_income
- **Type:** Float (US dollars)
- **Definition:** Census ACS median household income (5-year estimate)
- **Margin of error:** ±15% typical (see ACS documentation)
- **Target variable:** What the GNN will be trained to predict

### poverty_rate
- **Type:** Float (0–1)
- **Definition:** Fraction of population below Census poverty line (5-year ACS estimate)
- **Missing:** NaN if poverty universe <100

### pct_college_plus
- **Type:** Float (0–1)
- **Definition:** Fraction of population 25+ with bachelor's degree or higher
- **Bias:** May reflect population age structure (young vs. old neighborhoods)

### total_population
- **Type:** Integer
- **Definition:** Total population of Census tract (5-year ACS)
- **Use:** For validating aggregation threshold (all >1,000)

### ml_eligible
- **Type:** Boolean
- **Definition:** True if tract has ≥10 Street View images (meets training threshold)
- **Impact:** ML training set restricted to ml_eligible==True
- **Bias:** Excludes some neighborhoods due to Street View coverage

---

## Contact & Feedback

**Questions on data or ethics?** 
- File a GitHub issue with label `[ETHICS]` or `[DATA]`
- All feedback is welcome; we're committed to transparency

**Reporting misuse:**
- See "How to Report Misuses" section above

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-10 | Initial release |

---

**Last reviewed:** 2026-04-10  
**Author:** [Research Team]  
**License:** CC-BY-4.0 (this ethics document is public)
