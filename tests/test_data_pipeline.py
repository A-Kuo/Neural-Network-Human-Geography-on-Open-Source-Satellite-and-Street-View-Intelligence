"""
tests/test_data_pipeline.py — Unit tests for data pipeline modules.

Tests are designed to run without API keys (no network calls).
They validate logic, data transformations, and privacy guardrails.

Run with:
    pytest tests/test_data_pipeline.py -v
"""

import re
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, Polygon
from unittest.mock import patch, MagicMock


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_tracts_gdf():
    """Minimal GeoDataFrame of 5 synthetic Census tracts (Chicago-like coordinates)."""
    geometries = [
        Polygon([(-87.94, 41.84), (-87.93, 41.84), (-87.93, 41.85), (-87.94, 41.85)]),
        Polygon([(-87.93, 41.84), (-87.92, 41.84), (-87.92, 41.85), (-87.93, 41.85)]),
        Polygon([(-87.94, 41.85), (-87.93, 41.85), (-87.93, 41.86), (-87.94, 41.86)]),
        Polygon([(-87.93, 41.85), (-87.92, 41.85), (-87.92, 41.86), (-87.93, 41.86)]),
        Polygon([(-87.92, 41.84), (-87.91, 41.84), (-87.91, 41.85), (-87.92, 41.85)]),
    ]
    return gpd.GeoDataFrame(
        {"tract_id": [f"1703100{i:04d}00" for i in range(1, 6)],
         "total_population": [5000, 4200, 6100, 4800, 5500]},
        geometry=geometries,
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_buildings_gdf():
    """Minimal GeoDataFrame of building footprints (within tract boundaries, Chicago-like)."""
    footprints = [
        Polygon([(-87.935, 41.841), (-87.934, 41.841), (-87.934, 41.842), (-87.935, 41.842)]),
        Polygon([(-87.925, 41.851), (-87.924, 41.851), (-87.924, 41.852), (-87.925, 41.852)]),
        Polygon([(-87.915, 41.841), (-87.914, 41.841), (-87.914, 41.842), (-87.915, 41.842)]),
        Polygon([(-87.925, 41.851), (-87.924, 41.851), (-87.924, 41.852), (-87.925, 41.852)]),
        Polygon([(-87.915, 41.851), (-87.914, 41.851), (-87.914, 41.852), (-87.915, 41.852)]),
    ]
    return gpd.GeoDataFrame(
        {
            "osm_id": range(1001, 1006),
            "building_type": ["residential", "commercial", "residential", "yes", "apartments"],
            "height_m": [8.0, 15.0, None, 6.0, None],
            "area_m2": [400.0, 400.0, 100.0, 100.0, 100.0],
        },
        geometry=footprints,
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_acs_df():
    """Synthetic ACS income DataFrame for 5 tracts."""
    return pd.DataFrame({
        "tract_id": [f"1703100{i:04d}00" for i in range(1, 6)],
        "median_household_income": [75000, 42000, 95000, 38000, 62000],
        "total_population": [5000, 4200, 6100, 4800, 5500],
        "poverty_rate": [0.08, 0.22, 0.04, 0.28, 0.12],
        "pct_no_vehicle": [0.15, 0.35, 0.08, 0.40, 0.20],
        "pct_college_plus": [0.55, 0.22, 0.72, 0.18, 0.41],
        "total_housing_units": [2100, 1800, 2600, 2000, 2300],
        "acs_year": [2022] * 5,
    })


# ── test_clean_osm.py ──────────────────────────────────────────────────────────

class TestCleanOSM:
    """Tests for data_pipeline.clean_osm."""

    def test_classify_building_residential(self):
        from data_pipeline.clean_osm import classify_building
        assert classify_building("residential") == "residential"
        assert classify_building("house") == "residential"
        assert classify_building("apartments") == "residential"

    def test_classify_building_commercial(self):
        from data_pipeline.clean_osm import classify_building
        assert classify_building("commercial") == "commercial"
        assert classify_building("retail") == "commercial"
        assert classify_building("office") == "commercial"

    def test_classify_building_unknown(self):
        from data_pipeline.clean_osm import classify_building
        assert classify_building("yes") == "unspecified"
        assert classify_building("") == "unknown"
        assert classify_building(None) == "unknown"

    def test_compute_building_areas_positive(self, sample_buildings_gdf):
        from data_pipeline.clean_osm import compute_building_areas
        result = compute_building_areas(sample_buildings_gdf)
        assert "area_m2" in result.columns
        assert (result["area_m2"] > 0).all(), "All building areas must be positive"

    def test_spatial_join_assigns_tract_ids(self, sample_buildings_gdf, sample_tracts_gdf):
        from data_pipeline.clean_osm import compute_building_areas, spatial_join_to_tracts
        buildings_with_area = compute_building_areas(sample_buildings_gdf)
        joined = spatial_join_to_tracts(buildings_with_area, sample_tracts_gdf)
        assert "tract_id" in joined.columns
        # All joined buildings should have a valid tract_id
        assert joined["tract_id"].notna().all()

    def test_aggregate_to_tract_no_negative_density(self, sample_buildings_gdf, sample_tracts_gdf):
        from data_pipeline.clean_osm import compute_building_areas, spatial_join_to_tracts, aggregate_to_tract
        buildings_with_area = compute_building_areas(sample_buildings_gdf)
        joined = spatial_join_to_tracts(buildings_with_area, sample_tracts_gdf)
        result = aggregate_to_tract(joined, sample_tracts_gdf)
        # Density must be non-negative where not NaN
        valid_density = result["building_density_km2"].dropna()
        assert (valid_density >= 0).all(), "Building density must be non-negative"

    def test_aggregate_to_tract_covers_all_tracts(self, sample_buildings_gdf, sample_tracts_gdf):
        from data_pipeline.clean_osm import compute_building_areas, spatial_join_to_tracts, aggregate_to_tract
        buildings_with_area = compute_building_areas(sample_buildings_gdf)
        joined = spatial_join_to_tracts(buildings_with_area, sample_tracts_gdf)
        result = aggregate_to_tract(joined, sample_tracts_gdf)
        # All tracts should appear in output (even those with 0 buildings)
        assert len(result) == len(sample_tracts_gdf)
        assert set(result["tract_id"]) == set(sample_tracts_gdf["tract_id"])

    def test_pct_missing_height_in_range(self, sample_buildings_gdf, sample_tracts_gdf):
        from data_pipeline.clean_osm import compute_building_areas, spatial_join_to_tracts, aggregate_to_tract
        buildings_with_area = compute_building_areas(sample_buildings_gdf)
        joined = spatial_join_to_tracts(buildings_with_area, sample_tracts_gdf)
        result = aggregate_to_tract(joined, sample_tracts_gdf)
        valid = result["pct_missing_height"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all(), "pct_missing_height must be in [0, 1]"


# ── test_build_dataset.py ─────────────────────────────────────────────────────

class TestBuildDataset:
    """Tests for data_pipeline.build_dataset — especially privacy guardrails."""

    def test_privacy_guard_raises_on_lat(self):
        from data_pipeline.build_dataset import _privacy_guard
        df_bad = pd.DataFrame({"tract_id": ["1703100100"], "lat": [41.8], "income": [50000]})
        with pytest.raises(ValueError, match="PRIVACY VIOLATION"):
            _privacy_guard(df_bad)

    def test_privacy_guard_raises_on_longitude(self):
        from data_pipeline.build_dataset import _privacy_guard
        df_bad = pd.DataFrame({"tract_id": ["1703100100"], "longitude": [-87.6], "income": [50000]})
        with pytest.raises(ValueError, match="PRIVACY VIOLATION"):
            _privacy_guard(df_bad)

    def test_privacy_guard_raises_on_address(self):
        from data_pipeline.build_dataset import _privacy_guard
        df_bad = pd.DataFrame({"tract_id": ["1703100100"], "street_address": ["123 Main St"]})
        with pytest.raises(ValueError, match="PRIVACY VIOLATION"):
            _privacy_guard(df_bad)

    def test_privacy_guard_passes_clean_df(self):
        from data_pipeline.build_dataset import _privacy_guard
        df_clean = pd.DataFrame({
            "tract_id": ["1703100100"],
            "median_household_income": [75000],
            "transit_score": [62.5],
            "building_density_km2": [1200.0],
        })
        # Should not raise
        _privacy_guard(df_clean)

    def test_ml_eligible_flag_correct(self):
        from data_pipeline.build_dataset import _verify_image_coverage
        df = pd.DataFrame({
            "tract_id": ["A", "B", "C"],
            "n_images": [15, 5, 30],
        })
        result = _verify_image_coverage(df)
        assert result.loc[result["tract_id"] == "A", "ml_eligible"].iloc[0] == True
        assert result.loc[result["tract_id"] == "B", "ml_eligible"].iloc[0] == False
        assert result.loc[result["tract_id"] == "C", "ml_eligible"].iloc[0] == True

    def test_ml_eligible_missing_n_images(self):
        from data_pipeline.build_dataset import _verify_image_coverage
        df = pd.DataFrame({"tract_id": ["A", "B"]})
        result = _verify_image_coverage(df)
        assert "ml_eligible" in result.columns
        assert (result["ml_eligible"] == False).all()

    def test_verify_aggregation_level_warns_small(self, caplog):
        import logging
        from data_pipeline.build_dataset import _verify_aggregation_level
        df = pd.DataFrame({
            "tract_id": ["A", "B"],
            "total_population": [500, 5000],  # A is below 1000 threshold
        })
        with caplog.at_level(logging.WARNING):
            _verify_aggregation_level(df)
        # Should not raise but may warn — just verify it runs
        assert True  # function returned normally

    def test_transformation_log_saves(self, tmp_path):
        from data_pipeline.build_dataset import TransformationLog
        log = TransformationLog()
        df = pd.DataFrame({"a": [1, 2, 3]})
        log.log("step1", df, "test note")
        log.log("step2", df.head(2), "truncated")
        out = tmp_path / "log.csv"
        log.save(out)
        assert out.exists()
        result = pd.read_csv(out)
        assert len(result) == 2
        assert list(result["step"]) == ["step1", "step2"]
        assert list(result["n_rows"]) == [3, 2]


# ── test_compute_transit.py ───────────────────────────────────────────────────

class TestComputeTransit:
    """Tests for data_pipeline.compute_transit."""

    def test_transit_score_in_range(self):
        from data_pipeline.compute_transit import compute_transit_score
        df = pd.DataFrame({
            "nearest_stop_dist_km": [0.2, 1.5, 5.0, np.nan],
            "n_stops_500m": [8, 2, 0, 0],
            "median_travel_time_loop": [20, 45, 90, np.nan],
        })
        scores = compute_transit_score(df)
        assert (scores >= 0).all(), "Transit score must be >= 0"
        assert (scores <= 100).all(), "Transit score must be <= 100"

    def test_transit_score_closer_is_higher(self):
        from data_pipeline.compute_transit import compute_transit_score
        df = pd.DataFrame({
            "nearest_stop_dist_km": [0.1, 5.0],
            "n_stops_500m": [10, 0],
            "median_travel_time_loop": [10, 90],
        })
        scores = compute_transit_score(df)
        assert scores.iloc[0] > scores.iloc[1], "Closer stop → higher transit score"

    def test_transit_score_handles_all_nan(self):
        from data_pipeline.compute_transit import compute_transit_score
        df = pd.DataFrame({
            "nearest_stop_dist_km": [np.nan, np.nan],
            "n_stops_500m": [np.nan, np.nan],
            "median_travel_time_loop": [np.nan, np.nan],
        })
        scores = compute_transit_score(df)
        assert len(scores) == 2
        assert (scores >= 0).all()

    def test_project_to_utm_changes_crs(self):
        from data_pipeline.compute_transit import _project_to_utm
        gdf = gpd.GeoDataFrame(
            {"tract_id": ["A"]},
            geometry=[Point(-87.6, 41.8)],
            crs="EPSG:4326",
        )
        result = _project_to_utm(gdf)
        assert result.crs.to_epsg() == 32616


# ── test_gnn_theory.py ────────────────────────────────────────────────────────

class TestGNNTheory:
    """Tests for phase2_approximation.gnn_theory."""

    @pytest.fixture
    def small_graph(self):
        import networkx as nx
        return nx.path_graph(10)

    @pytest.fixture
    def chicago_like_graph(self):
        import networkx as nx
        return nx.barabasi_albert_graph(50, 2, seed=42)

    def test_wl_iteration_changes_colors(self, small_graph):
        from phase2_approximation.gnn_theory import weisfeiler_lehman_iteration
        import networkx as nx
        initial_colors = {n: small_graph.degree(n) for n in small_graph.nodes()}
        new_colors = weisfeiler_lehman_iteration(small_graph, initial_colors, iteration=1)
        assert isinstance(new_colors, dict)
        assert set(new_colors.keys()) == set(small_graph.nodes())

    def test_wl_dimension_positive(self, chicago_like_graph):
        from phase2_approximation.gnn_theory import compute_wl_dimension
        wl_dim, color_history = compute_wl_dimension(chicago_like_graph, max_iterations=6)
        assert wl_dim >= 0
        assert isinstance(color_history, dict)
        assert 0 in color_history

    def test_class_count_nondecreasing(self, chicago_like_graph):
        from phase2_approximation.gnn_theory import compute_wl_dimension, count_node_classes_by_iteration
        _, color_history = compute_wl_dimension(chicago_like_graph, max_iterations=6)
        class_counts = count_node_classes_by_iteration(color_history)
        counts = [class_counts[i] for i in sorted(class_counts.keys())]
        # Number of distinct classes must be non-decreasing across iterations
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], \
                f"Class count decreased from iteration {i-1} to {i}: {counts[i-1]} → {counts[i]}"

    def test_class_count_bounded_by_nodes(self, chicago_like_graph):
        from phase2_approximation.gnn_theory import compute_wl_dimension, count_node_classes_by_iteration
        n = chicago_like_graph.number_of_nodes()
        _, color_history = compute_wl_dimension(chicago_like_graph, max_iterations=6)
        class_counts = count_node_classes_by_iteration(color_history)
        for it, count in class_counts.items():
            assert count <= n, f"Iteration {it}: {count} classes > {n} nodes (impossible)"

    def test_receptive_field_grows_with_depth(self, chicago_like_graph):
        from phase2_approximation.gnn_theory import receptive_field_by_depth
        rf_1 = receptive_field_by_depth(chicago_like_graph, 1)
        rf_3 = receptive_field_by_depth(chicago_like_graph, 3)
        avg_1 = np.mean(list(rf_1.values()))
        avg_3 = np.mean(list(rf_3.values()))
        assert avg_3 >= avg_1, "Receptive field must grow (or stay same) with depth"

    def test_receptive_field_max_bounded_by_depth(self, small_graph):
        from phase2_approximation.gnn_theory import receptive_field_by_depth
        for depth in [1, 2, 3]:
            rf = receptive_field_by_depth(small_graph, depth)
            for node, max_dist in rf.items():
                assert max_dist <= depth, \
                    f"Node {node}: receptive field {max_dist} > depth {depth}"

    def test_universal_approximation_bound_returns_dict(self, chicago_like_graph):
        from phase2_approximation.gnn_theory import universal_approximation_bound
        result = universal_approximation_bound(chicago_like_graph, 3, 64, 0.05)
        required_keys = {"wl_dimension", "width_naive", "width_with_depth",
                         "hidden_dimension_provided", "num_layers", "estimated_error_bound"}
        assert required_keys.issubset(set(result.keys()))

    def test_deeper_network_lower_error_bound(self, chicago_like_graph):
        from phase2_approximation.gnn_theory import universal_approximation_bound
        bound_shallow = universal_approximation_bound(chicago_like_graph, 1, 64, 0.05)
        bound_deep = universal_approximation_bound(chicago_like_graph, 5, 64, 0.05)
        assert bound_deep["estimated_error_bound"] <= bound_shallow["estimated_error_bound"], \
            "Deeper network should have lower or equal error bound"

    def test_distance_to_km_conversion(self):
        from phase2_approximation.gnn_theory import distance_to_kilometers
        # 5 hops × 0.8 km/hop = 4.0 km
        result = distance_to_kilometers(5, avg_hop_distance_km=0.8)
        assert abs(result - 4.0) < 1e-6


# ── test_synthetic_tasks.py ───────────────────────────────────────────────────

class TestSyntheticTasks:
    """Tests for phase2_approximation.synthetic_tasks."""

    @pytest.fixture
    def small_graph(self):
        import networkx as nx
        return nx.barabasi_albert_graph(20, 2, seed=42)

    def test_income_function_in_realistic_range(self, small_graph):
        from phase2_approximation.synthetic_tasks import create_synthetic_income_function
        fn = create_synthetic_income_function(small_graph, num_hops=3, seed=42)
        incomes = fn.ground_truth
        assert incomes.min() >= 20000, "Income must be >= $20k"
        assert incomes.max() <= 200000, "Income must be <= $200k"

    def test_income_function_has_variance(self, small_graph):
        from phase2_approximation.synthetic_tasks import create_synthetic_income_function
        fn = create_synthetic_income_function(small_graph, num_hops=3, seed=42)
        std = fn.ground_truth.std()
        assert std > 1000, f"Income std = ${std:.0f} — too low to test expressivity"

    def test_income_function_node_count_matches_graph(self, small_graph):
        from phase2_approximation.synthetic_tasks import create_synthetic_income_function
        fn = create_synthetic_income_function(small_graph, num_hops=3, seed=42)
        assert len(fn.ground_truth) == small_graph.number_of_nodes()

    def test_income_function_required_hops_attribute(self, small_graph):
        from phase2_approximation.synthetic_tasks import create_synthetic_income_function
        fn = create_synthetic_income_function(small_graph, num_hops=4, seed=42)
        assert fn.required_hops == 4

    def test_income_function_transit_signals_keys(self, small_graph):
        from phase2_approximation.synthetic_tasks import create_synthetic_income_function
        fn = create_synthetic_income_function(small_graph, num_hops=3, seed=42)
        # transit_signals should have keys 1..num_hops
        assert set(fn.transit_signals.keys()) == {1, 2, 3}

    def test_income_function_reproducible(self, small_graph):
        from phase2_approximation.synthetic_tasks import create_synthetic_income_function
        fn1 = create_synthetic_income_function(small_graph, num_hops=3, seed=42)
        fn2 = create_synthetic_income_function(small_graph, num_hops=3, seed=42)
        np.testing.assert_array_equal(fn1.ground_truth, fn2.ground_truth)

    def test_income_function_different_seeds_differ(self, small_graph):
        from phase2_approximation.synthetic_tasks import create_synthetic_income_function
        fn1 = create_synthetic_income_function(small_graph, num_hops=3, seed=42)
        fn2 = create_synthetic_income_function(small_graph, num_hops=3, seed=99)
        assert not np.array_equal(fn1.ground_truth, fn2.ground_truth)

    def test_evaluate_gnn_returns_required_keys(self, small_graph):
        """Evaluate with a trivial linear model to test infrastructure."""
        import torch
        import torch.nn as nn
        from phase2_approximation.synthetic_tasks import (
            create_synthetic_income_function,
            evaluate_gnn_on_synthetic_task,
        )

        income_fn = create_synthetic_income_function(small_graph, num_hops=2, seed=42)

        # Trivial 1-layer linear model
        class TrivialGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 1)
            def forward(self, x, edge_index):
                return self.lin(x)

        model = TrivialGNN()
        result = evaluate_gnn_on_synthetic_task(
            model, small_graph, income_fn,
            num_hops_required=2, num_epochs=5, device="cpu",
        )
        required = {"test_mse", "test_mae", "success", "num_hops_required",
                    "predicted", "ground_truth"}
        assert required.issubset(set(result.keys()))

    def test_evaluate_gnn_mae_is_non_negative(self, small_graph):
        import torch.nn as nn
        from phase2_approximation.synthetic_tasks import (
            create_synthetic_income_function,
            evaluate_gnn_on_synthetic_task,
        )

        income_fn = create_synthetic_income_function(small_graph, num_hops=2, seed=42)

        class TrivialGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 1)
            def forward(self, x, edge_index):
                return self.lin(x)

        result = evaluate_gnn_on_synthetic_task(
            TrivialGNN(), small_graph, income_fn,
            num_hops_required=2, num_epochs=5,
        )
        assert result["test_mae"] >= 0
        assert result["test_mse"] >= 0


# ── test_fetch_streetview.py ──────────────────────────────────────────────────

class TestFetchStreetview:
    """Tests for data_pipeline.fetch_streetview — no network calls."""

    def test_sample_points_returns_correct_count(self):
        from data_pipeline.fetch_streetview import sample_points_in_tract
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        points = sample_points_in_tract(poly, n_points=5, seed=42)
        assert len(points) <= 5  # may be fewer if random hits don't land in polygon
        for lat, lon in points:
            assert 0 <= lat <= 1
            assert 0 <= lon <= 1

    def test_coverage_summary_runs(self, tmp_path):
        from data_pipeline.fetch_streetview import _log_coverage_summary
        df = pd.DataFrame({
            "tract_id": ["A", "B", "C"],
            "coverage_flag": ["full", "sparse", "missing"],
        })
        _log_coverage_summary(df)  # Should not raise

    def test_image_record_dataclass(self):
        from data_pipeline.fetch_streetview import ImageRecord
        record = ImageRecord(
            tract_id="17031001000",
            image_index=0,
            heading=90,
            filename="17031001000_0000_h90.jpg",
            fetch_date="2026-01-01",
            status="ok",
            pano_id="abc123",
        )
        assert record.tract_id == "17031001000"
        assert record.heading == 90


# ── test_fetch_census.py ──────────────────────────────────────────────────────

class TestFetchCensus:
    """Tests for data_pipeline.fetch_census — logic only, no API calls."""

    def test_clean_acs_creates_tract_id(self):
        from data_pipeline.fetch_census import _clean_acs
        raw = pd.DataFrame({
            "NAME": ["Census Tract 1, Cook County, Illinois"],
            "B19013_001E": ["75000"],
            "B01003_001E": ["5000"],
            "B25001_001E": ["2100"],
            "B08201_001E": ["1000"],
            "B08201_002E": ["150"],
            "B17001_001E": ["4800"],
            "B17001_002E": ["384"],
            "B15003_001E": ["3500"],
            "B15003_022E": ["700"],
            "B15003_023E": ["350"],
            "B15003_025E": ["100"],
            "state": ["17"],
            "county": ["031"],
            "tract": ["000100"],
        })
        result = _clean_acs(raw, year=2022)
        assert "tract_id" in result.columns
        assert result["tract_id"].iloc[0] == "17031000100"

    def test_clean_acs_sentinel_to_nan(self):
        from data_pipeline.fetch_census import _clean_acs
        raw = pd.DataFrame({
            "NAME": ["Tract X"],
            "B19013_001E": ["-666666666"],  # Census sentinel for missing
            "B01003_001E": ["5000"],
            "B25001_001E": ["2100"],
            "B08201_001E": ["1000"],
            "B08201_002E": ["0"],
            "B17001_001E": ["4000"],
            "B17001_002E": ["0"],
            "B15003_001E": ["3500"],
            "B15003_022E": ["0"],
            "B15003_023E": ["0"],
            "B15003_025E": ["0"],
            "state": ["17"],
            "county": ["031"],
            "tract": ["000200"],
        })
        result = _clean_acs(raw, year=2022)
        assert pd.isna(result["median_household_income"].iloc[0]), \
            "Census sentinel (-666666666) must be converted to NaN"

    def test_clean_acs_poverty_rate_in_range(self):
        from data_pipeline.fetch_census import _clean_acs
        raw = pd.DataFrame({
            "NAME": ["Tract A"],
            "B19013_001E": ["55000"],
            "B01003_001E": ["4500"],
            "B25001_001E": ["1800"],
            "B08201_001E": ["900"],
            "B08201_002E": ["180"],
            "B17001_001E": ["4300"],
            "B17001_002E": ["516"],  # 12% poverty
            "B15003_001E": ["3200"],
            "B15003_022E": ["640"],
            "B15003_023E": ["320"],
            "B15003_025E": ["96"],
            "state": ["17"],
            "county": ["031"],
            "tract": ["000300"],
        })
        result = _clean_acs(raw, year=2022)
        rate = result["poverty_rate"].iloc[0]
        assert 0 <= rate <= 1, f"Poverty rate {rate} out of [0, 1]"

    def test_audit_missing_reports_all_columns(self, sample_acs_df):
        from data_pipeline.fetch_census import _audit_missing
        # Inject some NaN
        sample_acs_df.loc[0, "median_household_income"] = np.nan
        audit = _audit_missing(sample_acs_df)
        assert "column" in audit.columns
        assert "pct_missing" in audit.columns
        assert "bias_flag" in audit.columns
        # median_household_income has 20% missing → should be HIGH or MODERATE
        income_row = audit[audit["column"] == "median_household_income"]
        assert not income_row.empty
        assert income_row["pct_missing"].iloc[0] == pytest.approx(20.0)


# ── test_config.py ────────────────────────────────────────────────────────────

class TestConfig:
    """Validate config.yaml is well-formed and contains required keys."""

    @pytest.fixture(autouse=True)
    def load_config(self):
        import yaml
        with open("config.yaml") as f:
            self.config = yaml.safe_load(f)

    def test_required_top_level_keys(self):
        required = {"region", "paths", "census", "streetview", "osm", "gtfs",
                    "transit", "features", "privacy", "gnn", "synthetic", "reproducibility"}
        missing = required - set(self.config.keys())
        assert not missing, f"config.yaml missing keys: {missing}"

    def test_bbox_valid(self):
        bbox = self.config["region"]["bbox"]
        assert bbox["north"] > bbox["south"], "north must be greater than south"
        assert bbox["east"] > bbox["west"], "east must be greater than west"
        assert -90 <= bbox["south"] <= 90
        assert -90 <= bbox["north"] <= 90
        assert -180 <= bbox["west"] <= 180
        assert -180 <= bbox["east"] <= 180

    def test_gnn_depths_list(self):
        depths = self.config["gnn"]["depths"]
        assert isinstance(depths, list)
        assert all(isinstance(d, int) and d > 0 for d in depths)
        assert depths == sorted(depths), "GNN depths must be in ascending order"

    def test_transit_score_weights_sum_to_100(self):
        w = self.config["transit"]["transit_score_weights"]
        total = sum(w.values())
        assert total == 100, f"Transit score weights sum to {total}, expected 100"

    def test_imagenet_stats_correct_length(self):
        mean = self.config["features"]["imagenet_mean"]
        std = self.config["features"]["imagenet_std"]
        assert len(mean) == 3, "ImageNet mean must have 3 channels"
        assert len(std) == 3, "ImageNet std must have 3 channels"

    def test_privacy_pii_patterns_are_valid_regex(self):
        import re
        patterns = self.config["privacy"]["pii_column_patterns"]
        for p in patterns:
            try:
                re.compile(p)
            except re.error as e:
                pytest.fail(f"Invalid regex in privacy.pii_column_patterns: '{p}' — {e}")

    def test_random_seeds_consistent(self):
        cfg = self.config["reproducibility"]
        assert cfg["global_seed"] == cfg["numpy_seed"] == cfg["torch_seed"], \
            "All random seeds must be identical for reproducibility"
