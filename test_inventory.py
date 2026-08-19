"""Tests for the stock alerting rules and the stockout cost arithmetic.

Run with: python test_inventory.py
"""

import pandas as pd

from inventory_helper import (
    classify,
    compute_alerts,
    estimate_stockout_cost,
    load_data,
)

TODAY = pd.Timestamp("2026-08-19")


def test_sample_data_loads():
    df = load_data()
    assert len(df) > 30, f"expected a realistic catalogue, got {len(df)} rows"
    assert set(df["type"]) == {"Production", "Retail"}
    assert df["quantity_on_hand"].notna().all()
    assert df["item"].is_unique


def test_production_items_carry_no_retail_price():
    production = load_data().query("type == 'Production'")
    assert production["unit_price"].isna().all(), (
        "production inputs are costs, not retail items, so they carry no price"
    )


def test_retail_items_all_priced():
    retail = load_data().query("type == 'Retail'")
    assert retail["unit_price"].notna().all()
    assert (retail["unit_price"] > 0).all()


def test_classify_thresholds():
    assert classify({"quantity_on_hand": 0, "reorder_point": 10}) == "Out of stock"
    assert classify({"quantity_on_hand": 4, "reorder_point": 10}) == "Critical"
    assert classify({"quantity_on_hand": 9, "reorder_point": 10}) == "Low"


def test_only_items_at_or_below_reorder_point_are_flagged():
    df = load_data()
    alerts = compute_alerts(df)
    assert (alerts["quantity_on_hand"] <= alerts["reorder_point"]).all()

    healthy = df[df["quantity_on_hand"] > df["reorder_point"]]
    assert not set(healthy["item"]) & set(alerts["item"])


def test_production_shortages_are_listed_before_retail():
    types = list(compute_alerts(load_data())["type"])
    first_retail = types.index("Retail") if "Retail" in types else len(types)
    assert "Production" not in types[first_retail:], (
        "production shortages must all appear before retail ones"
    )


def test_shortfall_is_the_gap_to_the_reorder_point():
    alerts = compute_alerts(load_data())
    expected = alerts["reorder_point"] - alerts["quantity_on_hand"]
    assert (alerts["shortfall"] == expected).all()
    assert (alerts["shortfall"] >= 0).all()


def test_healthy_inventory_produces_no_alerts():
    df = pd.DataFrame({
        "item": ["Juniper Berries", "Jigger"],
        "type": ["Production", "Retail"],
        "quantity_on_hand": [500, 90],
        "reorder_point": [60, 20],
    })
    assert compute_alerts(df).empty


def test_missing_columns_degrade_gracefully():
    df = pd.DataFrame({"item": ["Juniper Berries"], "notes": ["n/a"]})
    assert compute_alerts(df).empty
    assert estimate_stockout_cost(df, units_per_week=5).empty


def test_stockout_cost_covers_only_zero_stock_retail():
    df = load_data()
    costs = estimate_stockout_cost(df, units_per_week=5, today=TODAY)

    expected = set(
        df[(df["quantity_on_hand"] == 0) & (df["type"] == "Retail")]["item"]
    )
    assert set(costs["item"]) == expected
    assert len(costs) > 0, "sample data should contain stockouts to price"


def test_stockout_cost_arithmetic():
    df = pd.DataFrame({
        "item": ["Copa Glass"],
        "category": ["Glassware"],
        "type": ["Retail"],
        "quantity_on_hand": [0],
        "reorder_point": [36],
        "unit_price": [10.0],
        "last_restocked": ["2026-08-05"],
    })
    row = estimate_stockout_cost(df, units_per_week=7, today=TODAY).iloc[0]

    assert row["days_since_restock"] == 14
    assert row["est_units_missed"] == 14.0  # 14 days is 2 weeks at 7 a week
    assert row["est_revenue_lost"] == 140.0


def test_stockout_cost_scales_with_the_assumption():
    df = load_data()
    low = estimate_stockout_cost(df, units_per_week=2, today=TODAY)
    high = estimate_stockout_cost(df, units_per_week=8, today=TODAY)
    assert high["est_revenue_lost"].sum() > low["est_revenue_lost"].sum()


def test_stockout_cost_ignores_items_that_are_in_stock():
    df = pd.DataFrame({
        "item": ["Jigger"],
        "category": ["Bar Tools"],
        "type": ["Retail"],
        "quantity_on_hand": [42],
        "reorder_point": [20],
        "unit_price": [7.0],
        "last_restocked": ["2026-01-01"],
    })
    assert estimate_stockout_cost(df, units_per_week=5, today=TODAY).empty


def test_future_restock_date_does_not_produce_negative_loss():
    df = pd.DataFrame({
        "item": ["Martini Glass"],
        "category": ["Glassware"],
        "type": ["Retail"],
        "quantity_on_hand": [0],
        "reorder_point": [36],
        "unit_price": [10.0],
        "last_restocked": ["2026-12-01"],
    })
    costs = estimate_stockout_cost(df, units_per_week=5, today=TODAY)
    assert (costs["est_revenue_lost"] >= 0).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
