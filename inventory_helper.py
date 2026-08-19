"""Stock alerting and stockout cost estimation."""

import io

import pandas as pd
import streamlit as st
from openai import OpenAI

import config

# Retail prices are the listed prices from reidsdistillery.com. Items at zero
# stock are the ones the store showed as sold out when this was written.
SAMPLE_CSV = """item,category,type,quantity_on_hand,reorder_point,unit,unit_price,last_restocked
Juniper Berries,Botanical,Production,46,60,kg,,2026-07-28
Coriander Seed,Botanical,Production,31,20,kg,,2026-08-05
Angelica Root,Botanical,Production,9,12,kg,,2026-07-14
Orris Root,Botanical,Production,7,6,kg,,2026-07-14
Cardamom Pods,Botanical,Production,11,8,kg,,2026-08-02
Cassia Bark,Botanical,Production,4,8,kg,,2026-06-30
Fresh Citrus Peel (Lemon/Orange),Botanical,Production,18,25,kg,,2026-08-12
Neutral Grain Spirit,Spirit,Production,742,400,litres,,2026-08-08
Bottles 750ml (Flint),Packaging,Production,1840,800,units,,2026-08-10
Bottles 50ml (Mini),Packaging,Production,610,400,units,,2026-08-04
Labels - Signature Gin,Packaging,Production,2450,1000,units,,2026-08-01
Labels - Citrus Gin,Packaging,Production,780,1000,units,,2026-07-15
Labels - Spiced Gin,Packaging,Production,320,1000,units,,2026-07-09
Labels - Navy Strength Gin,Packaging,Production,140,600,units,,2026-06-24
Labels - Vodka,Packaging,Production,1100,600,units,,2026-08-02
Corks & Closures,Packaging,Production,1560,900,units,,2026-08-06
Reid's Signature Gin 750ml,Spirits,Retail,168,60,units,37.00,2026-08-14
Reid's Navy Strength Gin 750ml,Spirits,Retail,22,30,units,59.00,2026-07-21
Reid's Vodka 750ml,Spirits,Retail,74,30,units,30.00,2026-08-14
Reid's Negroni Ready to Serve,RTD,Retail,41,24,units,23.00,2026-08-11
French 75 Social Pack,Cocktail Kit,Retail,26,20,units,92.00,2026-08-09
Fever Tree Premium Tonic 4pk,Mixer,Retail,0,48,units,7.00,2026-07-19
Fever Tree Mediterranean Tonic 4pk,Mixer,Retail,66,48,units,7.00,2026-08-11
1642 Canadian Ginger Beer 4pk,Mixer,Retail,0,36,units,7.00,2026-07-05
1642 Pink Grapefruit Tonic 4pk,Mixer,Retail,54,36,units,7.00,2026-08-09
Fresh Lime Juice Mix 2x200ml,Mixer,Retail,31,24,units,6.00,2026-08-13
Angostura Orange Bitters,Bitters,Retail,29,15,units,10.00,2026-08-03
Lavender Jasmine Simple Syrup,Syrup,Retail,22,18,units,14.00,2026-08-13
Seedlip Garden 108 (Non-Alc),Non-Alcoholic,Retail,15,12,units,34.00,2026-08-06
Seedlip Grove 42 (Non-Alc),Non-Alcoholic,Retail,0,12,units,34.00,2026-06-20
Seedlip Spice 94 (Non-Alc),Non-Alcoholic,Retail,0,12,units,34.00,2026-06-20
Dehydrated Lemon Slices,Garnish,Retail,0,24,units,7.00,2026-06-27
Dehydrated Grapefruit Slices,Garnish,Retail,0,24,units,7.00,2026-06-27
Dehydrated Lime Slices,Garnish,Retail,0,24,units,7.00,2026-07-03
Dehydrated Orange Slices,Garnish,Retail,0,24,units,7.00,2026-07-03
Copa Glass,Glassware,Retail,0,36,units,10.00,2026-07-02
Martini Glass,Glassware,Retail,0,36,units,10.00,2026-07-02
Coupe Glass,Glassware,Retail,0,36,units,10.00,2026-07-30
Reid's Rocks Glass,Glassware,Retail,58,36,units,5.00,2026-08-12
Jigger,Bar Tools,Retail,42,20,units,7.00,2026-08-07
Hawthorne Cocktail Strainer,Bar Tools,Retail,25,15,units,6.00,2026-08-07
Cocktail Muddler,Bar Tools,Retail,19,15,units,10.00,2026-07-26
Reid's Cocktail Shaker,Bar Tools,Retail,33,20,units,16.00,2026-08-12
12 Inch Ice Saw,Bar Tools,Retail,0,8,units,13.00,2026-06-10
Canadian Spirits Book,Merchandise,Retail,0,10,units,22.00,2026-05-28
Gift Boxes (Single Bottle),Packaging,Retail,240,300,units,45.00,2026-07-22
"""


def get_client():
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def load_data(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(io.StringIO(SAMPLE_CSV.strip()))


def classify(row) -> str:
    if row["quantity_on_hand"] == 0:
        return "Out of stock"
    if row["quantity_on_hand"] < row["reorder_point"] * 0.5:
        return "Critical"
    return "Low"


def compute_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Return items at or below their reorder point, worst first."""
    if "quantity_on_hand" not in df.columns or "reorder_point" not in df.columns:
        return pd.DataFrame()

    alerts = df[df["quantity_on_hand"] <= df["reorder_point"]].copy()
    if alerts.empty:
        return alerts

    alerts["status"] = alerts.apply(classify, axis=1)
    alerts["shortfall"] = alerts["reorder_point"] - alerts["quantity_on_hand"]

    sort_cols = ["quantity_on_hand"]
    if "type" in alerts.columns:
        # Production shortages stall a bottling run, so list those first.
        alerts["_priority"] = (alerts["type"] != "Production").astype(int)
        sort_cols = ["_priority", "quantity_on_hand"]

    alerts = alerts.sort_values(sort_cols)
    return alerts.drop(columns=["_priority"], errors="ignore")


def estimate_stockout_cost(
    df: pd.DataFrame, units_per_week: float, today: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Estimate shop revenue lost while retail items sit at zero stock.

    Price and elapsed time come from the data. The one assumption is
    units_per_week, which the UI exposes as a slider.

    Production items are excluded because a botanical shortage delays a bottling
    run rather than losing a countable sale.
    """
    required = {"quantity_on_hand", "unit_price", "last_restocked"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    today = today or pd.Timestamp.today().normalize()
    out = df[df["quantity_on_hand"] == 0].copy()
    if "type" in out.columns:
        out = out[out["type"] == "Retail"]
    out = out[out["unit_price"].notna()]
    if out.empty:
        return out

    restocked = pd.to_datetime(out["last_restocked"], errors="coerce")
    # Days since restock is an upper bound on the stockout window, since the item
    # may have sold through some days after arriving.
    out["days_since_restock"] = (today - restocked).dt.days.clip(lower=0)
    out["est_units_missed"] = (out["days_since_restock"] / 7 * units_per_week).round(1)
    out["est_revenue_lost"] = (out["est_units_missed"] * out["unit_price"]).round(2)

    cols = [
        "item", "category", "unit_price",
        "days_since_restock", "est_units_missed", "est_revenue_lost",
    ]
    return out[[c for c in cols if c in out.columns]].sort_values(
        "est_revenue_lost", ascending=False
    )


def generate_summary(
    df: pd.DataFrame, alerts: pd.DataFrame, costs: pd.DataFrame | None = None
) -> str:
    data_snippet = df.to_string(index=False, max_rows=50)
    alert_snippet = alerts.to_string(index=False) if not alerts.empty else "None"
    cost_snippet = (
        costs.to_string(index=False)
        if costs is not None and not costs.empty
        else "None"
    )

    response = get_client().chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an operations assistant for Reid's Distillery, a small "
                    "craft distillery in Toronto. They produce Signature, Citrus, "
                    "Spiced and Navy Strength Gin plus Vodka, run a retail shop, and "
                    "host tours, cocktail classes, weddings and private events.\n\n"
                    "Summarise the inventory data and alerts below, separating:\n"
                    "- Production issues (botanicals, spirit, bottles, labels), which "
                    "stall a bottling run.\n"
                    "- Retail issues (mixers, glassware, garnishes, non-alcoholic), "
                    "which cost shop sales today and can leave a cocktail class short."
                    "\n\nLead with whichever is costing more. Where a revenue figure "
                    "is given, describe it as an estimate based on an assumed sales "
                    "rate, not as a known fact.\n\n"
                    "Give 3 to 5 bullet points the owner can act on today, citing "
                    "specific items and numbers. Be practical, not generic."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Inventory data:\n{data_snippet}\n\n"
                    f"Below reorder point:\n{alert_snippet}\n\n"
                    f"Estimated cost of current retail stockouts:\n{cost_snippet}\n\n"
                    "Provide an operational summary and recommended actions."
                ),
            },
        ],
        temperature=0.3,
        max_tokens=700,
    )
    return response.choices[0].message.content


def render():
    st.header("Inventory")
    st.write(
        "Tracks botanicals and bottling supplies for the gin and vodka lines "
        "alongside retail shop stock, flags what is short, and estimates what the "
        "current stockouts are costing."
    )

    uploaded = st.file_uploader("Upload inventory CSV (optional)", type=["csv"])
    use_sample = st.checkbox("Use sample inventory", value=uploaded is None)

    df = load_data(None if use_sample else uploaded)
    alerts = compute_alerts(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Items tracked", len(df))
    col2.metric("Below reorder point", len(alerts))
    col3.metric("Out of stock", int((df["quantity_on_hand"] == 0).sum()))

    if alerts.empty:
        st.success("All items are above their reorder points.")
    else:
        st.subheader("Needs attention")
        if "type" in alerts.columns:
            production = alerts[alerts["type"] == "Production"]
            retail = alerts[alerts["type"] != "Production"]
            if not production.empty:
                st.write("**Production.** Risks delaying a bottling run.")
                st.dataframe(production, width="stretch", hide_index=True)
            if not retail.empty:
                st.write("**Retail and classes.** Risks lost shop sales.")
                st.dataframe(retail, width="stretch", hide_index=True)
        else:
            st.dataframe(alerts, width="stretch", hide_index=True)

    with st.expander("Full inventory"):
        st.dataframe(df, width="stretch", hide_index=True)

    st.divider()

    st.subheader("Cost of current stockouts")
    units_per_week = st.slider(
        "Assumed units sold per week when an item is in stock",
        min_value=1.0, max_value=20.0, value=5.0, step=0.5,
        help=(
            "The only estimated input. Prices and dates come from the data. Set it "
            "to match your shop and the figures update."
        ),
    )
    costs = estimate_stockout_cost(df, units_per_week)

    if costs.empty:
        st.info("No retail items are currently out of stock.")
    else:
        total = costs["est_revenue_lost"].sum()
        st.metric(
            f"Estimated shop revenue missed across {len(costs)} out-of-stock items",
            f"${total:,.0f}",
        )
        st.dataframe(costs, width="stretch", hide_index=True)
        st.caption(
            "An estimate, not a measurement. Days are counted from the last restock, "
            "which is an upper bound on how long the item has been at zero. Replace "
            "the assumed sales rate above with your real figure to tighten it."
        )

    st.divider()

    if st.button("Generate summary and recommendations"):
        if not config.DEEPSEEK_API_KEY:
            st.error("OpenRouter API key not set. Add it to your .env file.")
            return
        with st.spinner("Analysing inventory..."):
            summary = generate_summary(df, alerts, costs)
        st.subheader("Operational summary")
        st.markdown(summary)
