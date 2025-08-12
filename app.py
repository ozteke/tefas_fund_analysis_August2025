<<<<<<< HEAD
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import datetime as dt
import requests

# --------- Date helpers ---------
def _ymd(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")

def _n_days_ago(days: int) -> dt.date:
    return (dt.date.today() - dt.timedelta(days=days))

def _pct(cur, prev):
    try:
        cur = float(cur); prev = float(prev)
        if prev == 0 or any(map(lambda x: not np.isfinite(x), [cur, prev])): return np.nan
        return (cur/prev - 1.0) * 100.0
    except Exception:
        return np.nan

# --------- Gold (GoldAPI) ---------
@st.cache_data(ttl=300)
def goldapi_get_price(date_iso: str | None = None):
    """
    Returns price (USD/oz) using GoldAPI.
    If date_iso provided (YYYY-MM-DD), fetches historical; otherwise latest.
    """
    key = st.secrets.get("GOLDAPI_KEY", "")
    if not key:
        raise RuntimeError("Missing GOLDAPI_KEY in secrets.toml")

    base = "https://www.goldapi.io/api/XAU/USD"
    url = f"{base}/{date_iso}" if date_iso else base
    headers = {
        "x-access-token": key,                 # required
        "Content-Type": "application/json",
        "User-Agent": "streamlit-tefas/1.0"   # some providers reject no UA
    }

    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 403:
        # Common causes: wrong/disabled key, IP/rate-limit, plan restrictions
        raise RuntimeError("GoldAPI 403 Forbidden — check that your key is correct/active and not over the plan limits.")
    r.raise_for_status()

    data = r.json()
    price = data.get("price")
    if price is None:
        items = data.get("data")
        if isinstance(items, list) and items:
            price = items[0].get("price")
    if price is None:
        raise RuntimeError(f"GoldAPI response did not include 'price': {data}")
    return float(price)

def goldapi_get_price_near(target: dt.date, max_back_days: int = 5):
    """Try target date, then walk back a few days (weekends/holidays)."""
    for i in range(max_back_days + 1):
        d = target - dt.timedelta(days=i)
        try:
            p = goldapi_get_price(_ymd(d))
            if p is not None: return d, p
        except Exception:
            continue
    return None, None

# --------- USD/TRY (exchangeratesapi.io OR fallback exchangerate.host) ---------
@st.cache_data(ttl=300)
def fx_latest_usdtry():
    base = (st.secrets.get("EXCHANGE_API_BASE") or "").strip()
    key  = (st.secrets.get("EXCHANGE_API_KEY") or "").strip()

    use_paid = base and key and "YOUR_EXCHANGE" not in key

    if use_paid:
        # Most plans: base is EUR only. Request USD & TRY and compute USD/TRY = TRY / USD.
        url = f"{base}/latest"
        params = {"access_key": key, "symbols": "USD,TRY"}   # no base=USD here
    else:
        # Free fallback (supports base): exchangerate.host
        url = "https://api.exchangerate.host/latest"
        params = {"base": "USD", "symbols": "TRY"}

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    rates = js.get("rates", {}) or {}

    if use_paid:
        eur_try = rates.get("TRY")
        eur_usd = rates.get("USD")
        if eur_try is None or eur_usd in (None, 0):
            return None
        return float(eur_try) / float(eur_usd)  # USD/TRY
    else:
        return float(rates.get("TRY")) if rates.get("TRY") is not None else None


@st.cache_data(ttl=300)
def fx_usdtry_at(date_iso: str):
    base = (st.secrets.get("EXCHANGE_API_BASE") or "").strip()
    key  = (st.secrets.get("EXCHANGE_API_KEY") or "").strip()

    use_paid = base and key and "YOUR_EXCHANGE" not in key

    if use_paid:
        url = f"{base}/{date_iso}"
        params = {"access_key": key, "symbols": "USD,TRY"}   # EUR base implied
    else:
        url = f"https://api.exchangerate.host/{date_iso}"
        params = {"base": "USD", "symbols": "TRY"}

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    rates = js.get("rates", {}) or {}

    if use_paid:
        eur_try = rates.get("TRY")
        eur_usd = rates.get("USD")
        if eur_try is None or eur_usd in (None, 0):
            return None
        return float(eur_try) / float(eur_usd)
    else:
        return float(rates.get("TRY")) if rates.get("TRY") is not None else None


def fx_usdtry_near(target: dt.date, max_back_days: int = 5):
    for i in range(max_back_days + 1):
        d = target - dt.timedelta(days=i)
        try:
            v = fx_usdtry_at(_ymd(d))
            if v is not None:
                return d, v
        except Exception:
            continue
    return None, None


st.set_page_config(page_title="TEFAS Funds EDA — Aug 2025", page_icon="📊", layout="wide")

def signature_footer():
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("© Burak Ozteke, 2025. All rights reserved.")

@st.cache_data
def load_funds(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # mild typing to avoid stray strings in numeric cols
    for c in ["annual_net_return","1Y_return","sharpe","volatility","fee"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data
def load_gold(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

@st.cache_data
def load_tufe(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%m-%Y", errors="coerce")
    return df

# ---------- Load data ----------
FUNDS = load_funds("tefas_funds_aug25.csv")
#GOLD = load_gold("gold_prices_aug25.csv")
TUFE = load_tufe("tufe_aug25.csv")

# ---------- Navigation ----------
PAGES = [
    "Overview",
    "Explore Funds",
    "Compare Funds",
    "Risk/Return Analytics",
    "Live Markets (Gold & USD)",
    "Inflation (TUFE)",
    "Author Picks — Top by Risk",
    "Methodology & Notes"
]
page = st.sidebar.radio("Navigate", PAGES)

# ---------- Filters ----------
# fallbacks if a column is missing
type_opts = FUNDS["fund_type"].dropna().unique().tolist() if "fund_type" in FUNDS.columns else []
sel_type = st.sidebar.multiselect("Fund type", type_opts)

search_text = st.sidebar.text_input("Search fund code/name").strip().lower()

def apply_filters(df):
    out = df.copy()
    if sel_type and "fund_type" in out.columns:
        out = out[out["fund_type"].isin(sel_type)]
    if search_text:
        code_ok = out["fund_code"].astype(str).str.lower().str.contains(search_text, na=False) if "fund_code" in out.columns else False
        name_ok = out["fund_name"].astype(str).str.lower().str.contains(search_text, na=False) if "fund_name" in out.columns else False
        if isinstance(code_ok, pd.Series) and isinstance(name_ok, pd.Series):
            out = out[code_ok | name_ok]
        elif isinstance(code_ok, pd.Series):
            out = out[code_ok]
        elif isinstance(name_ok, pd.Series):
            out = out[name_ok]
    return out

FILTERED = apply_filters(FUNDS)

# ---- Real/Nominal toggle only for non-Overview pages ----
if page != "Overview":
    show_real = st.sidebar.checkbox("Show inflation-adjusted returns", value=False)
else:
    show_real = False  # hidden on Overview

# Build VIEW based on the toggle (or plain copy on Overview)
base_return_col = (
    "annual_net_return" if "annual_net_return" in FILTERED.columns
    else ("1Y_return" if "1Y_return" in FILTERED.columns else None)
)

if show_real and "real_1y_return" in FILTERED.columns:
    VIEW = FILTERED.copy()
    if base_return_col:
        VIEW[base_return_col] = pd.to_numeric(VIEW["real_1y_return"], errors="coerce")
        ret_col = base_return_col
        ret_label = ("Annual" if base_return_col == "annual_net_return" else "1Y") + " Return (%) — REAL"
    else:
        ret_col = "real_1y_return"
        ret_label = "Return (%) — REAL (1Y)"
else:
    VIEW = FILTERED.copy()
    ret_col = base_return_col
    ret_label = ("Annual" if base_return_col == "annual_net_return" else "1Y") + " Return (%)" if base_return_col else "Return (%)"


# ---------- Overview Page ----------
if page == "Overview":
    st.title("TEFAS Funds EDA — August 2025")

    # Top KPIs (no volatility, no mode caption)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total funds", f"{len(FILTERED):,}")
    with k2:
        if "annual_net_return" in FILTERED.columns and FILTERED["annual_net_return"].notna().any():
            st.metric("Annual Return (%)", f"{FILTERED['annual_net_return'].median():.2f}")
        elif "1Y_return" in FILTERED.columns and FILTERED["1Y_return"].notna().any():
            st.metric("1Y Return (%)", f"{FILTERED['1Y_return'].median():.2f}")
        else:
            st.metric("Return (%)", "—")
    with k3:
        if "sharpe" in FILTERED.columns and FILTERED["sharpe"].notna().any():
            st.metric("Median Sharpe", f"{FILTERED['sharpe'].median():.2f}")
        else:
            st.metric("Median Sharpe", "—")

    # Only keep Fund Type Distribution
    st.subheader("Fund Type Distribution")
    if "fund_type" in FILTERED.columns:
        fig = px.histogram(FILTERED, x="fund_type")
        fig.update_layout(height=380, bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Column 'fund_type' not found in the dataset.")

    # Preview table + download
    st.subheader("Current filters preview")
    st.caption(f"{len(FILTERED):,} / {len(FUNDS):,} funds match")

    st.dataframe(FILTERED, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered CSV",
        FILTERED.to_csv(index=False).encode("utf-8"),
        file_name="filtered_funds.csv",
        mime="text/csv"
    )

    signature_footer()

elif page == "Explore Funds":
    st.title("Explore Funds")
    st.caption(f"Mode: {'Real' if show_real and 'real_1y_return' in FILTERED.columns else 'Nominal'} • "
               f"{len(VIEW):,} rows shown")

    # ---- Column selection (hide helper cols) ----
    all_cols = [c for c in VIEW.columns if c not in ["search_key"]]
    # sensible defaults, only keep if present
    default_cols_order = [
        "fund_code", "fund_name", "fund_type", "spk_risk_level", "fund_size_in_M",
        ret_col if ret_col else "annual_net_return", "1Y_return",  # return(s)
        "sharpe", "volatility", "max_loss", "annual_fee", "annual_max_total_expense_ratio"
    ]
    default_cols = [c for c in default_cols_order if c in all_cols]
    show_cols = st.multiselect("Columns to show", all_cols, default=default_cols)

    # ---- Ranking controls ----
    rank_cols = [c for c in ["sharpe", ret_col, "annual_net_return", "1Y_return", "volatility", "max_loss", "fund_size_in_M", "annual_fee"] if c in VIEW.columns]
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        sort_by = st.selectbox("Sort by", rank_cols, index=(0 if "sharpe" in rank_cols else 0))
    with c2:
        ascending = st.toggle("Ascending", value=False, help="Turn on for smallest-to-largest")
    with c3:
        top_n = st.slider("Show top N", min_value=10, max_value=min(1000, len(VIEW)), value=min(100, len(VIEW)))

    # ---- Apply sort & head ----
    df_sorted = VIEW.sort_values(by=sort_by, ascending=ascending, kind="mergesort")  # stable sort for ties
    df_shown = df_sorted.head(top_n)[show_cols] if show_cols else df_sorted.head(top_n)

    # ---- Pretty column labels / formats ----
    cfg = {}
    def numcol(title, fmt="%.2f"): return st.column_config.NumberColumn(title, format=fmt)

    label_map = {
        "fund_code": "Fund Code",
        "fund_name": "Fund Name",
        "fund_type": "Fund Type",
        "spk_risk_level": "SPK Risk",
        "fund_size_in_M": "AUM (Million TL)",
        "1Y_return": "1Y Return (%)",
        "annual_net_return": "Annual Return (%)",
        "sharpe": "Sharpe",
        "volatility": "Volatility",
        "max_loss": "Max Loss (%)",
        "annual_fee": "Annual Fee (%)",
        "annual_max_total_expense_ratio": "Max TER (%)",
    }
    # Use dynamic label for the active return column
    if ret_col and ret_col in df_shown.columns:
        label_map[ret_col] = ("Annual" if ret_col == "annual_net_return" else "1Y") + (" Return (%) — REAL" if (show_real and "real_1y_return" in FILTERED.columns) else " Return (%)")

    for c in df_shown.columns:
        if c in ["sharpe","volatility","max_loss","annual_fee","annual_max_total_expense_ratio","1Y_return","annual_net_return"]:
            cfg[c] = numcol(label_map.get(c, c), "%.2f")
        elif c == "fund_size_in_M":
            cfg[c] = numcol(label_map.get(c, c), "%.2f")
        else:
            # string/other columns – just rename
            cfg[c] = label_map.get(c, c)

    # ---- Table + download ----
    st.dataframe(df_shown, use_container_width=True, hide_index=True, column_config=cfg)

    csv = df_shown.to_csv(index=False).encode("utf-8")
    st.download_button("Download these rows (CSV)", csv, file_name="explore_funds.csv", mime="text/csv")

    signature_footer()


elif page == "Compare Funds":
    st.title("Compare Funds (A vs B vs C)")
    st.caption(f"Mode: {'Real' if show_real and 'real_1y_return' in FILTERED.columns else 'Nominal'}")

    # Pick identifier (code preferred, fallback to name)
    id_col = "fund_code" if "fund_code" in VIEW.columns else ("fund_name" if "fund_name" in VIEW.columns else None)
    if id_col is None:
        st.warning("Need either 'fund_code' or 'fund_name' to compare.")
        signature_footer()
        st.stop()

    options = VIEW[id_col].dropna().astype(str).unique().tolist()
    chosen = st.multiselect("Select up to 3 funds", options, max_selections=3)
    if not chosen:
        st.info("Pick funds from the dropdown to compare.")
        signature_footer()
        st.stop()

    dfc = VIEW[VIEW[id_col].astype(str).isin(chosen)].copy()

    # ----- Metrics to compare -----
    # Active return column & pretty label already computed earlier: ret_col, ret_label
    metrics = []
    metric_labels = {}

    def add_metric(col, label):
        if col in dfc.columns:
            metrics.append(col)
            metric_labels[col] = label

    # Order matters
    add_metric(ret_col if ret_col else "annual_net_return", ret_label if ret_col else "Annual Return (%)")
    add_metric("1Y_return", "1Y Return (%)")
    add_metric("sharpe", "Sharpe")
    add_metric("volatility", "Volatility")
    add_metric("max_loss", "Max Loss (%)")
    add_metric("annual_fee", "Annual Fee (%)")
    add_metric("annual_max_total_expense_ratio", "Max TER (%)")
    add_metric("fund_size_in_M", "AUM (Million TL)")

    if len(metrics) == 0:
        st.warning("No comparable numeric metrics found.")
        signature_footer()
        st.stop()

    # ----- Headline table (compact) -----
    st.subheader("Snapshot")
    cols_to_show = [id_col, "fund_name"] if id_col == "fund_code" and "fund_name" in dfc.columns else [id_col]
    cols_to_show += [m for m in metrics if m not in cols_to_show]
    st.dataframe(dfc[cols_to_show], use_container_width=True, hide_index=True)

    # ----- Grouped bar chart -----
    st.subheader("Metric comparison")
    fig = go.Figure()
    for _, r in dfc.iterrows():
        y_vals = [r[m] if pd.notna(r[m]) else None for m in metrics]
        fig.add_trace(go.Bar(
            name=str(r[id_col]),
            x=[metric_labels[m] for m in metrics],
            y=y_vals
        ))
    fig.update_layout(barmode="group", height=520, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    # ----- Radar chart (normalized 0..1) -----
    # Use only numeric columns that exist across the full VIEW for stable min-max
    st.subheader("Radar (normalized)")
    # compute min/max on the overall VIEW for consistency
    norm_df = dfc[[id_col] + metrics].copy()
    mins = {m: pd.to_numeric(VIEW[m], errors="coerce").min() if m in VIEW.columns else np.nan for m in metrics}
    maxs = {m: pd.to_numeric(VIEW[m], errors="coerce").max() if m in VIEW.columns else np.nan for m in metrics}

    # avoid zero-range issues
    for m in metrics:
        lo, hi = mins[m], maxs[m]
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            # fall back to the dfc spread or constant 0.5 if still invalid
            series = pd.to_numeric(dfc[m], errors="coerce")
            lo2, hi2 = series.min(), series.max()
            if pd.isna(lo2) or pd.isna(hi2) or hi2 == lo2:
                mins[m], maxs[m] = 0.0, 1.0
            else:
                mins[m], maxs[m] = lo2, hi2

    radar = go.Figure()
    theta = [metric_labels[m] for m in metrics]
    for _, r in dfc.iterrows():
        values = []
        for m in metrics:
            v = pd.to_numeric(r[m], errors="coerce")
            lo, hi = mins[m], maxs[m]
            if pd.isna(v) or pd.isna(lo) or pd.isna(hi) or hi == lo:
                values.append(0.5)
            else:
                values.append(float((v - lo) / (hi - lo)))
        radar.add_trace(go.Scatterpolar(
            r=values, theta=theta, fill="toself", name=str(r[id_col])
        ))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=560)
    st.plotly_chart(radar, use_container_width=True)

    signature_footer()

elif page == "Risk/Return Analytics":
    st.title("Risk / Return Analytics")
    st.caption(f"Mode: {'Real' if show_real and 'real_1y_return' in FILTERED.columns else 'Nominal'}")

    # --- Guards ---
    y_metric = ret_col if ret_col and ret_col in VIEW.columns else None
    default_x = "volatility" if "volatility" in VIEW.columns else None

    numeric_candidates = [c for c in VIEW.columns if pd.api.types.is_numeric_dtype(VIEW[c])]
    if y_metric is None:
        st.warning("No return column available. Need 'annual_net_return' or '1Y_return' (or a real version).")
        signature_footer(); st.stop()
    if default_x is None:
        default_x = next((c for c in numeric_candidates if c != y_metric), None)
    if default_x is None:
        st.warning("No numeric columns to plot on X axis.")
        signature_footer(); st.stop()

    # --- Controls ---
    st.subheader("Controls")
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1, 1])
    with c1:
        x_metric = st.selectbox(
            "X axis",
            [default_x] + [c for c in numeric_candidates if c not in [default_x, y_metric]],
            index=0
        )
    with c2:
        color_choices = [c for c in ["fund_type", "spk_risk_level", "company", "currency"] if c in VIEW.columns]
        color_by = st.selectbox("Color by", color_choices, index=0 if color_choices else None)
    with c3:
        size_choices = [c for c in ["fund_size_in_M"] + numeric_candidates if c in VIEW.columns]
        size_by = st.selectbox(
            "Bubble size",
            size_choices,
            index=(size_choices.index("fund_size_in_M") if "fund_size_in_M" in size_choices else 0)
        )
    with c4:
        log_x = st.toggle("Log scale (X)", value=False)

    # --- Build a clean plotting frame & safe size column ---
    df_plot = VIEW.copy()

    # Coerce axes to numeric and drop rows with NaNs on axes
    df_plot[x_metric] = pd.to_numeric(df_plot[x_metric], errors="coerce")
    df_plot[y_metric] = pd.to_numeric(df_plot[y_metric], errors="coerce")
    df_plot = df_plot.dropna(subset=[x_metric, y_metric])

    # Prepare size (only if valid; otherwise skip)
    size_arg = None
    if size_by in df_plot.columns:
        s = pd.to_numeric(df_plot[size_by], errors="coerce")
        if s.notna().sum() >= 3 and (s > 0).any():
            s = s.clip(lower=0)
            s = s.fillna(s.median() if s.notna().any() else 1)
            df_plot["_size"] = s
            size_arg = "_size"

    # --- Hover fields (keep only those present) ---
    hover_cols = [c for c in ["fund_code","fund_name","company","currency","sharpe","max_loss"]
                  if c in df_plot.columns]

    # --- Scatter plot ---
    fig = px.scatter(
        df_plot,
        x=x_metric,
        y=y_metric,
        color=color_by if color_by in df_plot.columns else None,
        size=size_arg,  # None if invalid → no crash
        hover_data=hover_cols,
        labels={y_metric: (ret_label if ret_col and y_metric == ret_col else y_metric)},
        height=640
    )
    if log_x:
        fig.update_xaxes(type="log")

    # Median guide lines
    try:
        x_med = pd.to_numeric(df_plot[x_metric], errors="coerce").median()
        y_med = pd.to_numeric(df_plot[y_metric], errors="coerce").median()
        if pd.notna(x_med): fig.add_vline(x=x_med, line_dash="dot", opacity=0.5)
        if pd.notna(y_med): fig.add_hline(y=y_med, line_dash="dot", opacity=0.5)
    except Exception:
        pass

    st.plotly_chart(fig, use_container_width=True)

    # --- Table of the plotted view (sorted by return desc by default) ---
    st.subheader("Data used in chart")

    # Build an ordered, de-duplicated column list
    cols_raw = ["fund_code", "fund_name", x_metric, y_metric, "sharpe", "volatility", "max_loss", "fund_size_in_M"]
    cols_table = []
    for c in cols_raw:
        if c in df_plot.columns and c not in cols_table:
            cols_table.append(c)

    shown = df_plot[cols_table].copy()
    
    

    # pretty label for y column
    col_cfg = {}
    if y_metric in shown.columns:
        col_cfg[y_metric] = st.column_config.NumberColumn(ret_label, format="%.2f")
        

    st.dataframe(
        shown.sort_values(by=y_metric, ascending=False) if y_metric in shown.columns else shown,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg
    )

    csv = shown.to_csv(index=False).encode("utf-8")
    st.download_button("Download plotted data (CSV)", csv, file_name="risk_return_view.csv", mime="text/csv")

    signature_footer()

elif page == "Live Markets (Gold & USD)":
    st.title("Live Markets (Gold & USD)")

    # -------- Gold (latest + 1M/6M/1Y) --------
    st.subheader("Gold (XAU/USD) — via GoldAPI")

    colg1, colg2 = st.columns([1.2, 1])
    with colg1:
        try:
            # latest
            latest_date = dt.date.today()
            latest_price = goldapi_get_price(None)  # latest endpoint

            # history anchors
            d1m, p1m = goldapi_get_price_near(_n_days_ago(30))
            d6m, p6m = goldapi_get_price_near(_n_days_ago(180))
            d1y, p1y = goldapi_get_price_near(_n_days_ago(365))

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Latest", f"{latest_price:.2f}" if latest_price else "—")
            with k2:
                st.metric("1M", f"{_pct(latest_price, p1m):.2f}%" if latest_price and p1m else "—")
            with k3:
                st.metric("6M", f"{_pct(latest_price, p6m):.2f}%" if latest_price and p6m else "—")
            with k4:
                st.metric("1Y", f"{_pct(latest_price, p1y):.2f}%" if latest_price and p1y else "—")

            st.caption(f"Latest fetched from GoldAPI. 1M/6M/1Y compare to: "
                       f"{_ymd(d1m) if d1m else 'n/a'}, {_ymd(d6m) if d6m else 'n/a'}, {_ymd(d1y) if d1y else 'n/a'}")

        except Exception as e:
            st.error("GoldAPI request failed. Check your GOLDAPI_KEY in secrets or network access.")
            st.exception(e)

    with colg2:
        st.info("Tip: On Streamlit Cloud, put `GOLDAPI_KEY` in **secrets.toml**. We fetch latest and compare to ~1M/6M/1Y ago (adjusted for weekends/holidays).")

    st.divider()

    # -------- USD/TRY (latest + 1M/6M/1Y) --------
    st.subheader("USD/TRY — Exchange Rates")

    colf1, colf2 = st.columns([1.2, 1])
    with colf1:
        try:
            fx_latest = fx_latest_usdtry()
            d1m, fx1m = fx_usdtry_near(_n_days_ago(30))
            d6m, fx6m = fx_usdtry_near(_n_days_ago(180))
            d1y, fx1y = fx_usdtry_near(_n_days_ago(365))

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Latest", f"{fx_latest:.4f}" if fx_latest else "—")
            with k2:
                st.metric("1M", f"{_pct(fx_latest, fx1m):.2f}%" if fx_latest and fx1m else "—")
            with k3:
                st.metric("6M", f"{_pct(fx_latest, fx6m):.2f}%" if fx_latest and fx6m else "—")
            with k4:
                st.metric("1Y", f"{_pct(fx_latest, fx1y):.2f}%" if fx_latest and fx1y else "—")

            if d1m or d6m or d1y:
                st.caption(f"Anchors: {_ymd(d1m) if d1m else 'n/a'}, {_ymd(d6m) if d6m else 'n/a'}, {_ymd(d1y) if d1y else 'n/a'}")
        except Exception as e:
            st.error("FX API request failed. If you have an exchangeratesapi.io key, set EXCHANGE_API_BASE & EXCHANGE_API_KEY in secrets. Otherwise we fallback to exchangerate.host.")
            st.exception(e)

    with colf2:
        st.info("If you have a paid key, set:\nEXCHANGE_API_BASE = 'https://api.exchangeratesapi.io/v1'\nEXCHANGE_API_KEY = '...' \nOtherwise we use exchangerate.host automatically.")


elif page == "Inflation (TUFE)":
    st.title("Turkey CPI (TUFE)")

    if TUFE is None or "date" not in TUFE.columns:
        st.warning("tufe_aug25.csv not loaded or missing 'date' column.")
        signature_footer(); st.stop()

    # --- prep ---
    t = TUFE.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"]).sort_values("date")
    for c in ["annual_change", "monthly_change"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    latest_dt = t["date"].max()
    latest_row = t.loc[t["date"] == latest_dt].iloc[0]

    yoy = float(latest_row["annual_change"]) if "annual_change" in t.columns and pd.notna(latest_row.get("annual_change")) else None
    mom = float(latest_row["monthly_change"]) if "monthly_change" in t.columns and pd.notna(latest_row.get("monthly_change")) else None

    # --- KPIs ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Latest (YoY)", f"{yoy:.2f}%" if yoy is not None else "—")
    with k2:
        st.metric("Latest (MoM)", f"{mom:.2f}%" if mom is not None else "—")
    with k3:
        # 3M avg MoM (annualized)
        if "monthly_change" in t.columns and t["monthly_change"].notna().any():
            last3 = t.tail(3)["monthly_change"].dropna()
            ann3 = (np.prod(1 + last3/100) - 1) * 100
            st.metric("3M Cum (MoM→ann.)", f"{ann3:.2f}%")
        else:
            st.metric("3M Cum (MoM→ann.)", "—")
    with k4:
        # YTD cumulative from Jan of current year
        if "monthly_change" in t.columns:
            year = latest_dt.year
            ytd = t[(t["date"].dt.year == year) & (t["date"] <= latest_dt)]["monthly_change"].dropna()
            ytd_val = (np.prod(1 + ytd/100) - 1) * 100 if len(ytd) else np.nan
            st.metric("YTD (cum MoM)", f"{ytd_val:.2f}%" if pd.notna(ytd_val) else "—")
        else:
            st.metric("YTD (cum MoM)", "—")

    st.caption(f"Last update: {latest_dt.date()}")

    # --- charts ---
    tab1, tab2, tab3 = st.tabs(["YoY %", "MoM %", "Index (base=100)"])

    with tab1:
        if "annual_change" in t.columns and t["annual_change"].notna().any():
            fig = px.line(t, x="date", y="annual_change", markers=True, title=None,
                          labels={"annual_change": "YoY (%)", "date": ""})
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'annual_change' column found.")

    with tab2:
        if "monthly_change" in t.columns and t["monthly_change"].notna().any():
            fig = px.bar(t, x="date", y="monthly_change",
                         labels={"monthly_change": "MoM (%)", "date": ""})
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'monthly_change' column found.")

    with tab3:
        if "monthly_change" in t.columns and t["monthly_change"].notna().any():
            tt = t[["date", "monthly_change"]].dropna().copy()
            tt["index"] = (1 + tt["monthly_change"]/100.0).cumprod() * 100.0

            # rebase option
            c1, c2 = st.columns([1,1])
            with c1:
                base_mode = st.selectbox("Rebase", ["First available", f"Start of {latest_dt.year}"], index=0)
            with c2:
                smooth = st.toggle("Show 12M SMA (on index)", value=False)

            if base_mode != "First available":
                jan = pd.Timestamp(year=latest_dt.year, month=1, day=1)
                base_row = tt[tt["date"] >= jan].head(1)
                if not base_row.empty:
                    base_val = float(base_row["index"].iloc[0])
                    tt["index"] = tt["index"] / base_val * 100.0

            fig = px.line(tt, x="date", y="index", labels={"index": "Index (base=100)", "date": ""})
            if smooth:
                s = tt["index"].rolling(12, min_periods=4).mean()
                fig.add_scatter(x=tt["date"], y=s, mode="lines", name="SMA 12")
            fig.update_layout(height=440)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(tt.tail(24), use_container_width=True, hide_index=True)
            st.download_button("Download index (CSV)", tt.to_csv(index=False).encode("utf-8"),
                               file_name="tufe_index.csv", mime="text/csv")
        else:
            st.info("Need 'monthly_change' to build an index.")

    signature_footer()
    
elif page == "Author Picks — Top by Risk":
    st.title("Author Picks — Top Funds by Risk")
    st.caption("Fixed methodology (independent of sidebar filters). Uses full dataset to keep results stable.")

    # ------- Guards -------
    required_cols = [
        "spk_risk_level","sharpe","max_loss","annual_fee","annual_net_return",
        "1Y_return","real_1y_return","fund_size_in_M","fund_code","fund_type"
    ]
    missing = [c for c in required_cols if c not in FUNDS.columns]
    if missing:
        st.warning(f"Missing columns for scoring: {', '.join(missing)}")
        signature_footer(); st.stop()

    df_base = FUNDS.copy()
    # coerce numerics
    for c in ["spk_risk_level","sharpe","max_loss","annual_fee","annual_net_return",
              "1Y_return","real_1y_return","fund_size_in_M"]:
        df_base[c] = pd.to_numeric(df_base[c], errors="coerce")

    # ------- Controls (optional tweaks) -------
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        top_n = st.number_input("Top N", min_value=5, max_value=50, value=15, step=5)
    with c2:
        min_sharpe = st.number_input("Sharpe >", value=1.0, step=0.1, format="%.1f")
    with c3:
        min_max_loss = st.number_input("Max Loss >", value=-20.0, step=1.0, format="%.1f")
    with c4:
        max_fee = st.number_input("Annual Fee ≤", value=2.0, step=0.1, format="%.1f")
    with c5:
        min_aum = st.number_input("AUM ≥ (M TL)", value=50.0, step=10.0, format="%.0f")

    st.caption("Filters: Sharpe > threshold • Max Loss > threshold • Fee ≤ threshold • Real 1Y Return > 0 • AUM ≥ threshold")

    # ------- Scoring helper (no scipy needed) -------
    def percentile_rank(s: pd.Series, invert: bool = False) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if invert:
            s = -s
        return (s.rank(pct=True, method="average") * 100.0)

    def get_top_funds_by_risk(df: pd.DataFrame, risk_levels, top_n=10) -> pd.DataFrame:
        # Step 1 – universe filter
        filtered = df[
            df["spk_risk_level"].isin(risk_levels) &
            (df["sharpe"] > min_sharpe) &
            (df["max_loss"] > min_max_loss) &
            (df["annual_fee"] <= max_fee) &
            (df["real_1y_return"] > 0) &
            (df["fund_size_in_M"] >= min_aum)
        ].dropna(subset=[
            "sharpe","max_loss","annual_fee","annual_net_return",
            "1Y_return","real_1y_return","fund_size_in_M"
        ]).copy()

        if filtered.empty:
            return filtered

        # Step 2 – percentile ranks
        filtered["1Y_return_score"]        = percentile_rank(filtered["1Y_return"])
        filtered["sharpe_score"]           = percentile_rank(filtered["sharpe"])
        filtered["max_loss_score"]         = percentile_rank(filtered["max_loss"], invert=True)   # less loss is better
        filtered["annual_fee_score"]       = percentile_rank(filtered["annual_fee"], invert=True) # lower fee is better
        filtered["annual_net_return_score"]= percentile_rank(filtered["annual_net_return"])

        # Step 3 – weighted final score
        filtered["final_score"] = (
            filtered["1Y_return_score"]         * 0.20 +
            filtered["sharpe_score"]            * 0.30 +
            filtered["max_loss_score"]          * 0.20 +
            filtered["annual_fee_score"]        * 0.15 +
            filtered["annual_net_return_score"] * 0.15
        ).round(2)

        # Step 4 – top N
        cols = [
            "fund_code","fund_type","fund_size_in_M","spk_risk_level",
            "1Y_return","real_1y_return","3Y_return","5Y_return",
            "sharpe","max_loss","annual_fee","final_score"
        ]
        cols = [c for c in cols if c in filtered.columns]
        return filtered.sort_values("final_score", ascending=False).head(int(top_n))[cols].reset_index(drop=True)

    # ------- Output: 3 tabs (Low / Medium / High) -------
    tab_low, tab_mid, tab_high = st.tabs(["Low risk (1–3)", "Medium risk (4–5)", "High risk (6–7)"])

    with tab_low:
        low = get_top_funds_by_risk(df_base, [1,2,3], top_n=top_n)
        if low.empty:
            st.info("No funds matched the criteria for Low risk.")
        else:
            st.subheader("Top picks — Low risk")
            st.dataframe(low, use_container_width=True, hide_index=True,
                         column_config={"final_score": st.column_config.NumberColumn("Final Score", format="%.2f")})
            st.download_button("Download Low risk picks (CSV)", low.to_csv(index=False).encode("utf-8"),
                               file_name="author_picks_low.csv", mime="text/csv")

    with tab_mid:
        mid = get_top_funds_by_risk(df_base, [4,5], top_n=top_n)
        if mid.empty:
            st.info("No funds matched the criteria for Medium risk.")
        else:
            st.subheader("Top picks — Medium risk")
            st.dataframe(mid, use_container_width=True, hide_index=True,
                         column_config={"final_score": st.column_config.NumberColumn("Final Score", format="%.2f")})
            st.download_button("Download Medium risk picks (CSV)", mid.to_csv(index=False).encode("utf-8"),
                               file_name="author_picks_medium.csv", mime="text/csv")

    with tab_high:
        high = get_top_funds_by_risk(df_base, [6,7], top_n=top_n)
        if high.empty:
            st.info("No funds matched the criteria for High risk.")
        else:
            st.subheader("Top picks — High risk")
            st.dataframe(high, use_container_width=True, hide_index=True,
                         column_config={"final_score": st.column_config.NumberColumn("Final Score", format="%.2f")})
            st.download_button("Download High risk picks (CSV)", high.to_csv(index=False).encode("utf-8"),
                               file_name="author_picks_high.csv", mime="text/csv")

    # small note about independence from sidebar filters
    st.caption("Note: Author picks are computed on the full dataset (not affected by sidebar filters).")
    signature_footer()


elif page == "Methodology & Notes":
    st.title("Methodology & Notes")
    
    st.markdown("""
    ### Data Sources
    - **TEFAS (Fund Analysis)** — August 2025 data downloaded from TEFAS.gov.tr  
    - **Fonbul.com** — Risk analysis and fund performance metrics  
    - **TCMB & Inflation Verileri** — TUFE inflation rates  
    - **Gold Prices** — Historical data scraped from [GoldAPI.io](https://www.goldapi.io)  
    - **USD/TRY** — [ExchangeRatesAPI.io](https://exchangeratesapi.io) (fallback: [ExchangeRate.host](https://exchangerate.host))

    ### Data Processing
    - All CSVs were merged and cleaned manually  
    - Removed duplicates and redundant columns  
    - Converted date fields to `datetime` format  
    - Standardized column names for consistency  
    - Inflation-adjusted returns calculated where possible (`real_1y_return`)

    ### Limitations
    - Dataset is **static** for now — live updates apply only to gold & USD/TRY prices  
    - Inflation adjustment based on **CPI (TUFE)** year-over-year changes  
    - Some risk metrics depend on provider calculations (Sharpe ratio, max loss, etc.)

    ### Usage
    - Use filters in the sidebar to search by **fund type**, **code**, **company**, etc.  
    - Switch between **Nominal** and **Inflation-adjusted** returns  
    - Compare funds side-by-side using charts  
    - View macro indicators on the **Live Markets** and **TUFE** pages

    ---
    © Burak Ozteke, 2025. All rights reserved.
    """)
=======
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import datetime as dt
import requests

# --------- Date helpers ---------
def _ymd(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")

def _n_days_ago(days: int) -> dt.date:
    return (dt.date.today() - dt.timedelta(days=days))

def _pct(cur, prev):
    try:
        cur = float(cur); prev = float(prev)
        if prev == 0 or any(map(lambda x: not np.isfinite(x), [cur, prev])): return np.nan
        return (cur/prev - 1.0) * 100.0
    except Exception:
        return np.nan

# --------- Gold (GoldAPI) ---------
@st.cache_data(ttl=300)
def goldapi_get_price(date_iso: str | None = None):
    """
    Returns price (USD/oz) using GoldAPI.
    If date_iso provided (YYYY-MM-DD), fetches historical; otherwise latest.
    """
    key = st.secrets.get("GOLDAPI_KEY", "")
    if not key:
        raise RuntimeError("Missing GOLDAPI_KEY in secrets.toml")

    base = "https://www.goldapi.io/api/XAU/USD"
    url = f"{base}/{date_iso}" if date_iso else base
    headers = {
        "x-access-token": key,                 # required
        "Content-Type": "application/json",
        "User-Agent": "streamlit-tefas/1.0"   # some providers reject no UA
    }

    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 403:
        # Common causes: wrong/disabled key, IP/rate-limit, plan restrictions
        raise RuntimeError("GoldAPI 403 Forbidden — check that your key is correct/active and not over the plan limits.")
    r.raise_for_status()

    data = r.json()
    price = data.get("price")
    if price is None:
        items = data.get("data")
        if isinstance(items, list) and items:
            price = items[0].get("price")
    if price is None:
        raise RuntimeError(f"GoldAPI response did not include 'price': {data}")
    return float(price)

def goldapi_get_price_near(target: dt.date, max_back_days: int = 5):
    """Try target date, then walk back a few days (weekends/holidays)."""
    for i in range(max_back_days + 1):
        d = target - dt.timedelta(days=i)
        try:
            p = goldapi_get_price(_ymd(d))
            if p is not None: return d, p
        except Exception:
            continue
    return None, None

# --------- USD/TRY (exchangeratesapi.io OR fallback exchangerate.host) ---------
@st.cache_data(ttl=300)
def fx_latest_usdtry():
    base = (st.secrets.get("EXCHANGE_API_BASE") or "").strip()
    key  = (st.secrets.get("EXCHANGE_API_KEY") or "").strip()

    use_paid = base and key and "YOUR_EXCHANGE" not in key

    if use_paid:
        # Most plans: base is EUR only. Request USD & TRY and compute USD/TRY = TRY / USD.
        url = f"{base}/latest"
        params = {"access_key": key, "symbols": "USD,TRY"}   # no base=USD here
    else:
        # Free fallback (supports base): exchangerate.host
        url = "https://api.exchangerate.host/latest"
        params = {"base": "USD", "symbols": "TRY"}

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    rates = js.get("rates", {}) or {}

    if use_paid:
        eur_try = rates.get("TRY")
        eur_usd = rates.get("USD")
        if eur_try is None or eur_usd in (None, 0):
            return None
        return float(eur_try) / float(eur_usd)  # USD/TRY
    else:
        return float(rates.get("TRY")) if rates.get("TRY") is not None else None


@st.cache_data(ttl=300)
def fx_usdtry_at(date_iso: str):
    base = (st.secrets.get("EXCHANGE_API_BASE") or "").strip()
    key  = (st.secrets.get("EXCHANGE_API_KEY") or "").strip()

    use_paid = base and key and "YOUR_EXCHANGE" not in key

    if use_paid:
        url = f"{base}/{date_iso}"
        params = {"access_key": key, "symbols": "USD,TRY"}   # EUR base implied
    else:
        url = f"https://api.exchangerate.host/{date_iso}"
        params = {"base": "USD", "symbols": "TRY"}

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    rates = js.get("rates", {}) or {}

    if use_paid:
        eur_try = rates.get("TRY")
        eur_usd = rates.get("USD")
        if eur_try is None or eur_usd in (None, 0):
            return None
        return float(eur_try) / float(eur_usd)
    else:
        return float(rates.get("TRY")) if rates.get("TRY") is not None else None


def fx_usdtry_near(target: dt.date, max_back_days: int = 5):
    for i in range(max_back_days + 1):
        d = target - dt.timedelta(days=i)
        try:
            v = fx_usdtry_at(_ymd(d))
            if v is not None:
                return d, v
        except Exception:
            continue
    return None, None


st.set_page_config(page_title="TEFAS Funds EDA — Aug 2025", page_icon="📊", layout="wide")

def signature_footer():
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("© Burak Ozteke, 2025. All rights reserved.")

@st.cache_data
def load_funds(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # mild typing to avoid stray strings in numeric cols
    for c in ["annual_net_return","1Y_return","sharpe","volatility","fee"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data
def load_gold(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

@st.cache_data
def load_tufe(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%m-%Y", errors="coerce")
    return df

# ---------- Load data ----------
FUNDS = load_funds("tefas_funds_aug25.csv")
#GOLD = load_gold("gold_prices_aug25.csv")
TUFE = load_tufe("tufe_aug25.csv")

# ---------- Navigation ----------
PAGES = [
    "Overview",
    "Explore Funds",
    "Compare Funds",
    "Risk/Return Analytics",
    "Live Markets (Gold & USD)",
    "Inflation (TUFE)",
    "Author Picks — Top by Risk",
    "Methodology & Notes"
]
page = st.sidebar.radio("Navigate", PAGES)

# ---------- Filters ----------
# fallbacks if a column is missing
type_opts = FUNDS["fund_type"].dropna().unique().tolist() if "fund_type" in FUNDS.columns else []
sel_type = st.sidebar.multiselect("Fund type", type_opts)

search_text = st.sidebar.text_input("Search fund code/name").strip().lower()

def apply_filters(df):
    out = df.copy()
    if sel_type and "fund_type" in out.columns:
        out = out[out["fund_type"].isin(sel_type)]
    if search_text:
        code_ok = out["fund_code"].astype(str).str.lower().str.contains(search_text, na=False) if "fund_code" in out.columns else False
        name_ok = out["fund_name"].astype(str).str.lower().str.contains(search_text, na=False) if "fund_name" in out.columns else False
        if isinstance(code_ok, pd.Series) and isinstance(name_ok, pd.Series):
            out = out[code_ok | name_ok]
        elif isinstance(code_ok, pd.Series):
            out = out[code_ok]
        elif isinstance(name_ok, pd.Series):
            out = out[name_ok]
    return out

FILTERED = apply_filters(FUNDS)

# ---- Always nominal mode (no toggle) ----
VIEW = FILTERED.copy()

base_return_col = (
    "annual_net_return" if "annual_net_return" in VIEW.columns
    else ("1Y_return" if "1Y_return" in VIEW.columns else None)
)
ret_col = base_return_col
ret_label = ("Annual" if ret_col == "annual_net_return" else "1Y") + " Return (%)" if ret_col else "Return (%)"


# ---------- Overview Page ----------
if page == "Overview":
    st.title("TEFAS Funds EDA — August 2025")

    # Top KPIs (no volatility, no mode caption)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total funds", f"{len(FILTERED):,}")
    with k2:
        if "annual_net_return" in FILTERED.columns and FILTERED["annual_net_return"].notna().any():
            st.metric("Annual Return (%)", f"{FILTERED['annual_net_return'].median():.2f}")
        elif "1Y_return" in FILTERED.columns and FILTERED["1Y_return"].notna().any():
            st.metric("1Y Return (%)", f"{FILTERED['1Y_return'].median():.2f}")
        else:
            st.metric("Return (%)", "—")
    with k3:
        if "sharpe" in FILTERED.columns and FILTERED["sharpe"].notna().any():
            st.metric("Median Sharpe", f"{FILTERED['sharpe'].median():.2f}")
        else:
            st.metric("Median Sharpe", "—")

    # Only keep Fund Type Distribution
    st.subheader("Fund Type Distribution")
    if "fund_type" in FILTERED.columns:
        fig = px.histogram(FILTERED, x="fund_type")
        fig.update_layout(height=380, bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Column 'fund_type' not found in the dataset.")

    # Preview table + download
    st.subheader("Current filters preview")
    st.caption(f"{len(FILTERED):,} / {len(FUNDS):,} funds match")

    st.dataframe(FILTERED, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered CSV",
        FILTERED.to_csv(index=False).encode("utf-8"),
        file_name="filtered_funds.csv",
        mime="text/csv"
    )

    signature_footer()

elif page == "Explore Funds":
    st.title("Explore Funds")
    st.caption(f"{len(VIEW):,} rows shown")
    

    # ---- Column selection (hide helper cols) ----
    all_cols = [c for c in VIEW.columns if c not in ["search_key"]]
    # sensible defaults, only keep if present
    default_cols_order = [
        "fund_code", "fund_name", "fund_type", "spk_risk_level", "fund_size_in_M",
        ret_col if ret_col else "annual_net_return", "1Y_return",  # return(s)
        "sharpe", "volatility", "max_loss", "annual_fee", "annual_max_total_expense_ratio"
    ]
    default_cols = [c for c in default_cols_order if c in all_cols]
    show_cols = st.multiselect("Columns to show", all_cols, default=default_cols)

    # ---- Ranking controls ----
    rank_cols = [c for c in ["sharpe", ret_col, "annual_net_return", "1Y_return", "volatility", "max_loss", "fund_size_in_M", "annual_fee"] if c in VIEW.columns]
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        sort_by = st.selectbox("Sort by", rank_cols, index=(0 if "sharpe" in rank_cols else 0))
    with c2:
        ascending = st.toggle("Ascending", value=False, help="Turn on for smallest-to-largest")
    with c3:
        top_n = st.slider("Show top N", min_value=10, max_value=min(1000, len(VIEW)), value=min(100, len(VIEW)))

    # ---- Apply sort & head ----
    df_sorted = VIEW.sort_values(by=sort_by, ascending=ascending, kind="mergesort")  # stable sort for ties
    df_shown = df_sorted.head(top_n)[show_cols] if show_cols else df_sorted.head(top_n)

    # ---- Pretty column labels / formats ----
    cfg = {}
    def numcol(title, fmt="%.2f"): return st.column_config.NumberColumn(title, format=fmt)

    label_map = {
        "fund_code": "Fund Code",
        "fund_name": "Fund Name",
        "fund_type": "Fund Type",
        "spk_risk_level": "SPK Risk",
        "fund_size_in_M": "AUM (Million TL)",
        "1Y_return": "1Y Return (%)",
        "annual_net_return": "Annual Return (%)",
        "sharpe": "Sharpe",
        "volatility": "Volatility",
        "max_loss": "Max Loss (%)",
        "annual_fee": "Annual Fee (%)",
        "annual_max_total_expense_ratio": "Max TER (%)",
    }
    # Use dynamic label for the active return column
    if ret_col and ret_col in df_shown.columns:
        label_map[ret_col] = ("Annual" if ret_col == "annual_net_return" else "1Y") + " Return (%)"

    for c in df_shown.columns:
        if c in ["sharpe","volatility","max_loss","annual_fee","annual_max_total_expense_ratio","1Y_return","annual_net_return"]:
            cfg[c] = numcol(label_map.get(c, c), "%.2f")
        elif c == "fund_size_in_M":
            cfg[c] = numcol(label_map.get(c, c), "%.2f")
        else:
            # string/other columns – just rename
            cfg[c] = label_map.get(c, c)

    # ---- Table + download ----
    st.dataframe(df_shown, use_container_width=True, hide_index=True, column_config=cfg)

    csv = df_shown.to_csv(index=False).encode("utf-8")
    st.download_button("Download these rows (CSV)", csv, file_name="explore_funds.csv", mime="text/csv")

    signature_footer()


elif page == "Compare Funds":
    st.title("Compare Funds (A vs B vs C)")
    
    # Pick identifier (code preferred, fallback to name)
    id_col = "fund_code" if "fund_code" in VIEW.columns else ("fund_name" if "fund_name" in VIEW.columns else None)
    if id_col is None:
        st.warning("Need either 'fund_code' or 'fund_name' to compare.")
        signature_footer()
        st.stop()

    options = VIEW[id_col].dropna().astype(str).unique().tolist()
    chosen = st.multiselect("Select up to 3 funds", options, max_selections=3)
    if not chosen:
        st.info("Pick funds from the dropdown to compare.")
        signature_footer()
        st.stop()

    dfc = VIEW[VIEW[id_col].astype(str).isin(chosen)].copy()

    # ----- Metrics to compare -----
    # Active return column & pretty label already computed earlier: ret_col, ret_label
    metrics = []
    metric_labels = {}

    def add_metric(col, label):
        if col in dfc.columns:
            metrics.append(col)
            metric_labels[col] = label

    # Order matters
    add_metric(ret_col if ret_col else "annual_net_return", ret_label if ret_col else "Annual Return (%)")
    add_metric("1Y_return", "1Y Return (%)")
    add_metric("sharpe", "Sharpe")
    add_metric("volatility", "Volatility")
    add_metric("max_loss", "Max Loss (%)")
    add_metric("annual_fee", "Annual Fee (%)")
    add_metric("annual_max_total_expense_ratio", "Max TER (%)")
    add_metric("fund_size_in_M", "AUM (Million TL)")

    if len(metrics) == 0:
        st.warning("No comparable numeric metrics found.")
        signature_footer()
        st.stop()

    # ----- Headline table (compact) -----
    st.subheader("Snapshot")
    cols_to_show = [id_col, "fund_name"] if id_col == "fund_code" and "fund_name" in dfc.columns else [id_col]
    cols_to_show += [m for m in metrics if m not in cols_to_show]
    st.dataframe(dfc[cols_to_show], use_container_width=True, hide_index=True)

    # ----- Grouped bar chart -----
    st.subheader("Metric comparison")
    fig = go.Figure()
    for _, r in dfc.iterrows():
        y_vals = [r[m] if pd.notna(r[m]) else None for m in metrics]
        fig.add_trace(go.Bar(
            name=str(r[id_col]),
            x=[metric_labels[m] for m in metrics],
            y=y_vals
        ))
    fig.update_layout(barmode="group", height=520, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    # ----- Radar chart (normalized 0..1) -----
    # Use only numeric columns that exist across the full VIEW for stable min-max
    st.subheader("Radar (normalized)")
    # compute min/max on the overall VIEW for consistency
    norm_df = dfc[[id_col] + metrics].copy()
    mins = {m: pd.to_numeric(VIEW[m], errors="coerce").min() if m in VIEW.columns else np.nan for m in metrics}
    maxs = {m: pd.to_numeric(VIEW[m], errors="coerce").max() if m in VIEW.columns else np.nan for m in metrics}

    # avoid zero-range issues
    for m in metrics:
        lo, hi = mins[m], maxs[m]
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            # fall back to the dfc spread or constant 0.5 if still invalid
            series = pd.to_numeric(dfc[m], errors="coerce")
            lo2, hi2 = series.min(), series.max()
            if pd.isna(lo2) or pd.isna(hi2) or hi2 == lo2:
                mins[m], maxs[m] = 0.0, 1.0
            else:
                mins[m], maxs[m] = lo2, hi2

    radar = go.Figure()
    theta = [metric_labels[m] for m in metrics]
    for _, r in dfc.iterrows():
        values = []
        for m in metrics:
            v = pd.to_numeric(r[m], errors="coerce")
            lo, hi = mins[m], maxs[m]
            if pd.isna(v) or pd.isna(lo) or pd.isna(hi) or hi == lo:
                values.append(0.5)
            else:
                values.append(float((v - lo) / (hi - lo)))
        radar.add_trace(go.Scatterpolar(
            r=values, theta=theta, fill="toself", name=str(r[id_col])
        ))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=560)
    st.plotly_chart(radar, use_container_width=True)

    signature_footer()

elif page == "Risk/Return Analytics":
    st.title("Risk / Return Analytics")

    # --- Guards ---
    y_metric = ret_col if ret_col and ret_col in VIEW.columns else None
    default_x = "volatility" if "volatility" in VIEW.columns else None

    numeric_candidates = [c for c in VIEW.columns if pd.api.types.is_numeric_dtype(VIEW[c])]
    if y_metric is None:
        st.warning("No return column available. Need 'annual_net_return' or '1Y_return' (or a real version).")
        signature_footer(); st.stop()
    if default_x is None:
        default_x = next((c for c in numeric_candidates if c != y_metric), None)
    if default_x is None:
        st.warning("No numeric columns to plot on X axis.")
        signature_footer(); st.stop()

    # --- Controls ---
    st.subheader("Controls")
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1, 1])
    with c1:
        x_metric = st.selectbox(
            "X axis",
            [default_x] + [c for c in numeric_candidates if c not in [default_x, y_metric]],
            index=0
        )
    with c2:
        color_choices = [c for c in ["fund_type", "spk_risk_level", "company", "currency"] if c in VIEW.columns]
        color_by = st.selectbox("Color by", color_choices, index=0 if color_choices else None)
    with c3:
        size_choices = [c for c in ["fund_size_in_M"] + numeric_candidates if c in VIEW.columns]
        size_by = st.selectbox(
            "Bubble size",
            size_choices,
            index=(size_choices.index("fund_size_in_M") if "fund_size_in_M" in size_choices else 0)
        )
    with c4:
        log_x = st.toggle("Log scale (X)", value=False)

    # --- Build a clean plotting frame & safe size column ---
    df_plot = VIEW.copy()

    # Coerce axes to numeric and drop rows with NaNs on axes
    df_plot[x_metric] = pd.to_numeric(df_plot[x_metric], errors="coerce")
    df_plot[y_metric] = pd.to_numeric(df_plot[y_metric], errors="coerce")
    df_plot = df_plot.dropna(subset=[x_metric, y_metric])

    # Prepare size (only if valid; otherwise skip)
    size_arg = None
    if size_by in df_plot.columns:
        s = pd.to_numeric(df_plot[size_by], errors="coerce")
        if s.notna().sum() >= 3 and (s > 0).any():
            s = s.clip(lower=0)
            s = s.fillna(s.median() if s.notna().any() else 1)
            df_plot["_size"] = s
            size_arg = "_size"

    # --- Hover fields (keep only those present) ---
    hover_cols = [c for c in ["fund_code","fund_name","company","currency","sharpe","max_loss"]
                  if c in df_plot.columns]

    # --- Scatter plot ---
    fig = px.scatter(
        df_plot,
        x=x_metric,
        y=y_metric,
        color=color_by if color_by in df_plot.columns else None,
        size=size_arg,  # None if invalid → no crash
        hover_data=hover_cols,
        labels={y_metric: (ret_label if ret_col and y_metric == ret_col else y_metric)},
        height=640
    )
    if log_x:
        fig.update_xaxes(type="log")

    # Median guide lines
    try:
        x_med = pd.to_numeric(df_plot[x_metric], errors="coerce").median()
        y_med = pd.to_numeric(df_plot[y_metric], errors="coerce").median()
        if pd.notna(x_med): fig.add_vline(x=x_med, line_dash="dot", opacity=0.5)
        if pd.notna(y_med): fig.add_hline(y=y_med, line_dash="dot", opacity=0.5)
    except Exception:
        pass

    st.plotly_chart(fig, use_container_width=True)

    # --- Table of the plotted view (sorted by return desc by default) ---
    st.subheader("Data used in chart")

    # Build an ordered, de-duplicated column list
    cols_raw = ["fund_code", "fund_name", x_metric, y_metric, "sharpe", "volatility", "max_loss", "fund_size_in_M"]
    cols_table = []
    for c in cols_raw:
        if c in df_plot.columns and c not in cols_table:
            cols_table.append(c)

    shown = df_plot[cols_table].copy()
    
    

    # pretty label for y column
    col_cfg = {}
    if y_metric in shown.columns:
        col_cfg[y_metric] = st.column_config.NumberColumn(ret_label, format="%.2f")
        

    st.dataframe(
        shown.sort_values(by=y_metric, ascending=False) if y_metric in shown.columns else shown,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg
    )

    csv = shown.to_csv(index=False).encode("utf-8")
    st.download_button("Download plotted data (CSV)", csv, file_name="risk_return_view.csv", mime="text/csv")

    signature_footer()

elif page == "Live Markets (Gold & USD)":
    st.title("Live Markets (Gold & USD)")

    # -------- Gold (latest + 1M/6M/1Y) --------
    st.subheader("Gold (XAU/USD) — via GoldAPI")

    colg1, colg2 = st.columns([1.2, 1])
    with colg1:
        try:
            # latest
            latest_date = dt.date.today()
            latest_price = goldapi_get_price(None)  # latest endpoint

            # history anchors
            d1m, p1m = goldapi_get_price_near(_n_days_ago(30))
            d6m, p6m = goldapi_get_price_near(_n_days_ago(180))
            d1y, p1y = goldapi_get_price_near(_n_days_ago(365))

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Latest", f"{latest_price:.2f}" if latest_price else "—")
            with k2:
                st.metric("1M", f"{_pct(latest_price, p1m):.2f}%" if latest_price and p1m else "—")
            with k3:
                st.metric("6M", f"{_pct(latest_price, p6m):.2f}%" if latest_price and p6m else "—")
            with k4:
                st.metric("1Y", f"{_pct(latest_price, p1y):.2f}%" if latest_price and p1y else "—")

            st.caption(f"Latest fetched from GoldAPI. 1M/6M/1Y compare to: "
                       f"{_ymd(d1m) if d1m else 'n/a'}, {_ymd(d6m) if d6m else 'n/a'}, {_ymd(d1y) if d1y else 'n/a'}")

        except Exception as e:
            st.error("GoldAPI request failed. Check your GOLDAPI_KEY in secrets or network access.")
            st.exception(e)

    with colg2:
        st.info("Tip: On Streamlit Cloud, put `GOLDAPI_KEY` in **secrets.toml**. We fetch latest and compare to ~1M/6M/1Y ago (adjusted for weekends/holidays).")

    st.divider()

    # -------- USD/TRY (latest + 1M/6M/1Y) --------
    st.subheader("USD/TRY — Exchange Rates")

    colf1, colf2 = st.columns([1.2, 1])
    with colf1:
        try:
            fx_latest = fx_latest_usdtry()
            d1m, fx1m = fx_usdtry_near(_n_days_ago(30))
            d6m, fx6m = fx_usdtry_near(_n_days_ago(180))
            d1y, fx1y = fx_usdtry_near(_n_days_ago(365))

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Latest", f"{fx_latest:.4f}" if fx_latest else "—")
            with k2:
                st.metric("1M", f"{_pct(fx_latest, fx1m):.2f}%" if fx_latest and fx1m else "—")
            with k3:
                st.metric("6M", f"{_pct(fx_latest, fx6m):.2f}%" if fx_latest and fx6m else "—")
            with k4:
                st.metric("1Y", f"{_pct(fx_latest, fx1y):.2f}%" if fx_latest and fx1y else "—")

            if d1m or d6m or d1y:
                st.caption(f"Anchors: {_ymd(d1m) if d1m else 'n/a'}, {_ymd(d6m) if d6m else 'n/a'}, {_ymd(d1y) if d1y else 'n/a'}")
        except Exception as e:
            st.error("FX API request failed. If you have an exchangeratesapi.io key, set EXCHANGE_API_BASE & EXCHANGE_API_KEY in secrets. Otherwise we fallback to exchangerate.host.")
            st.exception(e)

    with colf2:
        st.info("If you have a paid key, set:\nEXCHANGE_API_BASE = 'https://api.exchangeratesapi.io/v1'\nEXCHANGE_API_KEY = '...' \nOtherwise we use exchangerate.host automatically.")


elif page == "Inflation (TUFE)":
    st.title("Turkey CPI (TUFE)")

    if TUFE is None or "date" not in TUFE.columns:
        st.warning("tufe_aug25.csv not loaded or missing 'date' column.")
        signature_footer(); st.stop()

    # --- prep ---
    t = TUFE.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"]).sort_values("date")
    for c in ["annual_change", "monthly_change"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    latest_dt = t["date"].max()
    latest_row = t.loc[t["date"] == latest_dt].iloc[0]

    yoy = float(latest_row["annual_change"]) if "annual_change" in t.columns and pd.notna(latest_row.get("annual_change")) else None
    mom = float(latest_row["monthly_change"]) if "monthly_change" in t.columns and pd.notna(latest_row.get("monthly_change")) else None

    # --- KPIs ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Latest (YoY)", f"{yoy:.2f}%" if yoy is not None else "—")
    with k2:
        st.metric("Latest (MoM)", f"{mom:.2f}%" if mom is not None else "—")
    with k3:
        # 3M avg MoM (annualized)
        if "monthly_change" in t.columns and t["monthly_change"].notna().any():
            last3 = t.tail(3)["monthly_change"].dropna()
            ann3 = (np.prod(1 + last3/100) - 1) * 100
            st.metric("3M Cum (MoM→ann.)", f"{ann3:.2f}%")
        else:
            st.metric("3M Cum (MoM→ann.)", "—")
    with k4:
        # YTD cumulative from Jan of current year
        if "monthly_change" in t.columns:
            year = latest_dt.year
            ytd = t[(t["date"].dt.year == year) & (t["date"] <= latest_dt)]["monthly_change"].dropna()
            ytd_val = (np.prod(1 + ytd/100) - 1) * 100 if len(ytd) else np.nan
            st.metric("YTD (cum MoM)", f"{ytd_val:.2f}%" if pd.notna(ytd_val) else "—")
        else:
            st.metric("YTD (cum MoM)", "—")

    st.caption(f"Last update: {latest_dt.date()}")

    # --- charts ---
    tab1, tab2, tab3 = st.tabs(["YoY %", "MoM %", "Index (base=100)"])

    with tab1:
        if "annual_change" in t.columns and t["annual_change"].notna().any():
            fig = px.line(t, x="date", y="annual_change", markers=True, title=None,
                          labels={"annual_change": "YoY (%)", "date": ""})
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'annual_change' column found.")

    with tab2:
        if "monthly_change" in t.columns and t["monthly_change"].notna().any():
            fig = px.bar(t, x="date", y="monthly_change",
                         labels={"monthly_change": "MoM (%)", "date": ""})
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'monthly_change' column found.")

    with tab3:
        if "monthly_change" in t.columns and t["monthly_change"].notna().any():
            tt = t[["date", "monthly_change"]].dropna().copy()
            tt["index"] = (1 + tt["monthly_change"]/100.0).cumprod() * 100.0

            # rebase option
            c1, c2 = st.columns([1,1])
            with c1:
                base_mode = st.selectbox("Rebase", ["First available", f"Start of {latest_dt.year}"], index=0)
            with c2:
                smooth = st.toggle("Show 12M SMA (on index)", value=False)

            if base_mode != "First available":
                jan = pd.Timestamp(year=latest_dt.year, month=1, day=1)
                base_row = tt[tt["date"] >= jan].head(1)
                if not base_row.empty:
                    base_val = float(base_row["index"].iloc[0])
                    tt["index"] = tt["index"] / base_val * 100.0

            fig = px.line(tt, x="date", y="index", labels={"index": "Index (base=100)", "date": ""})
            if smooth:
                s = tt["index"].rolling(12, min_periods=4).mean()
                fig.add_scatter(x=tt["date"], y=s, mode="lines", name="SMA 12")
            fig.update_layout(height=440)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(tt.tail(24), use_container_width=True, hide_index=True)
            st.download_button("Download index (CSV)", tt.to_csv(index=False).encode("utf-8"),
                               file_name="tufe_index.csv", mime="text/csv")
        else:
            st.info("Need 'monthly_change' to build an index.")

    signature_footer()
    
elif page == "Author Picks — Top by Risk":
    st.title("Author Picks — Top Funds by Risk")
    st.caption("Fixed methodology (independent of sidebar filters). Uses full dataset to keep results stable.")

    # ------- Guards -------
    required_cols = [
        "spk_risk_level","sharpe","max_loss","annual_fee","annual_net_return",
        "1Y_return","real_1y_return","fund_size_in_M","fund_code","fund_type"
    ]
    missing = [c for c in required_cols if c not in FUNDS.columns]
    if missing:
        st.warning(f"Missing columns for scoring: {', '.join(missing)}")
        signature_footer(); st.stop()

    df_base = FUNDS.copy()
    # coerce numerics
    for c in ["spk_risk_level","sharpe","max_loss","annual_fee","annual_net_return",
              "1Y_return","real_1y_return","fund_size_in_M"]:
        df_base[c] = pd.to_numeric(df_base[c], errors="coerce")

    # ------- Controls (optional tweaks) -------
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        top_n = st.number_input("Top N", min_value=5, max_value=50, value=15, step=5)
    with c2:
        min_sharpe = st.number_input("Sharpe >", value=1.0, step=0.1, format="%.1f")
    with c3:
        min_max_loss = st.number_input("Max Loss >", value=-20.0, step=1.0, format="%.1f")
    with c4:
        max_fee = st.number_input("Annual Fee ≤", value=2.0, step=0.1, format="%.1f")
    with c5:
        min_aum = st.number_input("AUM ≥ (M TL)", value=50.0, step=10.0, format="%.0f")

    st.caption("Filters: Sharpe > threshold • Max Loss > threshold • Fee ≤ threshold • Real 1Y Return > 0 • AUM ≥ threshold")

    # ------- Scoring helper (no scipy needed) -------
    def percentile_rank(s: pd.Series, invert: bool = False) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if invert:
            s = -s
        return (s.rank(pct=True, method="average") * 100.0)

    def get_top_funds_by_risk(df: pd.DataFrame, risk_levels, top_n=10) -> pd.DataFrame:
        # Step 1 – universe filter
        filtered = df[
            df["spk_risk_level"].isin(risk_levels) &
            (df["sharpe"] > min_sharpe) &
            (df["max_loss"] > min_max_loss) &
            (df["annual_fee"] <= max_fee) &
            (df["real_1y_return"] > 0) &
            (df["fund_size_in_M"] >= min_aum)
        ].dropna(subset=[
            "sharpe","max_loss","annual_fee","annual_net_return",
            "1Y_return","real_1y_return","fund_size_in_M"
        ]).copy()

        if filtered.empty:
            return filtered

        # Step 2 – percentile ranks
        filtered["1Y_return_score"]        = percentile_rank(filtered["1Y_return"])
        filtered["sharpe_score"]           = percentile_rank(filtered["sharpe"])
        filtered["max_loss_score"]         = percentile_rank(filtered["max_loss"], invert=True)   # less loss is better
        filtered["annual_fee_score"]       = percentile_rank(filtered["annual_fee"], invert=True) # lower fee is better
        filtered["annual_net_return_score"]= percentile_rank(filtered["annual_net_return"])

        # Step 3 – weighted final score
        filtered["final_score"] = (
            filtered["1Y_return_score"]         * 0.20 +
            filtered["sharpe_score"]            * 0.30 +
            filtered["max_loss_score"]          * 0.20 +
            filtered["annual_fee_score"]        * 0.15 +
            filtered["annual_net_return_score"] * 0.15
        ).round(2)

        # Step 4 – top N
        cols = [
            "fund_code","fund_type","fund_size_in_M","spk_risk_level",
            "1Y_return","real_1y_return","3Y_return","5Y_return",
            "sharpe","max_loss","annual_fee","final_score"
        ]
        cols = [c for c in cols if c in filtered.columns]
        return filtered.sort_values("final_score", ascending=False).head(int(top_n))[cols].reset_index(drop=True)

    # ------- Output: 3 tabs (Low / Medium / High) -------
    tab_low, tab_mid, tab_high = st.tabs(["Low risk (1–3)", "Medium risk (4–5)", "High risk (6–7)"])

    with tab_low:
        low = get_top_funds_by_risk(df_base, [1,2,3], top_n=top_n)
        if low.empty:
            st.info("No funds matched the criteria for Low risk.")
        else:
            st.subheader("Top picks — Low risk")
            st.dataframe(low, use_container_width=True, hide_index=True,
                         column_config={"final_score": st.column_config.NumberColumn("Final Score", format="%.2f")})
            st.download_button("Download Low risk picks (CSV)", low.to_csv(index=False).encode("utf-8"),
                               file_name="author_picks_low.csv", mime="text/csv")

    with tab_mid:
        mid = get_top_funds_by_risk(df_base, [4,5], top_n=top_n)
        if mid.empty:
            st.info("No funds matched the criteria for Medium risk.")
        else:
            st.subheader("Top picks — Medium risk")
            st.dataframe(mid, use_container_width=True, hide_index=True,
                         column_config={"final_score": st.column_config.NumberColumn("Final Score", format="%.2f")})
            st.download_button("Download Medium risk picks (CSV)", mid.to_csv(index=False).encode("utf-8"),
                               file_name="author_picks_medium.csv", mime="text/csv")

    with tab_high:
        high = get_top_funds_by_risk(df_base, [6,7], top_n=top_n)
        if high.empty:
            st.info("No funds matched the criteria for High risk.")
        else:
            st.subheader("Top picks — High risk")
            st.dataframe(high, use_container_width=True, hide_index=True,
                         column_config={"final_score": st.column_config.NumberColumn("Final Score", format="%.2f")})
            st.download_button("Download High risk picks (CSV)", high.to_csv(index=False).encode("utf-8"),
                               file_name="author_picks_high.csv", mime="text/csv")

    # small note about independence from sidebar filters
    st.caption("Note: Author picks are computed on the full dataset (not affected by sidebar filters).")
    signature_footer()


elif page == "Methodology & Notes":
    st.title("Methodology & Notes")
    
    st.markdown("""
    ### Data Sources
    - **TEFAS (Fund Analysis)** — August 2025 data downloaded from TEFAS.gov.tr  
    - **Fonbul.com** — Risk analysis and fund performance metrics  
    - **TCMB & Inflation Verileri** — TUFE inflation rates  
    - **Gold Prices** — Historical data scraped from [GoldAPI.io](https://www.goldapi.io)  
    - **USD/TRY** — [ExchangeRatesAPI.io](https://exchangeratesapi.io) (fallback: [ExchangeRate.host](https://exchangerate.host))

    ### Data Processing
    - All CSVs were merged and cleaned manually  
    - Removed duplicates and redundant columns  
    - Converted date fields to `datetime` format  
    - Standardized column names for consistency  
    - Inflation-adjusted returns calculated where possible (`real_1y_return`)

    ### Limitations
    - Dataset is **static** for now — live updates apply only to gold & USD/TRY prices  
    - Inflation adjustment based on **CPI (TUFE)** year-over-year changes  
    - Some risk metrics depend on provider calculations (Sharpe ratio, max loss, etc.)

    ### Usage
    - Use filters in the sidebar to search by **fund type**, **code**, **company**, etc.  
    - Switch between **Nominal** and **Inflation-adjusted** returns  
    - Compare funds side-by-side using charts  
    - View macro indicators on the **Live Markets** and **TUFE** pages

    ---
    © Burak Ozteke, 2025. All rights reserved.
    """)
>>>>>>> 20d5fa1 (Initial commit)
