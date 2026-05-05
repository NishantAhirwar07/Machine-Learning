import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CO₂ Emission Predictor",
    page_icon="🚗",
    layout="centered",
)

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0f1117 !important;
    font-family: 'Inter', sans-serif;
    color: #e0e0e0;
}
[data-testid="stHeader"] { background: transparent !important; }

.input-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #9ca3af;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Hide auto-generated Streamlit labels */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stSelectSlider"] label { display: none !important; }

/* Selectbox */
div[data-baseweb="select"] > div {
    background: #252836 !important;
    border: 1px solid #363a4f !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
}
div[data-baseweb="select"] svg { color: #6b7280 !important; }

/* Slider accent */
[data-testid="stSlider"] > div > div > div > div {
    background: #3b82f6 !important;
}
[data-testid="stSelectSlider"] > div > div > div > div {
    background: #3b82f6 !important;
}

/* Button */
.stButton > button {
    background: #3b82f6 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px !important;
    letter-spacing: 0.5px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #2563eb !important;
}

#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2330 !important;
}
[data-testid="stSidebar"] .input-label { color: #9ca3af; }
[data-testid="stSidebarContent"] { padding: 24px 16px !important; }
</style>
""", unsafe_allow_html=True)


# ── Train model ───────────────────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1200
    cylinders = np.random.choice([3, 4, 5, 6, 8, 10, 12, 16], size=n,
                                  p=[0.05, 0.38, 0.04, 0.28, 0.17, 0.04, 0.03, 0.01])
    engine    = cylinders * 0.35 + np.random.normal(0, 0.4, n)
    engine    = np.clip(engine, 1.0, 8.5)
    fuel_comb = engine * 2.1 + cylinders * 0.5 + np.random.normal(0, 0.9, n)
    fuel_comb = np.clip(fuel_comb, 4.0, 22.0)
    co2       = 7.0 * cylinders + 11.6 * engine + 9.3 * fuel_comb + 65 + np.random.normal(0, 12, n)

    X = np.column_stack([cylinders, engine, fuel_comb])
    X_train, _, y_train, _ = train_test_split(X, co2, test_size=0.2, random_state=2)
    lr = LinearRegression().fit(X_train, y_train)
    return lr

model = train_model()


# ── All car makes from FuelConsumptionCo2.csv ─────────────────────
CAR_MAKES_DEFAULT = [
    "ACURA", "ALFA ROMEO", "ASTON MARTIN", "AUDI", "BENTLEY",
    "BMW", "BUICK", "CADILLAC", "CHEVROLET", "CHRYSLER",
    "DODGE", "FIAT", "FORD", "GMC", "HONDA",
    "HYUNDAI", "INFINITI", "JAGUAR", "JEEP", "KIA",
    "LAMBORGHINI", "LAND ROVER", "LEXUS", "LINCOLN", "MASERATI",
    "MAZDA", "MERCEDES-BENZ", "MINI", "MITSUBISHI", "NISSAN",
    "PORSCHE", "RAM", "ROLLS-ROYCE", "SCION", "SMART",
    "SUBARU", "TOYOTA", "VOLKSWAGEN", "VOLVO",
]

try:
    df_csv = pd.read_csv("FuelConsumptionCo2.csv")
    car_makes = sorted(df_csv["MAKE"].dropna().unique().tolist())
except Exception:
    car_makes = CAR_MAKES_DEFAULT

# Default slider hints per brand
BRAND_HINTS = {
    "ACURA":         (4,  2.0,  9.5), "ALFA ROMEO":    (4,  2.0,  9.8),
    "ASTON MARTIN":  (8,  4.7, 17.0), "AUDI":          (4,  2.0, 10.5),
    "BENTLEY":       (8,  4.0, 18.5), "BMW":           (4,  2.0, 10.0),
    "BUICK":         (4,  2.5, 10.8), "CADILLAC":      (6,  3.6, 13.5),
    "CHEVROLET":     (4,  2.5, 11.0), "CHRYSLER":      (6,  3.6, 13.0),
    "DODGE":         (6,  3.6, 13.5), "FIAT":          (4,  1.4,  7.5),
    "FORD":          (4,  2.0, 10.5), "GMC":           (6,  4.3, 15.0),
    "HONDA":         (4,  1.8,  8.5), "HYUNDAI":       (4,  2.0,  9.0),
    "INFINITI":      (6,  3.5, 12.5), "JAGUAR":        (6,  3.0, 13.0),
    "JEEP":          (4,  2.4, 11.5), "KIA":           (4,  2.0,  9.2),
    "LAMBORGHINI":   (10, 5.2, 20.0), "LAND ROVER":    (6,  3.0, 14.0),
    "LEXUS":         (6,  3.5, 12.0), "LINCOLN":       (4,  2.0, 12.0),
    "MASERATI":      (6,  3.0, 14.5), "MAZDA":         (4,  2.0,  9.0),
    "MERCEDES-BENZ": (4,  2.0, 10.5), "MINI":          (4,  1.5,  7.8),
    "MITSUBISHI":    (4,  2.0,  9.5), "NISSAN":        (4,  1.8,  9.0),
    "PORSCHE":       (6,  3.0, 12.5), "RAM":           (6,  3.6, 14.5),
    "ROLLS-ROYCE":   (12, 6.6, 20.5), "SCION":         (4,  1.8,  8.5),
    "SMART":         (3,  1.0,  6.5), "SUBARU":        (4,  2.5, 10.0),
    "TOYOTA":        (4,  2.0,  9.0), "VOLKSWAGEN":    (4,  2.0,  9.5),
    "VOLVO":         (4,  2.0, 10.0),
}


# ══════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="text-align:center;padding:30px 0 18px">
    <div style="font-size:2.8rem"></div>
    <h1 style="font-size:1.8rem;font-weight:600;color:#f3f4f6;margin:10px 0 6px">
        CO₂ Emission Predictor
    </h1>
</div>
""", unsafe_allow_html=True)

# ── Sidebar inputs ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="input-label">🏷️ &nbsp; Car Brand / Make</div>', unsafe_allow_html=True)
    selected_make = st.selectbox(
        "Car Make",
        options=["— Choose a brand —"] + car_makes,
        key="make_select",
    )

    hint = BRAND_HINTS.get(selected_make, (4, 2.0, 9.5))
    default_cyl, default_eng, default_fuel = hint

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="input-label">🔩 &nbsp; Number of Cylinders</div>', unsafe_allow_html=True)
    cylinders = st.select_slider(
        "Cylinders",
        options=[2, 3, 4, 5, 6, 8, 10, 12, 16],
        value=default_cyl,
        key="cyl_slider",
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="input-label">⚙️ &nbsp; Engine Size <span style="color:#4b5563;font-weight:400">(Litres)</span></div>', unsafe_allow_html=True)
    engine_size = st.slider(
        "Engine Size", min_value=1.0, max_value=8.5,
        value=float(default_eng), step=0.1, key="eng_slider", format="%.1f L",
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="input-label">⛽ &nbsp; Combined Fuel Consumption <span style="color:#4b5563;font-weight:400">(L / 100 km)</span></div>', unsafe_allow_html=True)
    fuel_comb = st.slider(
        "Fuel Consumption", min_value=4.0, max_value=22.0,
        value=float(default_fuel), step=0.1, key="fuel_slider", format="%.1f L",
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Predict button ─────────────────────────────────────────────
    st.button("🔍  Predict CO₂ Emission", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────
prediction = float(model.predict([[cylinders, engine_size, fuel_comb]])[0])
prediction = max(80.0, prediction)

if   prediction < 150: level, color, icon, tip = "LOW",      "#22c55e", "🌿", "Below average — relatively eco-friendly"
elif prediction < 200: level, color, icon, tip = "MODERATE", "#eab308", "⚠️", "Average range — moderate environmental impact"
elif prediction < 270: level, color, icon, tip = "HIGH",     "#f97316", "🔥", "Above average — significant CO₂ output"
else:                  level, color, icon, tip = "EXTREME",  "#ef4444", "☠️", "Extreme emitter — heavy environmental burden"

# ── Result card ───────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,{color}12,{color}06);
            border:2px solid {color}55;border-radius:14px;
            padding:32px 24px;text-align:center;margin-bottom:20px">
    <div style="font-size:2rem;margin-bottom:10px">{icon}</div>
    <div style="font-size:0.75rem;font-weight:600;letter-spacing:3px;
                text-transform:uppercase;color:#9ca3af;margin-bottom:8px">
        Predicted CO₂ Emission
    </div>
    <div style="font-size:4.2rem;font-weight:800;color:{color};line-height:1;margin-bottom:4px">
        {prediction:.0f}
    </div>
    <div style="font-size:1rem;color:#6b7280;margin-bottom:18px">g / km</div>
    <span style="display:inline-block;padding:5px 22px;border-radius:999px;
                 background:{color}22;border:1px solid {color}66;
                 font-size:0.78rem;font-weight:700;letter-spacing:2px;
                 text-transform:uppercase;color:{color}">
        {level}
    </span>
    <p style="color:#9ca3af;font-size:0.87rem;margin-top:14px;margin-bottom:0">{tip}</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────
annual_kg    = prediction * 15_000 / 1_000
trees_needed = annual_kg / 21
vs_avg       = (prediction / 180 - 1) * 100
vs_color     = "#22c55e" if vs_avg < 0 else "#ef4444"

c1, c2, c3 = st.columns(3)
for col, ico, val, lbl in [
    (c1, "📅", f"{annual_kg:,.0f} kg",    "Annual CO₂ (15k km)"),
    (c2, "🌳", f"{trees_needed:,.0f}",     "Trees to offset / yr"),
    (c3, "📊", f"{vs_avg:+.0f}%",          "vs Avg vehicle"),
]:
    with col:
        st.markdown(f"""
        <div style="background:#1a1d27;border:1px solid #252836;
                    border-radius:12px;padding:16px 12px;text-align:center">
            <div style="font-size:1.3rem;margin-bottom:6px">{ico}</div>
            <div style="font-size:1.25rem;font-weight:700;color:#f3f4f6">{val}</div>
            <div style="font-size:0.7rem;color:#6b7280;margin-top:4px">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── CO₂ level guide ───────────────────────────────────────────────
st.markdown("""
<div style="font-size:0.78rem;font-weight:600;letter-spacing:2px;
            text-transform:uppercase;color:#6b7280;margin-bottom:12px">
    📏 &nbsp; CO₂ Level Reference
</div>
""", unsafe_allow_html=True)

for ico, clr, lbl, rng, ex in [
    ("🌿", "#22c55e", "LOW",      "< 150 g/km",   "Eco / Hybrid vehicles"),
    ("⚠️", "#eab308", "MODERATE", "150–200 g/km", "Compact cars / Sedans"),
    ("🔥", "#f97316", "HIGH",     "200–270 g/km", "SUVs / Large engines"),
    ("☠️", "#ef4444", "EXTREME",  "> 270 g/km",   "Performance / Heavy trucks"),
]:
    border = f"2px solid {clr}66" if lbl == level else "1px solid #252836"
    bg     = f"{clr}0d"          if lbl == level else "#1a1d27"
    st.markdown(f"""
    <div style="background:{bg};border:{border};border-radius:10px;
                padding:11px 16px;margin-bottom:8px;
                display:flex;align-items:center;gap:14px">
        <span style="font-size:1.1rem;min-width:24px">{ico}</span>
        <span style="font-weight:700;color:{clr};font-size:0.85rem;min-width:90px;letter-spacing:1px">{lbl}</span>
        <span style="color:#9ca3af;font-size:0.85rem;min-width:115px">{rng}</span>
        <span style="color:#6b7280;font-size:0.82rem">{ex}</span>
    </div>
    """, unsafe_allow_html=True)
