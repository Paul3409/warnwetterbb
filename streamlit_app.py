import streamlit as st
import xarray as xr
import requests, bz2, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
import numpy as np

st.set_page_config(page_title="WarnwetterBB | Live", layout="wide")
st.title("🛰️ WarnwetterBB | ICON-D2 Profi-Check")

# --- FARBEN ---
def get_ww_cmap():
    cols = ['#FFFFFF00', '#FFFF00', '#90EE90', '#00FF00', '#006400', '#FFA500', '#ADD8E6', '#0000FF', '#00008B', '#FF0000', '#8B0000', '#FF00FF', '#800080']
    return mcolors.ListedColormap(cols)

W_LEVELS = [0, 10, 20, 30, 40, 50, 75, 100, 125, 150]
W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Menü")
    mode = st.radio("Was willst du sehen?", ["Sign. Wetter", "Windböen (km/h)"])
    hr = st.select_slider("Vorhersagezeit (+h):", options=[1, 3, 6, 12, 24, 48], value=1)

# --- DATEN-DOWNLOAD ---
@st.cache_data(ttl=1800)
def fetch(p_key, h):
    for offset in [3, 4, 6]:
        now = datetime.now(timezone.utc) - timedelta(hours=offset)
        run = (now.hour // 3) * 3
        dt_str = now.replace(hour=run, minute=0, second=0).strftime("%Y%m%d%H")
        url = f"https://opendata.dwd.de/weather/nwp/icon-d2/grib/{run:02d}/{p_key}/icon-d2_germany_regular-lat-lon_single-level_{dt_str}_{h:03d}_2d_{p_key}.grib2.bz2"
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with bz2.open(r.raw) as f_in, open(f"{p_key}.grib", "wb") as f_out:
                f_out.write(f_in.read())
            return xr.open_dataset(f"{p_key}.grib", engine='cfgrib'), dt_str
    return None, None

# --- KARTE ---
ds_cl, run_id = fetch("clct", hr) # Wolken
p_map = {"Sign. Wetter": "ww", "Windböen (km/h)": "vmax_10m"}
ds_main, _ = fetch(p_map[mode], hr)

if ds_cl and ds_main:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([5.8, 15.2, 47.2, 55.1])
    
    # 1. Schicht: Wolken
    ax.pcolormesh(ds_cl.longitude, ds_cl.latitude, ds_cl[list(ds_cl.data_vars)[0]], cmap='Greys', alpha=0.3, shading='auto', zorder=2)
    
    # 2. Schicht: Hauptdaten
    vals = ds_main[list(ds_main.data_vars)[0]].values
    if mode == "Windböen (km/h)":
        im = ax.pcolormesh(ds_main.longitude, ds_main.latitude, vals * 3.6, cmap=mcolors.ListedColormap(W_COLORS), norm=mcolors.BoundaryNorm(W_LEVELS, ncolors=len(W_COLORS)), alpha=0.7, shading='auto', zorder=5)
        plt.colorbar(im, label="km/h", shrink=0.5)
    else:
        c_grid = np.zeros_like(vals)
        c_grid[(vals >= 40) & (vals <= 49)] = 1 # Nebel
        c_grid[(vals == 60) | (vals == 50) | (vals == 80)] = 2 # R. leicht
        c_grid[(vals == 61) | (vals == 81)] = 3 # R. mäßig
        c_grid[(vals >= 62) & (vals <= 65)] = 4 # R. stark
        c_grid[vals >= 95] = 11 # Gewitter
        c_grid = np.ma.masked_where(c_grid == 0, c_grid)
        ax.pcolormesh(ds_main.longitude, ds_main.latitude, c_grid, cmap=get_ww_cmap(), alpha=0.8, shading='auto', zorder=5)

    ax.add_feature(cfeature.BORDERS, linewidth=1, zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=10)
    st.pyplot(fig)
    st.success(f"Lauf: {run_id} UTC | Vorhersage: +{hr}h")
else:
    st.warning("Echtzeit-Daten werden geladen...")
