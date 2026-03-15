import streamlit as st
import xarray as xr
import requests, bz2, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
import numpy as np

# --- CONFIG ---
st.set_page_config(page_title="WarnwetterBB Ultra", layout="wide")

# --- DEINE TEMPERATUR-SKALA (Exakt nach Liste) ---
# Wir definieren 14 Bereiche für deine 14 Farben
T_LEVELS = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
T_COLORS = [
    '#FF00FF', # -20 bis -15 (Magenta)
    '#800080', # -15 bis -10 (Lila)
    '#00008B', # -10 bis -5 (Dunkelblau)
    '#0000FF', # -5 bis 0 (Blau)
    '#ADD8E6', # -0 (Hellblau)
    '#006400', # +0 (Dunkelgrün)
    '#008000', # +5 (Grün)
    '#ADFF2F', # +10 (Gelbgrün)
    '#FFFF00', # +15 (Gelb)
    '#FFD700', # +20 (Hellorange)
    '#FFA500', # +25 (Orange)
    '#FF0000', # +30 (Rot)
    '#8B0000', # +35 (Dunkelrot)
    '#800080'  # +40 (Lila)
]

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Steuerung")
    region = st.selectbox("Region", ["Brandenburg/Berlin", "Deutschland"])
    param = st.selectbox("Parameter", ["Temperatur 2m (°C)", "Signifikantes Wetter", "Windböen (km/h)"])
    hour = st.slider("Stunde (+h)", 1, 48, 1)

# --- SPEED-LOADER ---
@st.cache_data(ttl=1200)
def fast_fetch(p_key, hr):
    for off in [3, 4, 6]:
        now = datetime.now(timezone.utc) - timedelta(hours=off)
        run = (now.hour // 3) * 3
        dt_s = now.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
        url = f"https://opendata.dwd.de/weather/nwp/icon-d2/grib/{run:02d}/{p_key}/icon-d2_germany_regular-lat-lon_single-level_{dt_s}_{hr:03d}_2d_{p_key}.grib2.bz2"
        try:
            r = requests.get(url, timeout=5, stream=True)
            if r.status_code == 200:
                with open("temp.bz2", "wb") as f: f.write(r.content)
                with bz2.open("temp.bz2", "rb") as f_in, open("temp.grib", "wb") as f_out:
                    f_out.write(f_in.read())
                ds = xr.open_dataset("temp.grib", engine='cfgrib')
                # Slicing & Squeeze fixen den TypeError
                data_slice = ds.sel(latitude=slice(56, 46), longitude=slice(5, 16)).load()
                return data_slice, dt_s
        except: continue
    return None, None

# --- VERARBEITUNG ---
p_map = {"Signifikantes Wetter": "ww", "Windböen (km/h)": "vmax_10m", "Temperatur 2m (°C)": "t_2m"}
with st.spinner('Daten werden verarbeitet...'):
    ds, run_info = fast_fetch(p_map[param], hour)

if ds:
    # --- WICHTIG: Dimensionen glätten ---
    # Das hier verhindert den "TypeError: Dimensions of C should match"
    var_name = list(ds.data_vars)[0]
    data_raw = ds[var_name].values.squeeze() 
    lons = ds.longitude.values
    lats = ds.latitude.values
    
    # Karte erstellen
    fig, ax = plt.subplots(figsize=(6, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ext = {"Deutschland": [5.8, 15.2, 47.2, 55.1], "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6]}
    ax.set_extent(ext[region])

    if "t_2m" in p_map[param]:
        plot_data = data_raw - 273.15 # Kelvin zu Celsius
        im = ax.pcolormesh(lons, lats, plot_data, cmap=mcolors.ListedColormap(T_COLORS), 
                           norm=mcolors.BoundaryNorm(T_LEVELS, ncolors=len(T_COLORS)), shading='nearest', zorder=5)
        plt.colorbar(im, label="°C", shrink=0.5, pad=0.02)
    
    # ... (Andere Parameter wie Wind folgen hier analog mit .squeeze())
    
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=10)
    
    # Info-Box oben links in der Karte
    valid_time = datetime.strptime(run_info, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=hour)
    info = f"Region: {region}\nZeit: {valid_time.strftime('%d.%m. %H:00')} UTC\nModell: ICON-D2 ({run_info[-2:]}Z Lauf)"
    ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8), zorder=25)
    
    st.pyplot(fig)
    
    # Aufräumen
    if os.path.exists("temp.grib"): os.remove("temp.grib")
    if os.path.exists("temp.bz2"): os.remove("temp.bz2")
else:
    st.error("Der DWD-Server antwortet gerade nicht. Bitte kurz warten und Seite neu laden.")
