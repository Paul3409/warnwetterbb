import streamlit as st
import xarray as xr
import requests, bz2, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
import numpy as np

# --- CONFIG ---
st.set_page_config(page_title="WarnwetterBB Turbo", layout="wide")

# --- FARBSKALEN ---
WW_COLORS = {
    "Nebel": ("#FFFF00", range(40, 50)),
    "Regen leicht": ("#90EE90", [50, 51, 60, 80]),
    "Regen mäßig": ("#00FF00", [53, 61, 81]),
    "Regen stark": ("#006400", [55, 63, 65, 82]),
    "Schneeregen": ("#FFA500", [68, 69, 83, 84]),
    "Schnee leicht": ("#ADD8E6", [70, 71, 85]),
    "Schnee mäßig": ("#0000FF", [73, 86]),
    "Schnee stark": ("#00008B", [75, 87]),
    "gefr. Regen": ("#FF0000", [66, 67]),
    "Gewitter": ("#800080", [95, 96, 97, 99])
}

# Deine Temperatur-Skala
T_LEVELS = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
T_COLORS = ['#FF00FF','#800080','#00008B','#0000FF','#ADD8E6','#006400','#008000','#ADFF2F','#FFFF00','#FFD700','#FFA500','#FF0000','#8B0000','#800080']

W_LEVELS = [0, 10, 20, 30, 40, 50, 75, 100, 125, 150]
W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Steuerung")
    region = st.selectbox("Region", ["Brandenburg/Berlin", "Deutschland", "Europa"])
    param = st.selectbox("Parameter", ["Signifikantes Wetter", "Windböen (km/h)", "Temperatur 2m (°C)"])
    hour = st.slider("Vorhersage (+h)", 1, 48, 1)
    st.markdown("---")
    st.caption("Modell: ICON-D2 (DWD)")

# --- TURBO-DATA-FETCH ---
@st.cache_data(ttl=1200) # 20 Min Cache
def get_weather(p_key, hr):
    for off in [3, 4, 6, 9]:
        now = datetime.now(timezone.utc) - timedelta(hours=off)
        run = (now.hour // 3) * 3
        dt = now.replace(hour=run, minute=0, second=0, microsecond=0)
        dt_s = dt.strftime("%Y%m%d%H")
        url = f"https://opendata.dwd.de/weather/nwp/icon-d2/grib/{run:02d}/{p_key}/icon-d2_germany_regular-lat-lon_single-level_{dt_s}_{hr:03d}_2d_{p_key}.grib2.bz2"
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                with bz2.open(r.raw if hasattr(r, 'raw') else requests.get(url, stream=True).raw) as f_in:
                    with open(f"{p_key}.grib", "wb") as f: f.write(f_in.read())
                ds = xr.open_dataset(f"{p_key}.grib", engine='cfgrib')
                # Slicing: Nur Deutschland + Puffer laden, spart massiv Zeit beim Plotten
                return ds.sel(latitude=slice(56, 46), longitude=slice(5, 16)), dt
        except: continue
    return None, None

# --- PROCESSING ---
p_map = {"Signifikantes Wetter": "ww", "Windböen (km/h)": "vmax_10m", "Temperatur 2m (°C)": "t_2m"}
with st.spinner('Berechne Karte...'):
    ds_cl, run_dt = get_weather("clct", hour) # Wolken
    ds_main, _ = get_weather(p_map[param], hour)

if ds_cl and ds_main:
    valid_time = run_dt + timedelta(hours=hour)
    # Karte schmal für Mobile
    fig, ax = plt.subplots(figsize=(6, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    ext = {"Deutschland": [5.8, 15.2, 47.2, 55.1], "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6], "Europa": [-5, 25, 40, 60]}
    ax.set_extent(ext[region])

    # 1. Wolken (Hintergrund - 0.7 alpha für Power)
    ax.pcolormesh(ds_cl.longitude, ds_cl.latitude, ds_cl.clct, cmap='Greys', alpha=0.7, shading='nearest', zorder=2)

    # 2. Daten
    v = ds_main[list(ds_main.data_vars)[0]].values
    if param == "Temperatur 2m (°C)":
        # Formel: $T_{Celsius} = T_{Kelvin} - 273.15$
        data = v - 273.15
        im = ax.pcolormesh(ds_main.longitude, ds_main.latitude, data, cmap=mcolors.ListedColormap(T_COLORS), 
                           norm=mcolors.BoundaryNorm(T_LEVELS + [50], ncolors=len(T_COLORS)), alpha=0.6, shading='nearest', zorder=5)
        plt.colorbar(im, label="°C", shrink=0.5, pad=0.02)
    elif param == "Windböen (km/h)":
        im = ax.pcolormesh(ds_main.longitude, ds_main.latitude, v * 3.6, cmap=mcolors.ListedColormap(W_COLORS), 
                           norm=mcolors.BoundaryNorm(W_LEVELS, ncolors=len(W_COLORS)), alpha=0.6, shading='nearest', zorder=5)
        plt.colorbar(im, label="km/h", shrink=0.5, pad=0.02)
    else:
        grid = np.zeros_like(v); c_list = ['#FFFFFF00']
        for i, (l, (c, codes)) in enumerate(WW_COLORS.items(), 1):
            for code in codes: grid[v == code] = i
            c_list.append(c)
        ax.pcolormesh(ds_main.longitude, ds_main.latitude, grid, cmap=mcolors.ListedColormap(c_list), alpha=0.8, shading='nearest', zorder=5)
        patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_COLORS.items()]
        ax.legend(handles=patches, loc='lower left', fontsize='x-small', framealpha=0.8).set_zorder(20)

    # Deko & Info
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=10)
    
    # Info-Box oben in der Karte
    info = f"Region: {region}\nTermin: {valid_time.strftime('%d.%m. %H:00')} UTC\nModell: ICON-D2 ({run_dt.strftime('%H')}Z Lauf)"
    ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8), zorder=25)
    
    st.pyplot(fig, clear_figure=True) # clear_figure spart Speicher
else:
    st.info("📡 Hole frische Daten vom DWD...")
