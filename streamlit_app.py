import streamlit as st
import xarray as xr
import requests, bz2, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
import numpy as np

st.set_page_config(page_title="WarnwetterBB Ultra", layout="wide")

# --- FARBSKALEN (DEINE VORGABE) ---
T_LEVELS = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
T_COLORS = ['#FF00FF','#800080','#00008B','#0000FF','#ADD8E6','#006400','#008000','#ADFF2F','#FFFF00','#FFD700','#FFA500','#FF0000','#8B0000','#800080']

WW_COLORS = {'Nebel': ('#FFFF00', range(40, 50)), 'Regen L': ('#90EE90', [50, 60]), 'Regen M': ('#00FF00', [61, 81]), 'Regen S': ('#006400', [63, 82]), 'Schnee': ('#0000FF', [71, 73, 75]), 'Gewitter': ('#800080', [95, 96, 97])}
W_LEVELS = [0, 10, 20, 30, 40, 50, 75, 100, 125, 150]
W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Steuerung")
    region = st.selectbox("Region", ["Brandenburg/Berlin", "Deutschland"])
    param = st.selectbox("Parameter", ["Temperatur 2m (°C)", "Signifikantes Wetter", "Windböen (km/h)"])
    hour = st.slider("Stunde (+h)", 1, 48, 1)

# --- SPEED-LOADER ---
@st.cache_data(ttl=3600) # 1 Stunde speichern
def fast_fetch(p_key, hr):
    for off in [3, 4, 6]:
        now = datetime.now(timezone.utc) - timedelta(hours=off)
        run = (now.hour // 3) * 3
        dt_s = now.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
        url = f"https://opendata.dwd.de/weather/nwp/icon-d2/grib/{run:02d}/{p_key}/icon-d2_germany_regular-lat-lon_single-level_{dt_s}_{hr:03d}_2d_{p_key}.grib2.bz2"
        try:
            r = requests.get(url, timeout=3, stream=True)
            if r.status_code == 200:
                with open("data.grib.bz2", "wb") as f:
                    for chunk in r.iter_content(chunk_size=128*1024): f.write(chunk)
                with bz2.open("data.grib.bz2", "rb") as f_in, open("data.grib", "wb") as f_out:
                    f_out.write(f_in.read())
                ds = xr.open_dataset("data.grib", engine='cfgrib')
                # Radikales Slicing für Geschwindigkeit
                return ds.sel(latitude=slice(55.5, 47), longitude=slice(5.5, 15.5)).load(), dt_s
        except: continue
    return None, None

# --- PLOTTING ---
p_map = {"Signifikantes Wetter": "ww", "Windböen (km/h)": "vmax_10m", "Temperatur 2m (°C)": "t_2m"}
ds, run_info = fast_fetch(p_map[param], hour)

if ds:
    fig, ax = plt.subplots(figsize=(5, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ext = {"Deutschland": [5.8, 15.2, 47.2, 55.1], "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6]}
    ax.set_extent(ext[region])
    
    data = ds[list(ds.data_vars)[0]].values
    lons, lats = ds.longitude.values, ds.latitude.values

    if "t_2m" in p_map[param]:
        im = ax.pcolormesh(lons, lats, data - 273.15, cmap=mcolors.ListedColormap(T_COLORS), 
                           norm=mcolors.BoundaryNorm(T_LEVELS + [50], ncolors=len(T_COLORS)), shading='nearest', zorder=5)
        plt.colorbar(im, label="°C", shrink=0.5)
    elif "vmax" in p_map[param]:
        im = ax.pcolormesh(lons, lats, data * 3.6, cmap=mcolors.ListedColormap(W_COLORS), 
                           norm=mcolors.BoundaryNorm(W_LEVELS, ncolors=len(W_COLORS)), shading='nearest', zorder=5)
        plt.colorbar(im, label="km/h", shrink=0.5)
    else:
        grid = np.zeros_like(data); c_list = ['#FFFFFF00']
        for i, (l, (c, codes)) in enumerate(WW_COLORS.items(), 1):
            for code in codes: grid[data == code] = i
            c_list.append(c)
        ax.pcolormesh(lons, lats, grid, cmap=mcolors.ListedColormap(c_list), shading='nearest', zorder=5)

    ax.add_feature(cfeature.BORDERS, linewidth=0.8, zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=10)
    
    # Info-Box
    info = f"Region: {region}\nTermin: +{hour}h\nLauf: {run_info} UTC"
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=8, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8), zorder=20)
    
    st.pyplot(fig)
    os.remove("data.grib") # Speicher sofort freigeben
    os.remove("data.grib.bz2")
else:
    st.error("Server-Timeout beim DWD. Bitte in 10 Sek. noch mal probieren.")
