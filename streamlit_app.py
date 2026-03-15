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

# --- SEITEN-KONFIGURATION ---
st.set_page_config(page_title="WarnwetterBB | Live", layout="wide")

# --- FARBSKALEN DEFINIEREN ---
WW_LEGEND_DATA = {
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

# DEINE TEMPERATUR-SKALA
TEMP_LEVELS = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
TEMP_COLORS = [
    '#FF00FF', # -20 (Magenta)
    '#800080', # -15 (Lila)
    '#00008B', # -10 (Dunkelblau)
    '#0000FF', # -5 (Blau)
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

WIND_LEVELS = [0, 10, 20, 30, 40, 50, 75, 100, 125, 150]
WIND_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Steuerung")
    sel_region = st.selectbox("Region", ["Deutschland", "Brandenburg/Berlin", "Europa"])
    sel_model = st.selectbox("Modell", ["ICON-D2 (DWD)"])
    sel_param = st.selectbox("Parameter", ["Signifikantes Wetter", "Windböen (km/h)", "Temperatur 2m (°C)"])
    sel_hour = st.slider("Vorhersage (+h)", 1, 48, 1)
    st.markdown("---")
    st.caption("Datenquelle: DWD | Downsampling aktiv für Speed")

# --- DATEN-LOADER (MIT SPEED-OPTIMIERUNG) ---
@st.cache_data(ttl=1800)
def fetch_dwd(param_key, hr):
    for offset in [3, 4, 6, 9]:
        now = datetime.now(timezone.utc) - timedelta(hours=offset)
        run_h = (now.hour // 3) * 3
        dt_obj = now.replace(hour=run_h, minute=0, second=0, microsecond=0)
        dt_str = dt_obj.strftime("%Y%m%d%H")
        url = f"https://opendata.dwd.de/weather/nwp/icon-d2/grib/{run_h:02d}/{param_key}/icon-d2_germany_regular-lat-lon_single-level_{dt_str}_{hr:03d}_2d_{param_key}.grib2.bz2"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                with bz2.open(r.raw if hasattr(r, 'raw') else requests.get(url, stream=True).raw) as f_in:
                    with open(f"{param_key}.grib", "wb") as f_out: f_out.write(f_in.read())
                ds = xr.open_dataset(f"{param_key}.grib", engine='cfgrib')
                # SPEED-TRICK: Daten-Auflösung halbieren für schnelleres Plotten
                return ds.coarsen(latitude=2, longitude=2, boundary='trim').mean(), dt_obj
        except: continue
    return None, None

# --- DATEN LADEN ---
with st.spinner('Lade Wetterdaten...'):
    p_map = {"Signifikantes Wetter": "ww", "Windböen (km/h)": "vmax_10m", "Temperatur 2m (°C)": "t_2m"}
    ds_cl, run_dt = fetch_dwd("clct", sel_hour) # Bewölkung
    ds_main, _ = fetch_dwd(p_map[sel_param], sel_hour)

if ds_cl and ds_main:
    valid_time = run_dt + timedelta(hours=sel_hour)
    fig, ax = plt.subplots(figsize=(6, 8.5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    extents = {
        "Deutschland": [5.8, 15.2, 47.2, 55.1],
        "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6],
        "Europa": [-10, 30, 35, 65]
    }
    ax.set_extent(extents[sel_region])
    
    # 1. Wolken (Deutlicher Sichtbar)
    ax.pcolormesh(ds_cl.longitude, ds_cl.latitude, ds_cl.clct, 
                  cmap='Greys', alpha=0.6, shading='auto', zorder=2)
    
    # 2. Haupt-Datenlayer
    vals = ds_main[list(ds_main.data_vars)[0]].values
    
    if sel_param == "Temperatur 2m (°C)":
        temp_c = vals - 273.15 # Kelvin zu Celsius
        im = ax.pcolormesh(ds_main.longitude, ds_main.latitude, temp_c, 
                           cmap=mcolors.ListedColormap(TEMP_COLORS), 
                           norm=mcolors.BoundaryNorm(TEMP_LEVELS, ncolors=len(TEMP_COLORS)), 
                           alpha=0.6, shading='auto', zorder=5)
        plt.colorbar(im, label="°C", shrink=0.5, pad=0.02)
        
    elif sel_param == "Windböen (km/h)":
        im = ax.pcolormesh(ds_main.longitude, ds_main.latitude, vals * 3.6, 
                           cmap=mcolors.ListedColormap(WIND_COLORS), 
                           norm=mcolors.BoundaryNorm(WIND_LEVELS, ncolors=len(WIND_COLORS)), 
                           alpha=0.6, shading='auto', zorder=5)
        plt.colorbar(im, label="km/h", shrink=0.5, pad=0.02)
        
    else: # Signifikantes Wetter
        c_grid = np.zeros_like(vals)
        color_list = ['#FFFFFF00']
        for i, (label, (color, codes)) in enumerate(WW_LEGEND_DATA.items(), 1):
            for code in codes: c_grid[vals == code] = i
            color_list.append(color)
        ax.pcolormesh(ds_main.longitude, ds_main.latitude, c_grid, 
                      cmap=mcolors.ListedColormap(color_list), alpha=0.8, shading='auto', zorder=5)
        
        legend_patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND_DATA.items()]
        leg = ax.legend(handles=legend_patches, loc='lower left', title="Wetter", fontsize='x-small', framealpha=0.9)
        leg.set_zorder(20)

    # Grenzen & Karte
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=10)

    # Info-Box oben
    info = f"Region: {sel_region}\nZeit: {valid_time.strftime('%d.%m. %H:00')} UTC\nModell: {sel_model} ({run_dt.strftime('%H')}Z)"
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=25)
    
    st.pyplot(fig)
else:
    st.info("🔄 Daten werden vom DWD-Server abgerufen...")
