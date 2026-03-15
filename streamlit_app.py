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
st.set_page_config(page_title="WarnwetterBB Profi", layout="wide")

# --- FARBSKALEN ---
# Deine Temperatur-Skala
T_LEVELS = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
T_COLORS = ['#FF00FF','#800080','#00008B','#0000FF','#ADD8E6','#006400','#008000','#ADFF2F','#FFFF00','#FFD700','#FFA500','#FF0000','#8B0000','#800080']

WW_LEGEND = {
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

W_LEVELS = [0, 10, 20, 30, 40, 50, 75, 100, 125, 150]
W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Steuerung")
    sel_region = st.selectbox("Region", ["Brandenburg/Berlin", "Deutschland", "Europa"])
    sel_model = st.selectbox("Modell", ["ICON-D2 (DWD)"])
    sel_param = st.selectbox("Parameter", ["Temperatur 2m (°C)", "Signifikantes Wetter", "Windböen (km/h)"])
    sel_hour = st.slider("Stunde (+h)", 1, 48, 1)

# --- DATA FETCHING (FULL RESOLUTION) ---
@st.cache_data(ttl=1200)
def fetch_data(p_key, hr):
    # ICON-D2 läuft alle 3h (00, 03, 06...)
    for off in [2, 3, 4, 6]:
        now = datetime.now(timezone.utc) - timedelta(hours=off)
        run = (now.hour // 3) * 3
        dt_obj = now.replace(hour=run, minute=0, second=0, microsecond=0)
        dt_s = dt_obj.strftime("%Y%m%d%H")
        url = f"https://opendata.dwd.de/weather/nwp/icon-d2/grib/{run:02d}/{p_key}/icon-d2_germany_regular-lat-lon_single-level_{dt_s}_{hr:03d}_2d_{p_key}.grib2.bz2"
        
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with bz2.open(requests.get(url, stream=True).raw) as f_in:
                    content = f_in.read()
                    with open(f"{p_key}.grib", "wb") as f_out: f_out.write(content)
                ds = xr.open_dataset(f"{p_key}.grib", engine='cfgrib')
                return ds, dt_obj
        except: continue
    return None, None

# --- PROCESSING & PLOTTING ---
p_map = {"Signifikantes Wetter": "ww", "Windböen (km/h)": "vmax_10m", "Temperatur 2m (°C)": "t_2m"}

with st.spinner('Lade hochauflösende Daten...'):
    ds, run_dt = fetch_data(p_map[sel_param], sel_hour)

if ds:
    # --- DIMENSION FIX (DER ENTSCHEIDENDE TEIL) ---
    # Wir nehmen die erste Variable im Datensatz und quetschen alle leeren Dimensionen weg
    var_name = list(ds.data_vars)[0]
    data_array = ds[var_name].isel(step=0, height=0, missing_dims='ignore').values.squeeze()
    lons = ds.longitude.values
    lats = ds.latitude.values

    valid_time = run_dt + timedelta(hours=sel_hour)
    
    # Karte erstellen (Schmal für Mobile-Ansicht)
    fig, ax = plt.subplots(figsize=(6, 8.5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    extents = {
        "Deutschland": [5.8, 15.2, 47.2, 55.1],
        "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6],
        "Europa": [-5, 25, 40, 62]
    }
    ax.set_extent(extents[sel_region])

    # Hintergrund-Features
    ax.add_feature(cfeature.LAND, facecolor='#f9f9f9')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#444444', zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=10)

    # Plotting je nach Parameter
    if sel_param == "Temperatur 2m (°C)":
        plot_data = data_array - 273.15 # Kelvin zu Celsius
        im = ax.pcolormesh(lons, lats, plot_data, cmap=mcolors.ListedColormap(T_COLORS), 
                           norm=mcolors.BoundaryNorm(T_LEVELS, ncolors=len(T_COLORS)), shading='nearest', zorder=5)
        plt.colorbar(im, label="°C", shrink=0.5, pad=0.02)
    
    elif sel_param == "Windböen (km/h)":
        plot_data = data_array * 3.6 # m/s zu km/h
        im = ax.pcolormesh(lons, lats, plot_data, cmap=mcolors.ListedColormap(W_COLORS), 
                           norm=mcolors.BoundaryNorm(W_LEVELS, ncolors=len(W_COLORS)), shading='nearest', zorder=5)
        plt.colorbar(im, label="km/h", shrink=0.5, pad=0.02)
        
    else: # Signifikantes Wetter
        grid = np.zeros_like(data_array)
        c_list = ['#FFFFFF00']
        for i, (label, (color, codes)) in enumerate(WW_LEGEND.items(), 1):
            for code in codes: grid[data_array == code] = i
            c_list.append(color)
        
        ax.pcolormesh(lons, lats, grid, cmap=mcolors.ListedColormap(c_list), alpha=0.8, shading='nearest', zorder=5)
        
        patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND.items()]
        leg = ax.legend(handles=patches, loc='lower left', title="Wetter", fontsize='x-small', framealpha=0.9)
        leg.set_zorder(20)

    # Info-Text oben in die Karte einfügen
    info_box = (f"Region: {sel_region}\n"
                f"Datum: {valid_time.strftime('%d.%m.%Y %H:00')} UTC\n"
                f"Modell: {sel_model} ({run_dt.strftime('%H')}Z Lauf)")
    
    ax.text(0.03, 0.97, info_box, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'), zorder=25)

    st.pyplot(fig)
    
    # Temporäre Datei löschen um RAM zu sparen
    if os.path.exists(f"{p_map[sel_param]}.grib"):
        os.remove(f"{p_map[sel_param]}.grib")
else:
    st.warning("🔄 Suche nach dem aktuellsten Modelllauf beim DWD... bitte kurz warten.")
