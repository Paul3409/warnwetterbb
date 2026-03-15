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

# --- 1. SETUP ---
st.set_page_config(page_title="WarnwetterBB | Multi-Modell", layout="wide")

# --- 2. FARBSKALEN (DEINE VORGABE) ---
T_LEVELS = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
T_COLORS = ['#FF00FF', '#800080', '#00008B', '#0000FF', '#ADD8E6', '#006400', '#008000', '#ADFF2F', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080']
cmap_temp = mcolors.LinearSegmentedColormap.from_list("temp", T_COLORS, N=256)

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
cmap_ww = mcolors.ListedColormap(['#FFFFFF00'] + [c for c, _ in WW_LEGEND_DATA.values()])

W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']
cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind", W_COLORS, N=256)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Steuerung")
    sel_region = st.selectbox("Region wählen", ["Deutschland", "Brandenburg/Berlin", "Europa"], key="reg_sel")
    sel_model = st.selectbox("Modell wählen", ["ICON-D2 (3h)", "ICON-D2-RUC (1h)"], index=1)
    sel_param = st.selectbox("Parameter wählen", ["Temperatur 2m (°C)", "Signifikantes Wetter", "Windböen (km/h)"], index=0)
    
    max_h = 27 if "RUC" in sel_model else 48
    sel_hour = st.slider("Stunde (+h)", 1, max_h, 1)
    st.markdown("---")
    st.caption("Modell-Suche: aktiv")

# --- 4. DER VERBESSERTE MULTI-LOADER ---
@st.cache_data(ttl=600) # Nur 10 Min Cache, um neue Daten schnell zu finden
def fetch_bulletproof(model_choice, p_key, hr):
    is_ruc = "RUC" in model_choice
    m_path = "icon-d2-ruc" if is_ruc else "icon-d2"
    m_prefix = "icon-d2-ruc" if is_ruc else "icon-d2"
    
    # Such-Logik: Wir gehen bis zu 12 Stunden zurück
    now_utc = datetime.now(timezone.utc)
    
    for off in range(1, 12):
        check_time = now_utc - timedelta(hours=off)
        # RUC stündlich, D2 alle 3h
        run = check_time.hour if is_ruc else (check_time.hour // 3) * 3
        dt_s = check_time.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
        
        url = f"https://opendata.dwd.de/weather/nwp/{m_path}/grib/{run:02d}/{p_key}/{m_prefix}_germany_regular-lat-lon_single-level_{dt_s}_{hr:03d}_2d_{p_key}.grib2.bz2"
        
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open("latest.bz2", "wb") as f: f.write(r.content)
                with bz2.open("latest.bz2", "rb") as f_in, open("latest.grib", "wb") as f_out:
                    f_out.write(f_in.read())
                
                ds = xr.open_dataset("latest.grib", engine='cfgrib')
                var = list(ds.data_vars)[0]
                data = ds[var].isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                
                lons_raw, lats_raw = ds.longitude.values, ds.latitude.values
                lons, lats = np.meshgrid(lons_raw, lats_raw) if lons_raw.ndim == 1 else (lons_raw, lats_raw)
                
                return data, lons, lats, dt_s
        except: continue
    return None, None, None, None

# --- 5. HAUPTTEIL ---
p_map = {"Signifikantes Wetter": "ww", "Temperatur 2m (°C)": "t_2m", "Windböen (km/h)": "vmax_10m"}

with st.spinner(f'🛰️ Suche Daten für {sel_model}...'):
    data, lons, lats, run_info = fetch_bulletproof(sel_model, p_map[sel_param], sel_hour)

if data is not None:
    valid_time = datetime.strptime(run_info, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)
    fig, ax = plt.subplots(figsize=(7, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
    
    ext = {"Deutschland": [5.8, 15.2, 47.2, 55.1], "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6], "Europa": [-10, 30, 35, 62]}
    ax.set_extent(ext[sel_region])

    # Geografie
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#333333', zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=10)
    states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
    ax.add_feature(states, linewidth=0.5, edgecolor='black', zorder=10)

    if sel_param == "Temperatur 2m (°C)":
        temp_c = data - 273.15
        im = ax.pcolormesh(lons, lats, temp_c, cmap=cmap_temp, norm=mcolors.Normalize(vmin=-25, vmax=45), shading='auto', zorder=5)
        plt.colorbar(im, label="°C", shrink=0.4, pad=0.02, aspect=30)
        
        skip = 24 if sel_region == "Deutschland" else 10
        ny, nx = lats.shape
        for i in range(0, ny, skip):
            for j in range(0, nx, skip):
                lat, lon, val = lats[i, j], lons[i, j], temp_c[i, j]
                if ext[sel_region][0] < lon < ext[sel_region][1] and ext[sel_region][2] < lat < ext[sel_region][3]:
                    ax.text(lon, lat, f'{val:.0f}', color='black', fontsize=7, ha='center', va='center', fontweight='bold', zorder=15,
                            bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.1))

    elif sel_param == "Windböen (km/h)":
        im = ax.pcolormesh(lons, lats, data * 3.6, cmap=cmap_wind, norm=mcolors.Normalize(vmin=0, vmax=150), shading='auto', zorder=5)
        plt.colorbar(im, label="km/h", shrink=0.4, pad=0.02, aspect=30)
        
    else: # Wetter
        grid = np.zeros_like(data)
        for i, (l, (c, codes)) in enumerate(WW_LEGEND_DATA.items(), 1):
            for code in codes: grid[data == code] = i
        ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, shading='nearest', zorder=5)
        patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND_DATA.items()]
        leg = ax.legend(handles=patches, loc='lower left', title="Wetter", fontsize='xx-small', framealpha=0.8)
        leg.set_zorder(25)

    info = f"Region: {sel_region}\nZeit: {valid_time.strftime('%d.%m. %H:00')} UTC\nModell: {sel_model} ({run_info[-2:]}Z)"
    ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'), zorder=30)
    st.pyplot(fig)
else:
    st.error("Wartungsarbeiten oder Server-Stau beim DWD. Die App findet aktuell keine verfügbaren GRIB-Dateien. Bitte probiere es in 5 Minuten noch einmal.")
