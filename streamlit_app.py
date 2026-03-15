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

# --- 1. CONFIG ---
st.set_page_config(page_title="WarnwetterBB | Analyse-Zentrum", layout="wide")

# --- 2. FARBSKALEN (KONTINUIERLICH & INTERPOLIERT) ---
T_LEVELS = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
T_COLORS = ['#FF00FF', '#800080', '#00008B', '#0000FF', '#ADD8E6', '#006400', '#008000', '#ADFF2F', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080']
cmap_temp = mcolors.LinearSegmentedColormap.from_list("temp", T_COLORS, N=256)

W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']
cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind", W_COLORS, N=256)

WW_COLORS = {"Nebel": "#FFFF00", "Regen": "#00FF00", "Schnee": "#0000FF", "Gewitter": "#800080"}

# --- 3. STEUERUNG (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Konfiguration")
    
    sel_model = st.selectbox("Modell wählen", ["ICON-D2", "ICON-D2-RUC", "ICON-EU", "GFS (NOAA)", "ECMWF"])
    
    # Regionauswahl
    sel_region = st.selectbox("Region wählen", ["Deutschland", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"])
    
    # Dynamische Parameter
    param_options = ["Temperatur 2m (°C)", "Windböen (km/h)", "500 hPa Geopot. Höhe", "850 hPa Temp."]
    if "ICON-D2" in sel_model:
        param_options.insert(1, "Signifikantes Wetter")
    sel_param = st.selectbox("Parameter wählen", param_options)
    
    sel_hour = st.slider("Stunde (+h)", 1, 48 if "RUC" not in sel_model else 27, 1)
    
    st.markdown("---")
    show_isobars = st.checkbox("Isobaren (pmsl) einblenden", value=False)
    
    st.markdown("---")
    # START-KNOPF
    load_btn = st.button("🚀 Karte generieren", use_container_width=True)
    st.caption("Daten: DWD OpenData / ECMWF / NOAA")

# --- 4. DATA-LOADER ---
@st.cache_data(ttl=900)
def fetch_data(model, param_name, hr):
    p_map = {
        "Temperatur 2m (°C)": ("t_2m", "single-level"),
        "Signifikantes Wetter": ("ww", "single-level"),
        "Windböen (km/h)": ("vmax_10m", "single-level"),
        "500 hPa Geopot. Höhe": ("fi", "pressure-level"),
        "850 hPa Temp.": ("t", "pressure-level"),
        "Isobaren": ("pmsl", "single-level")
    }
    key, l_type = p_map[param_name]
    m_dir = model.lower().replace(" ", "-").replace("(noaa)", "").strip()
    prefix = m_dir
    
    step = 1 if "RUC" in model else 3
    
    for off in range(1, 12):
        now = datetime.now(timezone.utc) - timedelta(hours=off)
        run = (now.hour // step) * step
        dt_s = now.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
        
        # URL Logik
        if l_type == "single-level":
            url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{prefix}_germany_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_2d_{key}.grib2.bz2"
        else:
            lvl = "500" if "500" in param_name else "850"
            url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{prefix}_germany_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{lvl}_{key}.grib2.bz2"
            
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with bz2.open(requests.get(url, stream=True).raw) as f_in:
                    with open(f"data_{key}.grib", "wb") as f: f.write(f_in.read())
                ds = xr.open_dataset(f"data_{key}.grib", engine='cfgrib')
                var = list(ds.data_vars)[0]
                data = ds[var].isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                lons_raw, lats_raw = ds.longitude.values, ds.latitude.values
                lons, lats = np.meshgrid(lons_raw, lats_raw) if lons_raw.ndim == 1 else (lons_raw, lats_raw)
                return data, lons, lats, dt_s
        except: continue
    return None, None, None, None

# --- 5. HAUPTTEIL ---
if load_btn:
    with st.spinner(f'🛰️ Analysiere {sel_model}...'):
        data, lons, lats, run_info = fetch_data(sel_model, sel_param, sel_hour)
        iso_data = None
        if show_isobars:
            iso_data, ilons, ilats, _ = fetch_data(sel_model, "Isobaren", sel_hour)

    if data is not None:
        fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=120)
        
        # REGIONEN-EXTENTS
        ext = {
            "Deutschland": [5.8, 15.2, 47.2, 55.1],
            "Mitteleuropa (DE, PL)": [5.0, 25.0, 46.0, 56.0],
            "Alpenraum": [5.5, 17.0, 44.0, 49.5],
            "Europa": [-12.0, 35.0, 35.0, 65.0]
        }
        ax.set_extent(ext[sel_region])

        # Features
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#333333', zorder=10)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=10)
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(states, linewidth=0.4, edgecolor='black', zorder=10)

        # Plotting
        if "Temperatur" in sel_param or "850 hPa" in sel_param:
            v_min, v_max = (-25, 45) if "2m" in sel_param else (-35, 25)
            val_c = data - 273.15 if data.max() > 100 else data
            im = ax.pcolormesh(lons, lats, val_c, cmap=cmap_temp, norm=mcolors.Normalize(vmin=v_min, vmax=v_max), shading='auto', zorder=5)
            plt.colorbar(im, label="°C", shrink=0.4, pad=0.02, aspect=30)
            
            # Zahlen (Nur bei 2m Temp)
            if "2m" in sel_param:
                skip = 25 if "Deutschland" in sel_region else 40
                if "Alpenraum" in sel_region: skip = 15
                ny, nx = lats.shape
                for i in range(0, ny, skip):
                    for j in range(0, nx, skip):
                        if ext[sel_region][0] < lons[i,j] < ext[sel_region][1] and ext[sel_region][2] < lats[i,j] < ext[sel_region][3]:
                            ax.text(lons[i,j], lats[i,j], f'{val_c[i,j]:.0f}', fontsize=6, fontweight='bold', ha='center', va='center', zorder=15, alpha=0.7, bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))

        elif "Geopot" in sel_param:
            im = ax.pcolormesh(lons, lats, data/10, cmap='terrain', shading='auto', zorder=5)
            plt.colorbar(im, label="gpdm", shrink=0.4, pad=0.02)

        elif "Windböen" in sel_param:
            im = ax.pcolormesh(lons, lats, data * 3.6, cmap=cmap_wind, norm=mcolors.Normalize(vmin=0, vmax=150), shading='auto', zorder=5)
            plt.colorbar(im, label="km/h", shrink=0.4, pad=0.02)

        # Isobaren Layer
        if iso_data is not None:
            p_hpa = iso_data / 100 if iso_data.max() > 5000 else iso_data
            cs = ax.contour(ilons, ilats, p_hpa, colors='black', linewidths=0.7, levels=np.arange(940, 1060, 4), zorder=20)
            ax.clabel(cs, inline=True, fontsize=7, fmt='%1.0f')

        # Info-Box
        valid = datetime.strptime(run_info, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)
        info = f"{sel_model} | {sel_param}\nValid: {valid.strftime('%d.%m. %H:00')} UTC\nRun: {run_info[-2:]}Z"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'), zorder=30)

        st.pyplot(fig)
        
        # Cleanup
        for f in [f"data_{list(p_map.values())[i][0]}.grib" for i in range(len(p_map))]:
            if os.path.exists(f): os.remove(f)
    else:
        st.error("Keine Daten gefunden. DWD-Server liefert für diesen Zeitraum aktuell nichts.")
else:
    # STARTBILDSCHIRM
    st.info("👋 Willkommen! Wähle links Modell und Region aus und klicke auf 'Karte generieren'.")
    st.image("https://img.freepik.com/vektoren-kostenlos/meteorologische-vorhersage-isometrische-zusammensetzung_1284-23114.jpg", width=400)
