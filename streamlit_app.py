import streamlit as st
import xarray as xr
import requests, bz2, os, io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
import numpy as np

# --- 1. SETUP ---
st.set_page_config(page_title="WarnwetterBB | Analyse-Zentrum", layout="wide")

CITIES = {
    "Berlin": (13.40, 52.52), "Potsdam": (13.06, 52.40),
    "Cottbus": (14.33, 51.76), "Frankfurt (O)": (14.55, 52.34)
}

T_COLORS = ['#FF00FF','#800080','#00008B','#0000FF','#ADD8E6','#006400','#008000','#ADFF2F','#FFFF00','#FFD700','#FFA500','#FF0000','#8B0000','#800080']
cmap_temp = mcolors.LinearSegmentedColormap.from_list("temp", T_COLORS, N=256)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("🛰️ Konfiguration")
    sel_model = st.selectbox("Modell", ["ICON-D2", "ICON-D2-RUC", "ICON-EU", "GFS (NOAA)"])
    sel_region = st.selectbox("Region", ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"])
    
    # Dynamische Parameter
    p_opts = ["Temperatur 2m (°C)", "Windböen (km/h)", "500 hPa Geopot. Höhe", "850 hPa Temp."]
    if "ICON-D2" in sel_model: p_opts.append("Signifikantes Wetter")
    sel_param = st.selectbox("Parameter", p_opts)
    
    sel_hour = st.slider("Stunde (+h)", 1, 48 if "RUC" not in sel_model else 27, 1)
    show_isobars = st.checkbox("Isobaren (Luftdruck) anzeigen", value=True)
    
    st.markdown("---")
    # DER NEUE START-KNOPF
    generate = st.button("🚀 Karte generieren", use_container_width=True)

# --- 3. ROBUSTER MULTI-FETCH ---
@st.cache_data(ttl=900)
def fetch_any_model(model, param, hr):
    p_map = {
        "Temperatur 2m (°C)": "t_2m", "Windböen (km/h)": "vmax_10m",
        "500 hPa Geopot. Höhe": "fi", "850 hPa Temp.": "t", 
        "Signifikantes Wetter": "ww", "Isobaren": "pmsl"
    }
    key = p_map[param]
    now = datetime.now(timezone.utc)

    # ICON MODELLE (DWD)
    if "ICON" in model:
        is_ruc = "RUC" in model
        m_dir = "icon-d2-ruc" if is_ruc else ("icon-d2" if "D2" in model else "icon-eu")
        # RUC nutzt stündliche Runs, D2/EU alle 3h
        step = 1 if is_ruc else 3
        
        for off in range(1, 12):
            t = now - timedelta(hours=off)
            run = t.hour if is_ruc else (t.hour // 3) * 3
            dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
            
            l_type = "pressure-level" if key in ["fi", "t"] else "single-level"
            lvl = "500" if "500" in param else ("850" if "850" in param else "")
            
            url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{m_dir}_germany_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{lvl + ('_' if lvl else '')}{key}.grib2.bz2"
            
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    with bz2.open(io.BytesIO(r.content)) as f:
                        with open("temp.grib", "wb") as out: out.write(f.read())
                    ds = xr.open_dataset("temp.grib", engine='cfgrib')
                    # Dimensionen sofort fixieren
                    var = list(ds.data_vars)[0]
                    data = ds[var].isel(step=0, height=0, missing_dims='ignore').squeeze()
                    return data.values, ds.longitude.values, ds.latitude.values, dt_s
            except: continue

    # GFS (NOAA) - Über NOMADS Filter
    elif "GFS" in model:
        for off in [3, 6, 9, 12]:
            t = now - timedelta(hours=off)
            run = (t.hour // 6) * 6
            dt_s = t.strftime("%Y%m%d")
            # NOMADS Filter für Europa-Region
            url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t{run:02d}z.pgrb2.0p25.f{hr:03d}&lev_2_m_above_ground=on&lev_10_m_above_ground=on&lev_500_mb=on&lev_850_mb=on&var_TMP=on&var_HGT=on&var_GUST=on&var_PRMSL=on&subregion=&leftlon=0&rightlon=20&toplat=56&bottomlat=44&dir=%2Fgfs.{dt_s}%2F{run:02d}%2Fatmos"
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200:
                    with open("temp.grib", "wb") as out: out.write(r.content)
                    ds = xr.open_dataset("temp.grib", engine='cfgrib')
                    var = list(ds.data_vars)[0]
                    data = ds[var].isel(step=0, height=0, missing_dims='ignore').squeeze()
                    return data.values, ds.longitude.values, ds.latitude.values, f"{dt_s}{run:02d}"
            except: continue
            
    return None, None, None, None

# --- 4. KARTEN-GENERIERUNG ---
if generate:
    with st.spinner(f"📡 Berechne {sel_model}..."):
        data, lons_raw, lats_raw, run_id = fetch_any_model(sel_model, sel_param, sel_hour)
        iso_data = None
        if show_isobars:
            iso_data, ilons, ilats, _ = fetch_any_model(sel_model, "Isobaren", sel_hour)

    if data is not None:
        # --- DIMENSIONS-CHECK & MESHGRID ---
        # Wichtig: shading='gouraud' braucht X, Y, C in gleicher 2D-Form
        if lons_raw.ndim == 1:
            lons, lats = np.meshgrid(lons_raw, lats_raw)
        else:
            lons, lats = lons_raw, lats_raw

        fig, ax = plt.subplots(figsize=(7, 9), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=120)
        
        ext = {
            "Deutschland": [5.8, 15.2, 47.2, 55.1],
            "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6],
            "Mitteleuropa (DE, PL)": [5.0, 22.0, 46.0, 56.0],
            "Alpenraum": [5.5, 17.0, 44.0, 49.5],
            "Europa": [-10, 35, 35, 65]
        }
        ax.set_extent(ext[sel_region])

        # Features
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='black', zorder=12)
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(states, linewidth=0.4, edgecolor='black', zorder=12)

        # Plotting
        if "Temperatur" in sel_param or "850 hPa" in sel_param:
            val = data - 273.15 if data.max() > 100 else data
            im = ax.pcolormesh(lons, lats, val, cmap=cmap_temp, norm=mcolors.Normalize(vmin=-25, vmax=45), shading='gouraud', zorder=5)
            plt.colorbar(im, label="°C", shrink=0.4, pad=0.02)
            
            # Zahlen (Nur bei Zoom auf Brandenburg/DE)
            if "2m" in sel_param and "Europa" not in sel_region:
                skip = 25 if "Deutschland" in sel_region else 8
                for i in range(0, lats.shape[0], skip):
                    for j in range(0, lons.shape[1], skip):
                        ax.text(lons[i,j], lats[i,j], f'{val[i,j]:.0f}', fontsize=7, fontweight='bold', ha='center', zorder=15)

        elif "Geopot" in sel_param:
            val = data / 10 if data.max() > 1000 else data # gpdm
            im = ax.pcolormesh(lons, lats, val, cmap='nipy_spectral', shading='gouraud', zorder=5)
            plt.colorbar(im, label="gpdm", shrink=0.4)

        # ISOBAREN
        if iso_data is not None:
            # GFS Isobaren sind oft in Pa (101325), ICON in hPa (1013)
            p_hpa = iso_data / 100 if iso_data.max() > 5000 else iso_data
            # Wir erzwingen auch hier Meshgrid falls nötig
            if ilons.ndim == 1: ilons, ilats = np.meshgrid(ilons, ilats)
            cs = ax.contour(ilons, ilats, p_hpa, colors='black', linewidths=0.6, levels=np.arange(960, 1050, 4), zorder=20)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

        # STÄDTE
        for name, (lon, lat) in CITIES.items():
            ax.plot(lon, lat, 'ko', markersize=3, zorder=25)
            ax.text(lon + 0.05, lat + 0.05, name, fontsize=8, fontweight='bold', zorder=26, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # Info-Box
        run_dt = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        valid = run_dt + timedelta(hours=sel_hour)
        info = f"{sel_model} | {sel_param}\nTermin: {valid.strftime('%d.%m. %H:00')} UTC\nLauf: {run_id[-2:]}Z"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8), zorder=30)

        st.pyplot(fig)
    else:
        st.error(f"Daten für {sel_model} konnten nicht geladen werden. Der DWD/NOAA Server hat diesen Lauf noch nicht freigegeben.")
else:
    st.info("Wähle deine Einstellungen links und klicke auf 'Karte generieren'.")
