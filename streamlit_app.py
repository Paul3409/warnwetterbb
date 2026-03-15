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

# --- 1. SETUP & STYLE ---
st.set_page_config(page_title="WarnwetterBB | Analyse-Zentrum", layout="wide")

CITIES = {
    "Berlin": (13.40, 52.52), "Potsdam": (13.06, 52.40),
    "Cottbus": (14.33, 51.76), "Frankfurt (O)": (14.55, 52.34)
}

T_COLORS = ['#FF00FF','#800080','#00008B','#0000FF','#ADD8E6','#006400','#008000','#ADFF2F','#FFFF00','#FFD700','#FFA500','#FF0000','#8B0000','#800080']
cmap_temp = mcolors.LinearSegmentedColormap.from_list("temp", T_COLORS, N=256)
W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']
cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind", W_COLORS, N=256)
WW_LEGEND_DATA = {"Nebel": "#FFFF00", "Regen leicht": "#90EE90", "Regen stark": "#006400", "Schnee": "#0000FF", "Gewitter": "#800080"}

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("🛰️ Konfiguration")
    sel_model = st.selectbox("Modell", ["ICON-D2", "ICON-D2-RUC", "ICON-EU", "GFS (NOAA)", "ECMWF"])
    sel_region = st.selectbox("Region", ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"])
    
    p_opts = ["Temperatur 2m (°C)", "Windböen (km/h)", "500 hPa Geopot. Höhe", "850 hPa Temp."]
    if "ICON-D2" in sel_model: p_opts.append("Signifikantes Wetter")
    sel_param = st.selectbox("Parameter", p_opts)
    
    sel_hour = st.slider("Stunde (+h)", 1, 48 if "RUC" not in sel_model else 27, 1)
    show_isobars = st.checkbox("Isobaren (Luftdruck) anzeigen", value=True)
    
    st.markdown("---")
    # DER START-KNOPF
    generate = st.button("🚀 Karte generieren", use_container_width=True)

# --- 3. ROBUSTER MULTI-FETCH ---
@st.cache_data(ttl=600)
def fetch_any_model(model, param, hr):
    if "ECMWF" in model: return "ECMWF_WIP", None, None, None

    # Mapping interner DWD/NOAA Namen auf GRIB Parameter-ShortNames
    p_map = {
        "Temperatur 2m (°C)": "t_2m", "Windböen (km/h)": "vmax_10m",
        "500 hPa Geopot. Höhe": "fi", "850 hPa Temp.": "t", "Signifikantes Wetter": "ww", "Isobaren": "pmsl"
    }
    key = p_map[param]
    now = datetime.now(timezone.utc)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    # ICON MODELLE (DWD)
    if "ICON" in model:
        is_ruc = "RUC" in model
        m_dir = "icon-d2-ruc" if is_ruc else ("icon-d2" if "D2" in model else "icon-eu")
        step = 1 if is_ruc else 3
        
        for off in range(1, 10):
            t = now - timedelta(hours=off)
            run = t.hour if is_ruc else (t.hour // 3) * 3
            dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
            
            l_type = "pressure-level" if key in ["fi", "t"] else "single-level"
            lvl = "500" if "500" in param else ("850" if "850" in param else "")
            
            url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{m_dir}_germany_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{lvl + ('_' if lvl else '')}{key}.grib2.bz2"
            
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with bz2.open(io.BytesIO(r.content)) as f:
                        with open("temp.grib", "wb") as out: out.write(f.read())
                    ds = xr.open_dataset("temp.grib", engine='cfgrib')
                    var = list(ds.data_vars)[0]
                    # Logischer Dimensions-Fix: Erst isobaricInhPa wählen falls nötig, dann den Rest flachdrücken
                    ds_var = ds[var]
                    if 'isobaricInhPa' in ds_var.dims:
                        isobaric_idx = 0 if lvl == "500" else 1 # Simple index selection for common DWD pressure level files
                        data_flat = ds_var.isel(isobaricInhPa=isobaric_idx)
                    else: data_flat = ds_var
                    
                    drop_dims = {d: 0 for d in ['step', 'height', 'time', 'valid_time'] if d in data_flat.dims}
                    data = data_flat.isel(**drop_dims).values.squeeze()
                    
                    lons, lats = ds.longitude.values, ds.latitude.values
                    if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                    return data, lons, lats, dt_s
            except: continue

    # GFS (NOAA) - Nutzt den NOMADS Filter für Europa-Region
    elif "GFS" in model:
        for off in [3, 6, 9, 12]:
            t = now - timedelta(hours=off)
            run = (t.hour // 6) * 6
            dt_s = t.strftime("%Y%m%d")
            url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t{run:02d}z.pgrb2.0p25.f{hr:03d}&lev_2_m_above_ground=on&lev_10_m_above_ground=on&lev_500_mb=on&lev_850_mb=on&lev_mean_sea_level=on&var_TMP=on&var_HGT=on&var_GUST=on&var_PRMSL=on&subregion=&leftlon=0&rightlon=25&toplat=60&bottomlat=40&dir=%2Fgfs.{dt_s}%2F{run:02d}%2Fatmos"
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200:
                    with open("temp_gfs.grib", "wb") as out: out.write(r.content)
                    ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                    var = list(ds.data_vars)[0]
                    ds_var = ds[var]
                    
                    # Dimensions-Fix: Druckfläche explizit auswählen falls nötig
                    if 'isobaricInhPa' in ds_var.dims:
                        isobaric_idx = 0 if lvl == "500" else 1 # Assuming common structure for NOMADS data on 2 pressure levels
                        data_flat = ds_var.isel(isobaricInhPa=isobaric_idx)
                    else: data_flat = ds_var
                    
                    drop_dims = {d: 0 for d in ['step', 'height', 'time', 'valid_time', 'meanSea'] if d in data_flat.dims}
                    data = data_flat.isel(**drop_dims).values.squeeze()
                    
                    lons, lats = ds.longitude.values, ds.latitude.values
                    if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                    return data, lons, lats, f"{dt_s}{run:02d}"
            except: continue
            
    return None, None, None, None

# --- 4. KARTEN-GENERIERUNG ---
if generate:
    with st.spinner(f"📡 Lade Daten für {sel_model}..."):
        data, lons, lats, run_id = fetch_any_model(sel_model, sel_param, sel_hour)
        iso_data = None
        if show_isobars and run_id != "ECMWF_WIP":
            iso_data, ilons, ilats, _ = fetch_any_model(sel_model, "Isobaren", sel_hour)

    if run_id == "ECMWF_WIP":
        st.info("ECMWF OpenData Integration ist in Vorbereitung! Bitte vorerst ICON oder GFS nutzen.")
    elif data is not None:
        fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=120)
        
        ext = {"Deutschland": [5.8, 15.2, 47.2, 55.1], "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6], "Mitteleuropa (DE, PL)": [4.0, 25.0, 45.0, 56.0], "Alpenraum": [5.5, 17.0, 44.0, 49.5], "Europa": [-10, 35, 35, 65]}
        ax.set_extent(ext[sel_region])

        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='black', zorder=12)
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(states, linewidth=0.4, edgecolor='black', zorder=12)

        # PLOTTING
        # Temperatur (shading='auto' für maximale Stabilität gegen Dimensions-Gitter)
        if "Temperatur" in sel_param or "850 hPa Temp." in sel_param:
            # GFS GRIB Daten für Temperatur sind Kelvin. ICON meist Celsius, muss explizit umgerechnet werden.
            val_c = data - 273.15 # Explicitly assume and convert Kelvin for GRIB temperature on pressure levels
            
            im = ax.pcolormesh(lons, lats, val_c, cmap=cmap_temp, norm=mcolors.Normalize(vmin=-30, vmax=20), shading='auto', zorder=5) # Normalisierung angepasst für T850
            plt.colorbar(im, label="°C", shrink=0.4, pad=0.02)
            
            # Zahlen (Nur 2m Temp und kleinerer Zoom)
            if "2m" in sel_param and sel_region != "Europa":
                skip = 30 if "Deutschland" in sel_region else 10
                for i in range(0, lats.shape[0], skip):
                    for j in range(0, lons.shape[1], skip):
                        if ext[sel_region][0] < lons[i,j] < ext[sel_region][1] and ext[sel_region][2] < lats[i,j] < ext[sel_region][3]:
                            ax.text(lons[i,j], lats[i,j], f'{val_c[i,j]:.0f}', fontsize=7, fontweight='bold', ha='center', zorder=15)

        elif "Geopot" in sel_param:
            val = data / 10 if data.max() > 1000 else data # gpdm conversion if data is in gpm
            im = ax.pcolormesh(lons, lats, val, cmap='nipy_spectral', shading='auto', zorder=5)
            plt.colorbar(im, label="gpdm", shrink=0.4)

        # ISOBAREN
        if iso_data is not None:
            p_hpa = iso_data / 100 if iso_data.max() > 5000 else iso_data #Pa to hPa ifPa
            if ilons.ndim == 1: ilons, ilats = np.meshgrid(ilons, ilats)
            cs = ax.contour(ilons, ilats, p_hpa, colors='black', linewidths=0.7, levels=np.arange(940, 1060, 4), zorder=20)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

        # CITIES (Kept for iterative improvement unless explicitly asked to remove)
        for name, (lon, lat) in CITIES.items():
            ax.plot(lon, lat, 'ko', markersize=3, zorder=25)
            ax.text(lon + 0.05, lat + 0.05, name, fontsize=8, fontweight='bold', zorder=26, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.2))

        # Header: KLEINERER TEXT
        valid = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)
        info = f"{sel_model} | {sel_param}\nTermin: {valid.strftime('%d.%m. %H:00')} UTC\nLauf: {run_id[-2:]}Z"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round', pad=0.1), zorder=30)

        st.pyplot(fig)
    else:
        st.error(f"Modell {sel_model} ist für diesen Zeitschritt gerade nicht auf dem Server erreichbar.")
else:
    st.info("Wähle deine Einstellungen links und klicke auf 'Karte generieren'.")
