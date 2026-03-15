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

# Städte-Koordinaten
CITIES = {
    "Berlin": (13.40, 52.52),
    "Potsdam": (13.06, 52.40),
    "Cottbus": (14.33, 51.76),
    "Frankfurt (O)": (14.55, 52.34)
}

# Farbskalen
T_COLORS = ['#FF00FF','#800080','#00008B','#0000FF','#ADD8E6','#006400','#008000','#ADFF2F','#FFFF00','#FFD700','#FFA500','#FF0000','#8B0000','#800080']
cmap_temp = mcolors.LinearSegmentedColormap.from_list("temp", T_COLORS, N=256)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("🛰️ Modell-Konfiguration")
    sel_model = st.selectbox("Modell", ["ICON-D2", "ICON-D2-RUC", "ICON-EU", "GFS (NOAA)", "ECMWF"])
    sel_region = st.selectbox("Region", ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"])
    
    # Parameter-Logik
    p_opts = ["Temperatur 2m (°C)", "Windböen (km/h)", "500 hPa Geopot. Höhe", "850 hPa Temp."]
    if "ICON-D2" in sel_model: p_opts.append("Signifikantes Wetter")
    sel_param = st.selectbox("Parameter", p_opts)
    
    sel_hour = st.slider("Stunde (+h)", 1, 48 if "RUC" not in sel_model else 27, 1)
    show_isobars = st.checkbox("Isobaren (Luftdruck) anzeigen", value=True)
    
    st.markdown("---")
    generate = st.button("🚀 Karte generieren", use_container_width=True)

# --- 3. MULTI-FETCH LOGIK (DWD / NOAA / ECMWF) ---
@st.cache_data(ttl=900)
def fetch_any_model(model, param, hr):
    p_map = {
        "Temperatur 2m (°C)": "t_2m", "Windböen (km/h)": "vmax_10m",
        "500 hPa Geopot. Höhe": "fi", "850 hPa Temp.": "t", "Signifikantes Wetter": "ww", "Isobaren": "pmsl"
    }
    key = p_map[param]
    now = datetime.now(timezone.utc)

    # A: DWD MODELLE (ICON)
    if "ICON" in model:
        m_dir = "icon-d2-ruc" if "RUC" in model else ("icon-d2" if "D2" in model else "icon-eu")
        step = 1 if "RUC" in model else 3
        for off in range(1, 10):
            t = now - timedelta(hours=off)
            run = t.hour if step == 1 else (t.hour // 3) * 3
            dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
            
            # Pfad-Logik für Pressure Levels
            l_type = "pressure-level" if key in ["fi", "t"] else "single-level"
            lvl = "500" if "500" in param else ("850" if "850" in param else "")
            
            url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{m_dir}_germany_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{lvl + ('_' if lvl else '')}{key}.grib2.bz2"
            
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with bz2.open(io.BytesIO(r.content)) as f:
                        with open("temp.grib", "wb") as out: out.write(f.read())
                    return xr.open_dataset("temp.grib", engine='cfgrib'), dt_s
            except: continue

    # B: GFS (NOAA) - Nutzt den NOMADS Filter (Schnell!)
    elif "GFS" in model:
        for off in [3, 6, 9, 12]:
            t = now - timedelta(hours=off)
            run = (t.hour // 6) * 6
            dt_s = t.strftime("%Y%m%d")
            # GFS ist global, wir brauchen nur Europa-Ausschnitt (Filter)
            url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t{run:02d}z.pgrb2.0p25.f{hr:03d}&lev_2_m_above_ground=on&lev_10_m_above_ground=on&lev_500_mb=on&lev_850_mb=on&var_TMP=on&var_HGT=on&var_GUST=on&var_PRMSL=on&subregion=&leftlon=0&rightlon=20&toplat=56&bottomlat=45&dir=%2Fgfs.{dt_s}%2F{run:02d}%2Fatmos"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    with open("temp.grib", "wb") as out: out.write(r.content)
                    return xr.open_dataset("temp.grib", engine='cfgrib'), f"{dt_s}{run:02d}"
            except: continue
            
    return None, None

# --- 4. KARTEN-GENERIERUNG ---
if generate:
    with st.spinner(f"📡 Berechne {sel_model}..."):
        ds, run_id = fetch_any_model(sel_model, sel_param, sel_hour)
        ds_iso, _ = fetch_any_model(sel_model, "Isobaren", sel_hour) if show_isobars else (None, None)

    if ds:
        # Daten-Extraktion
        var = list(ds.data_vars)[0]
        # .squeeze() und .isel fixen den Dimensions-Error
        data = ds[var].isel(step=0, height=0, missing_dims='ignore').values.squeeze()
        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values) if ds.longitude.ndim == 1 else (ds.longitude.values, ds.latitude.values)

        fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=120)
        
        # Regionen
        ext = {
            "Deutschland": [5.8, 15.2, 47.2, 55.1],
            "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6],
            "Mitteleuropa (DE, PL)": [5.0, 25.0, 46.0, 56.0],
            "Alpenraum": [5.5, 17.0, 44.0, 49.5],
            "Europa": [-10, 35, 35, 65]
        }
        ax.set_extent(ext[sel_region])

        # Features
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='black', zorder=12)
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(states, linewidth=0.4, edgecolor='black', zorder=12)

        # Plotting Parameter
        if "Temperatur" in sel_param or "850 hPa Temp." in sel_param:
            val = data - 273.15 if data.max() > 100 else data
            im = ax.pcolormesh(lons, lats, val, cmap=cmap_temp, norm=mcolors.Normalize(vmin=-25, vmax=45), shading='gouraud', zorder=5)
            plt.colorbar(im, label="°C", shrink=0.4, pad=0.02)
            
            # Zahlen (Nur 2m Temp & kleinerer Zoom)
            if "2m" in sel_param and sel_region != "Europa":
                skip = 30 if "Deutschland" in sel_region else 10
                for i in range(0, lats.shape[0], skip):
                    for j in range(0, lons.shape[1], skip):
                        ax.text(lons[i,j], lats[i,j], f'{val[i,j]:.0f}', fontsize=7, fontweight='bold', ha='center', zorder=15)

        elif "Geopot" in sel_param:
            im = ax.pcolormesh(lons, lats, data/10, cmap='nipy_spectral', shading='gouraud', zorder=5)
            plt.colorbar(im, label="gpdm", shrink=0.4)

        # ISOBAREN
        if ds_iso:
            p_data = ds_iso[list(ds_iso.data_vars)[0]].isel(step=0, height=0, missing_dims='ignore').values.squeeze()
            p_hpa = p_data / 100 if p_data.max() > 5000 else p_data
            cs = ax.contour(lons, lats, p_hpa, colors='black', linewidths=0.7, levels=np.arange(940, 1060, 4), zorder=20)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

        # STÄDTE
        for name, (lon, lat) in CITIES.items():
            ax.plot(lon, lat, 'ko', markersize=3, zorder=25)
            ax.text(lon + 0.1, lat + 0.1, name, fontsize=8, fontweight='bold', zorder=25, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Header
        valid = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)
        info = f"{sel_model} | {sel_param}\nTermin: {valid.strftime('%d.%m. %H:00')} UTC\nLauf: {run_id[-2:]}Z"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8), zorder=30)

        st.pyplot(fig)
    else:
        st.error("Keine Daten gefunden. Das Modell wurde für diesen Zeitraum noch nicht auf dem Server veröffentlicht.")
else:
    st.info("Wähle deine Einstellungen und klicke auf 'Karte generieren'.")
