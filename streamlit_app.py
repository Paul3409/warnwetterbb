import streamlit as st
import xarray as xr
import requests
import bz2
import os
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import gzip
import re

# ==============================================================================
# 1. INITIALISIERUNG & KONTROLL-EINSTELLUNGEN
# ==============================================================================
# Wir setzen die Seite auf Breitbild, damit die Karten ihre volle Wirkung entfalten.
st.set_page_config(
    page_title="WarnwetterBB | Pro-Zentrale", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Autorefresh-Logik für Live-Betrieb (Optional, falls Bibliothek vorhanden)
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

def cleanup_temp_files():
    """Löscht temporäre GRIB-Reste, um Speicherplatz im Container zu sparen."""
    temp_files = ["temp.grib", "temp_gfs.grib", "temp_ecmwf.grib", "temp.grib.idx", "temp_gfs.grib.idx"]
    for file in temp_files:
        if os.path.exists(file):
            try: os.remove(file)
            except: pass

# Zeitzonen-Management
LOCAL_TZ = ZoneInfo("Europe/Berlin")
WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

# ==============================================================================
# 2. PROFI-FARBSKALEN (MASTER-DEFINITIONS)
# ==============================================================================
# Skala für Temperatur und Taupunkt (-30 bis +30 Grad)
temp_colors = [
    (0.0, '#D3D3D3'), (5/60, '#FFFFFF'), (10/60, '#FFC0CB'), (15/60, '#FF00FF'),
    (20/60, '#800080'), (20.01/60, '#00008B'), (25/60, '#0000CD'), (29.99/60, '#ADD8E6'),
    (30/60, '#006400'), (35/60, '#008000'), (39/60, '#90EE90'), (39.99/60, '#90EE90'),
    (40/60, '#FFFF00'), (45/60, '#FFA500'), (50/60, '#FF0000'), (55/60, '#8B0000'), (60/60, '#800080')
]
cmap_temp = mcolors.LinearSegmentedColormap.from_list("custom_temp", temp_colors)

# Präzise Niederschlagsskala gemäß meteorologischen Standards
precip_values = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3, 4, 5, 8, 12, 15, 20, 30, 40, 50]
precip_colors = [
    '#FFFFFF', '#87CEEB', '#1E90FF', '#191970', '#006400', '#32CD32', '#FFFF00', 
    '#FFA500', '#FF0000', '#8B0000', '#800000', '#4B0082', '#800080', '#9400D3', 
    '#7B68EE', '#FFFFFF'  
]
cmap_precip = mcolors.LinearSegmentedColormap.from_list("custom_precip", list(zip([v/50.0 for v in precip_values], precip_colors)))
norm_precip = mcolors.Normalize(vmin=0, vmax=50.0)

# Cloud Base / Ceiling (Aviation)
base_levels = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 8000]
base_colors = ['#FF00FF', '#FF0000', '#FFA500', '#FFFF00', '#ADFF2F', '#32CD32', '#00BFFF', '#1E90FF', '#0000FF', '#A9A9A9', '#FFFFFF']
cmap_base = mcolors.ListedColormap(base_colors)
norm_base = mcolors.BoundaryNorm(base_levels, cmap_base.N)

# Radar-Reflektivität (dBZ)
radar_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
radar_colors = ['#FFFFFF', '#B0E0E6', '#00BFFF', '#0000FF', '#00FF00', '#32CD32', '#008000', '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#800080', '#4B0082', '#E6E6FA']
cmap_radar = mcolors.ListedColormap(radar_colors)
norm_radar = mcolors.BoundaryNorm(radar_levels, cmap_radar.N)

# CAPE Energie-Skala
cape_levels = [0, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]
cape_colors = ['#006400', '#2E8B57', '#ADFF2F', '#FFFF00', '#FFB347', '#FFA500', '#FF4500', '#FF0000', '#8B0000', '#800080', '#FF00FF', '#FFFFFF', '#808080', '#404040']
cmap_cape = mcolors.ListedColormap(cape_colors)
norm_cape = mcolors.BoundaryNorm(cape_levels, cmap_cape.N)

# Weitere Hilfs-Colormaps
cmap_cin = mcolors.LinearSegmentedColormap.from_list("cin", ['#FFFFFF', '#ADD8E6', '#0000FF', '#00008B', '#000000'])
cmap_clouds = mcolors.LinearSegmentedColormap.from_list("clouds", ['#1E90FF', '#87CEEB', '#D3D3D3', '#FFFFFF'])
cmap_relhum = mcolors.LinearSegmentedColormap.from_list("relhum", ['#8B4513', '#F4A460', '#FFFFE0', '#90EE90', '#008000', '#0000FF'])
cmap_snow = mcolors.LinearSegmentedColormap.from_list("snow", ['#CCFFCC', '#FFFFFF', '#ADD8E6', '#0000FF', '#800080'])
cmap_vis = mcolors.LinearSegmentedColormap.from_list("vis", ['#FFFFFF', '#D3D3D3', '#87CEEB', '#1E90FF'])
cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind", ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082'])

# Wetter-Codes (DWD ww-Keys)
WW_LEGEND_DATA = {
    "Nebel": ("#FFFF00", list(range(40, 50))),
    "Regen leicht": ("#00FF00", [50, 51, 58, 60, 80]),
    "Regen mäßig": ("#228B22", [53, 61, 62, 81]),
    "Regen stark": ("#006400", [54, 55, 63, 64, 65, 82]),
    "gefr. Regen leicht": ("#FF7F7F", [56, 66]),
    "gefr. Regen mäßig/stark": ("#FF0000", [57, 67]),
    "Schneeregen": ("#FFB347", [68, 69, 83, 84]),
    "Schnee leicht": ("#87CEEB", [70, 71, 85]),
    "Schnee mäßig/stark": ("#00008B", [72, 73, 74, 75, 86, 87, 88]),
    "Gewitter": ("#800080", [95, 96, 97, 99])
}
cmap_ww = mcolors.ListedColormap(['#FFFFFF00'] + [c for l, (c, _) in WW_LEGEND_DATA.items()])

# ==============================================================================
# 3. ROUTING & ZEIT-BERECHNUNG
# ==============================================================================
MODEL_ROUTER = {
    "DWD Echtzeit-Radar": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)"],
        "params": ["Live-Radar (dBZ)"]
    },
    "ICON-D2": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum"],
        "params": ["Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
                   "500 hPa Geopot. Höhe", "850 hPa Temp.", "Niederschlag (mm)", "CAPE (J/kg)", 
                   "Gesamtbedeckung (%)", "Signifikantes Wetter", "Wolkenuntergrenze (m)", "Simuliertes Radar (dBZ)"]
    },
    "ICON-EU": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Europa"],
        "params": ["Temperatur 2m (°C)", "Windböen (km/h)", "Niederschlag (mm)", "Gesamtbedeckung (%)"]
    },
    "GFS (NOAA)": {
        "regions": ["Deutschland", "Europa"],
        "params": ["Temperatur 2m (°C)", "Niederschlag (mm)", "CAPE (J/kg)", "Bodendruck (hPa)"]
    }
}

def estimate_latest_run(model, now_utc):
    """Berechnet den voraussichtlich verfügbaren Modelllauf unter Berücksichtigung von Delays."""
    if "D2" in model:
        # ICON-D2 läuft alle 3h, hat aber ca. 90 Min Delay bis zum Upload
        run_h = ((now_utc.hour - 2) // 3) * 3
        if run_h < 0: return (now_utc - timedelta(days=1)).replace(hour=21, minute=0, second=0)
        return now_utc.replace(hour=run_h, minute=0, second=0)
    elif "Radar" in model:
        return now_utc
    else:
        # GFS / ICON-Global alle 6h
        run_h = ((now_utc.hour - 4) // 6) * 6
        if run_h < 0: return (now_utc - timedelta(days=1)).replace(hour=18, minute=0, second=0)
        return now_utc.replace(hour=run_h, minute=0, second=0)

# ==============================================================================
# 4. DATA ENGINE: RADAR & GRIB FETCH
# ==============================================================================
def fetch_radar_data():
    """Zieht das binäre WN-Produkt vom DWD OpenData Server."""
    url_base = "https://opendata.dwd.de/weather/radar/composit/wn/"
    try:
        r_list = requests.get(url_base, timeout=5)
        files = re.findall(r'href="([^"]+?\.bin\.gz)"', r_list.text)
        if not files: return None, None, None, None
        # Wir nehmen die zweitletzte Datei, falls die letzte noch hochgeladen wird
        latest_file = sorted(files)[-2] 
        ts_match = re.search(r'(\d{10})', latest_file)
        ts_str = ts_match.group(0) if ts_match else "Unbekannt"
        
        r = requests.get(url_base + latest_file, timeout=10)
        if r.status_code == 200:
            with gzip.open(io.BytesIO(r.content), 'rb') as f:
                f.read(540) # Header überspringen
                raw = np.frombuffer(f.read(), dtype=np.uint8)
                data = raw[:810000].reshape(900, 900).astype(float)
                data = (data - 64) / 2.0
                data[data < 0] = np.nan
                
                # Koordinaten für das WN-Produkt (Deutschland-Grid)
                lons = np.linspace(2.0, 16.0, 900)
                lats = np.linspace(46.0, 56.0, 900)
                lon_g, lat_g = np.meshgrid(lons, lats)
                return data[::-1], lon_g, lat_g, ts_str
    except: return None, None, None, None
    return None, None, None, None

@st.cache_data(ttl=300, show_spinner=False)
def get_weather_data(model, param, hr, debug=False):
    """Hauptfunktion zum Laden der Wetterdaten mit automatischer Fehlerkorrektur."""
    if "Radar" in model:
        d, lo, la, ts = fetch_radar_data()
        return d, lo, la, ts, ["DWD-Radar-Server"]

    p_map = {
        "Temperatur 2m (°C)": "t_2m", "Windböen (km/h)": "vmax_10m", "Niederschlag (mm)": "tot_prec",
        "Bodendruck (hPa)": "pmsl", "CAPE (J/kg)": "cape_ml", "Signifikantes Wetter": "ww",
        "Gesamtbedeckung (%)": "clct", "Simuliertes Radar (dBZ)": "dbz_cmax",
        "500 hPa Geopot. Höhe": "fi", "850 hPa Temp.": "t", "Wolkenuntergrenze (m)": "ceiling",
        "Taupunkt 2m (°C)": "td_2m"
    }
    key = p_map.get(param, "t_2m")
    now = datetime.now(timezone.utc)
    logs = []

    # Modell-Routing Logik (DWD NWP)
    if "ICON" in model:
        m_dir = "icon-d2" if "D2" in model else "icon-eu"
        reg = "icon-d2_germany" if "D2" in model else "icon-eu_europe"
        
        # Versuche die letzten 3 Läufe (Fallback-Strategie)
        for offset in [0, 3, 6]:
            base_t = estimate_latest_run(model, now) - timedelta(hours=offset)
            dt_s = base_t.strftime("%Y%m%d%H")
            l_type = "pressure-level" if key in ["fi", "t"] else "single-level"
            lvl = "500_" if "500" in param else "850_" if "850" in param else "2d_"
            
            url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{base_t.hour:02d}/{key}/{reg}_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{lvl}{key}.grib2.bz2"
            logs.append(url)
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    with bz2.open(io.BytesIO(r.content)) as f_bz2:
                        with open("temp.grib", "wb") as f_out: f_out.write(f_bz2.read())
                    ds = xr.open_dataset("temp.grib", engine='cfgrib')
                    v_name = list(ds.data_vars)[0]
                    ds_var = ds[v_name]
                    if 'isobaricInhPa' in ds_var.dims:
                        ds_var = ds_var.sel(isobaricInhPa=(500 if "500" in param else 850))
                    data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                    return data, lons, lats, dt_s, logs
            except: continue
            
    return None, None, None, None, logs

# ==============================================================================
# 5. SIDEBAR & INTERFACE
# ==============================================================================
with st.sidebar:
    st.title("🛰️ WarnwetterBB")
    sel_model = st.selectbox("Modell", list(MODEL_ROUTER.keys()))
    sel_region = st.selectbox("Region", MODEL_ROUTER[sel_model]["regions"])
    sel_param = st.radio("Parameter", MODEL_ROUTER[sel_model]["params"])
    
    if "Radar" not in sel_model:
        hr_list = list(range(1, 25)) if "D2" in sel_model else list(range(3, 75, 3))
        sel_hour = st.select_slider("Vorhersage-Stunde", options=hr_list)
    else:
        st.info("Live-Radar: Keine Vorhersage möglich.")
        sel_hour = 0
        
    st.markdown("---")
    show_isobars = st.checkbox("Isobaren", value=True)
    show_storms = st.checkbox("⚡ Gewitter-Schraffur", value=True)
    auto_up = st.checkbox("🔄 Auto-Update", value=False)
    if auto_up and st_autorefresh:
        st_autorefresh(interval=300000, key="refresh")
        
    btn_gen = st.button("🚀 Karte berechnen", use_container_width=True)

# ==============================================================================
# 6. GENERIERUNG & PLOTTING
# ==============================================================================
if btn_gen or (auto_up and "Radar" in sel_model):
    data, lons, lats, run_id, d_logs = get_weather_data(sel_model, sel_param, sel_hour)
    
    if data is not None:
        fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
        
        # Regionale Extents
        ext = {"Deutschland": [5.5, 15.5, 47.0, 55.5], "Brandenburg/Berlin": [11.0, 15.0, 51.0, 54.0], "Europa": [-15, 35, 35, 65]}
        ax.set_extent(ext.get(sel_region, ext["Deutschland"]))
        
        # Features
        ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='black', zorder=10)
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black', zorder=10)
        ax.add_feature(cfeature.LAKES, alpha=0.3, zorder=1)
        
        # Plotting je nach Parameter
        if "Temperatur" in sel_param or "850" in sel_param:
            val = data - 273.15 if data.max() > 100 else data
            im = ax.pcolormesh(lons, lats, val, cmap=cmap_temp, norm=mcolors.Normalize(-25, 35), shading='auto', zorder=5)
            plt.colorbar(im, label="°C", shrink=0.5)
            
        elif "Radar" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_radar, norm=norm_radar, shading='auto', zorder=5)
            plt.colorbar(im, label="dBZ", shrink=0.5)
            
        elif "Niederschlag" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_precip, norm=norm_precip, shading='auto', zorder=5)
            plt.colorbar(im, label="mm / h", shrink=0.5)
            
        elif "Windböen" in sel_param:
            im = ax.pcolormesh(lons, lats, data * 3.6 if data.max() < 100 else data, cmap=cmap_wind, norm=mcolors.Normalize(0, 120), shading='auto', zorder=5)
            plt.colorbar(im, label="km/h", shrink=0.5)

        elif "CAPE" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_cape, norm=norm_cape, shading='auto', zorder=5)
            plt.colorbar(im, label="J/kg", shrink=0.5)

        elif "Signifikantes Wetter" in sel_param:
            grid = np.zeros_like(data)
            for i, (l, (c, codes)) in enumerate(WW_LEGEND_DATA.items(), 1):
                for code in codes: grid[data == code] = i
            ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, shading='nearest', zorder=5)
            patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND_DATA.items()]
            ax.legend(handles=patches, loc='lower left', fontsize=7, framealpha=0.8).set_zorder(50)

        # Isobaren einzeichnen (falls verfügbar)
        if show_isobars and "Radar" not in sel_model:
            iso_d, ilo, ila, _, _ = get_weather_data(sel_model, "Bodendruck (hPa)", sel_hour)
            if iso_d is not None:
                p_hpa = iso_d / 100 if iso_d.max() > 5000 else iso_d
                cs = ax.contour(ilo, ila, p_hpa, colors='black', levels=np.arange(960, 1050, 4), linewidths=0.8, zorder=20)
                ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

        # Header Info
        time_now = datetime.now(LOCAL_TZ).strftime("%d.%m.%Y %H:%M")
        ax.text(0.01, 0.99, f"{sel_model} | {sel_param}\nLauf: {run_id} | Erstellt: {time_now}", 
                transform=ax.transAxes, va='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
        
        st.pyplot(fig)
        
        # Download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        st.download_button("📥 Bild speichern", buf.getvalue(), f"Wetter_{run_id}.png", "image/png")
    else:
        st.error(f"⚠️ Keine Daten gefunden für {sel_model} - {sel_param}.")
        st.info("Häufige Ursache: Der Modelllauf wird gerade erst hochgeladen (Delay). Bitte versuche es in 5 Minuten erneut.")

cleanup_temp_files()
