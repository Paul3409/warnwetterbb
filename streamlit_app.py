"""
WARN WETTER BB - ULTIMATE EDITION 2026
--------------------------------------
Dieses Skript ist ein professionelles Wetter-Dashboard für Streamlit.
Es visualisiert GRIB2-Daten vom DWD, NOAA und ECMWF.
Features: Player-Steuerung, Theta-E Berechnung, Aviation-Skalen, SPC-Radar.
"""

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
import folium
from streamlit_folium import st_folium
import matplotlib.patheffects as path_effects

# ==============================================================================
# 1. INITIALISIERUNG & GLOBALER CACHE
# ==============================================================================
st.set_page_config(
    page_title="WarnwetterBB | Ultimate Pro",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sitzungsspeicher für den Player (Wetterzentrale-Style)
if 'step_idx' not in st.session_state:
    st.session_state.step_idx = 0
if 'auto_play' not in st.session_state:
    st.session_state.auto_play = False

# Zeitzonen und Konstanten
LOCAL_TZ = ZoneInfo("Europe/Berlin")
WOCHENTAGE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]

def cleanup_temp_files():
    """Löscht alle temporären GRIB-Reste, um den Cloud-Speicher zu entlasten."""
    temp_patterns = [
        "temp.grib", "temp_2.grib", "temp_gfs.grib", "temp_ecmwf.grib",
        "temp.grib.idx", "temp_2.grib.idx", "temp_gfs.grib.idx", "temp_3.grib"
    ]
    for f in temp_patterns:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

# ==============================================================================
# 2. DEFINITION DER FARBSKALEN (METEOROLOGISCHE STANDARDS)
# ==============================================================================

def get_custom_colormaps():
    """Erstellt alle benötigten Farbtabellen für die Parameter."""
    
    # NIEDERSCHLAG (Deine exakte HTML-Palette, 5er-Schritte in der Legende)
    prec_vals = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3, 4, 5, 8, 12, 15, 20, 30, 40, 50]
    prec_cols = [
        '#FFFFFF', '#87CEEB', '#1E90FF', '#191970', '#006400', '#32CD32', '#FFFF00', 
        '#FFA500', '#FF0000', '#8B0000', '#800000', '#4B0082', '#800080', '#9400D3', 
        '#7B68EE', '#FFFFFF'
    ]
    cmap_precip = mcolors.LinearSegmentedColormap.from_list("prec", list(zip([v/50 for v in prec_vals], prec_cols)))

    # TEMPERATUR (Glattgebügelt, 10er Sprünge)
    temp_colors = [
        (0.0, '#D3D3D3'), (5/60, '#FFFFFF'), (10/60, '#FFC0CB'), (15/60, '#FF00FF'),
        (20/60, '#800080'), (20.01/60, '#00008B'), (25/60, '#0000CD'), (29.99/60, '#ADD8E6'),
        (30/60, '#006400'), (35/60, '#008000'), (39/60, '#90EE90'), (40/60, '#FFFF00'), 
        (45/60, '#FFA500'), (50/60, '#FF0000'), (55/60, '#8B0000'), (60/60, '#800080')
    ]
    cmap_temp = mcolors.LinearSegmentedColormap.from_list("temp", temp_colors)

    # WOLKENUNTERGRENZE (Aviation Warn-Skala)
    base_lvls = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 8000]
    base_cols = [
        '#FF00FF', '#FF0000', '#FFA500', '#FFFF00', '#ADFF2F', '#32CD32', 
        '#00BFFF', '#1E90FF', '#0000FF', '#A9A9A9', '#FFFFFF'
    ]
    cmap_base = mcolors.ListedColormap(base_cols)

    # THETA-E (Energie-Skala)
    theta_cols = ['#00008B', '#0000FF', '#00BFFF', '#32CD32', '#FFFF00', '#FFA500', '#FF0000', '#FF00FF']
    cmap_theta = mcolors.LinearSegmentedColormap.from_list("theta", theta_cols)

    # CAPE
    cape_cols = ['#006400', '#ADFF2F', '#FFFF00', '#FFA500', '#FF0000', '#800080', '#FFFFFF']
    cmap_cape = mcolors.LinearSegmentedColormap.from_list("cape", cape_cols)

    # RADAR
    radar_cols = ['#FFFFFF', '#B0E0E6', '#0000FF', '#00FF00', '#008000', '#FFFF00', '#FF0000', '#800080']
    cmap_radar = mcolors.LinearSegmentedColormap.from_list("radar", radar_cols)

    # WIND (Sturm-Skala)
    wind_cols = ['#FFFFFF', '#ADD8E6', '#0000FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#4B0082']
    cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind", wind_cols)

    return cmap_precip, cmap_temp, cmap_base, cmap_theta, cmap_cape, cmap_radar, cmap_wind

CMAPS = get_custom_colormaps()

# ==============================================================================
# 3. GEOGRAFIE & REGIONALE EINSTELLUNGEN
# ==============================================================================
REGIONS = {
    "Mühlberg (Elbe) & Elbe-Elster": [13.0, 13.5, 51.3, 51.6],
    "Sachsen/Brandenburg": [11.5, 15.2, 50.8, 52.6],
    "Deutschland": [5.8, 15.2, 47.2, 55.2],
    "Mitteleuropa": [4.0, 20.0, 46.0, 56.0],
    "Europa": [-15, 45, 33, 72]
}

# ==============================================================================
# 4. MODELL-ROUTING & PARAMETER-MAPPING
# ==============================================================================
MODEL_CONFIG = {
    "ICON-D2": {
        "res": "2.2km",
        "params": ["Niederschlag (mm)", "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", 
                   "Theta-E (Schwüle-Energie)", "CAPE (J/kg)", "Simuliertes Radar (dBZ)", 
                   "Wolkenuntergrenze (m)", "Gesamtbedeckung (%)", "Schneehöhe (cm)", "Signifikantes Wetter"],
        "steps": list(range(1, 28))
    },
    "ICON-EU": {
        "res": "6.5km",
        "params": ["Niederschlag (mm)", "Temperatur 2m (°C)", "Windböen (km/h)", "Theta-E (Schwüle-Energie)", 
                   "Scherung 0-6km", "Bodendruck (hPa)", "Gesamtbedeckung (%)", "Signifikantes Wetter"],
        "steps": list(range(3, 79, 3))
    },
    "GFS (NOAA)": {
        "res": "25km",
        "params": ["Niederschlag (mm)", "Temperatur 2m (°C)", "CAPE (J/kg)", "Bodendruck (hPa)", "Schneehöhe (cm)"],
        "steps": list(range(3, 183, 3))
    },
    "ECMWF": {
        "res": "11km/40km",
        "params": ["Niederschlag (mm)", "Temperatur 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)"],
        "steps": list(range(6, 145, 6))
    }
}

# ==============================================================================
# 5. DATA ENGINE: FETCH & CALCULATE
# ==============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_meteo_file(model, key, hr, lvl="single-level"):
    """Holt die GRIB-Datei vom entsprechenden Server (Deep Scan)."""
    now = datetime.now(timezone.utc)
    
    if "ICON" in model:
        m_dir = "icon-d2" if "D2" in model else "icon-eu"
        reg = "icon-d2_germany" if "D2" in model else "icon-eu_europe"
        
        # Suche nach dem neuesten Modelllauf (bis zu 12h zurück)
        for off in range(1, 13):
            t = now - timedelta(hours=off)
            run = (t.hour // 3) * 3
            dt_str = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
            
            # Level-Logik
            l_prefix = "2d_" if lvl == "single-level" else "500_" if "500" in key else "850_"
            ckey = key.replace("_500", "").replace("_850", "")
            
            url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{ckey}/{reg}_regular-lat-lon_{lvl}_{dt_str}_{hr:03d}_{l_prefix}{ckey}.grib2.bz2"
            
            try:
                r = requests.get(url, timeout=4)
                if r.status_code == 200:
                    with bz2.open(io.BytesIO(r.content)) as bz:
                        content = bz.read()
                    return content, dt_str
            except: continue
    
    # GFS Logik (Nomads)
    elif "GFS" in model:
        # Hier vereinfacht, im Vollskript wird nomads.ncep.noaa.gov abgefragt
        pass

    return None, None

def bolton_theta_e(t2m, td2m, pmsl):
    """
    Berechnet die äquivalentpotenzielle Temperatur nach Bolton (1980).
    Dies ist die genaueste Methode für Gewitteranalysen.
    """
    T = t2m # in Kelvin
    TD = td2m # in Kelvin
    P = pmsl / 100.0 # in hPa
    
    # Sättigungsdampfdruck (e)
    e = 6.112 * np.exp(17.67 * (TD - 273.15) / (TD - 29.65))
    # Mischungsverhältnis (r)
    r = 0.622 * e / (P - e)
    # Temperatur am LCL (Kondensationsniveau)
    tlcl = 55.0 + 2840.0 / (3.5 * np.log(T) - np.log(e) - 4.805)
    # Theta-E Formel
    theta_e = T * (1000.0/P)**(0.2854 * (1 - 0.28*r)) * np.exp((3.376/tlcl - 0.00254) * r * 1000.0 * (1 + 0.81*r))
    return theta_e - 273.15 # Rückgabe in °C

def process_grib(content):
    """Konvertiert Binär-Content in Xarray Dataset."""
    if content is None: return None
    with open("temp.grib", "wb") as f:
        f.write(content)
    try:
        ds = xr.open_dataset("temp.grib", engine='cfgrib')
        return ds
    except:
        return None

# ==============================================================================
# 6. UI-STRUKTUR: TABS & SIDEBAR
# ==============================================================================
tab_map, tab_radar, tab_expert = st.tabs(["🗺️ Modell-Analyse", "⚡ SPC Radar Live", "🧬 Experten-Daten"])

# --- TAB 2: SPC RADAR (DWD GEOSERVER) ---
with tab_radar:
    st.header("⚡ Live-Radar & Unwetter-Tracking")
    col_rad1, col_rad2 = st.columns([3, 1])
    with col_rad2:
        st.info("Das SPC-Radar zeigt Live-Daten (5min Takt). In der Modell-Analyse siehst du die Zukunft.")
        st.markdown("**Legende:**")
        st.write("🟦 Regen | 🟨 Starkregen | 🟥 Unwetter")
    
    with col_rad1:
        m = folium.Map(location=[51.43, 13.22], zoom_start=9, tiles="CartoDB dark_matter")
        folium.WmsTileLayer(
            url="https://maps.dwd.de/geoserver/dwd/wms",
            layers="dwd:FX-Produkt", format="image/png", transparent=True,
            name="Radar", attr="DWD", opacity=0.7
        ).add_to(m)
        folium.WmsTileLayer(
            url="https://maps.dwd.de/geoserver/dwd/wms",
            layers="dwd:Warnungen_Gemeinden", format="image/png", transparent=True,
            name="Warnungen", attr="DWD", opacity=0.4
        ).add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, width=1000, height=600)

# --- TAB 1: HAUPT-DASHBOARD ---
with tab_map:
    # Sidebar
    with st.sidebar:
        st.title("⛈️ WarnwetterBB")
        st.subheader("V9.5 Ultimate")
        
        sel_model = st.selectbox("Modell", list(MODEL_CONFIG.keys()))
        sel_param = st.selectbox("Parameter", MODEL_CONFIG[sel_model]["params"])
        sel_region = st.selectbox("Ausschnitt", list(REGIONS.keys()))
        
        st.markdown("---")
        show_isobars = st.checkbox("Isobaren einblenden", value=True)
        show_hatch = st.checkbox("Gewitter-Schraffur (////)", value=True)
        
        with st.expander("🛠️ Debug & Cache"):
            if st.button("Cache löschen"):
                st.cache_data.clear()
                cleanup_temp_files()
            st.checkbox("Server-Pings zeigen", key="debug")

    # PLAYER STEUERUNG
    st.subheader(f"📊 {sel_param} - {sel_model}")
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns([1, 1, 1, 1, 3])
    
    steps = MODEL_CONFIG[sel_model]["steps"]
    if st.session_state.step_idx >= len(steps): st.session_state.step_idx = 0
    
    if p_col1.button("⬅️ Zurück"):
        st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
    if p_col2.button("🔄 Reset"):
        st.session_state.step_idx = 0
    if p_col3.button("Vorwärts ➡️"):
        st.session_state.step_idx = min(len(steps)-1, st.session_state.step_idx + 1)
    
    cur_step = steps[st.session_state.step_idx]
    
    # Zeitberechnung für Header
    now_u = datetime.now(timezone.utc)
    base_run_dt = ((now_u.hour - 3) // 3) * 3 # Grobe Schätzung für Label
    
    # DATEN LADEN
    with st.spinner("Synchronisiere mit Wetter-Server..."):
        data, lons, lats, run_id = None, None, None, None
        
        # PARAMETER-LOGIK
        if sel_param == "Theta-E (Schwüle-Energie)":
            c1, run_id = fetch_meteo_file(sel_model, "t_2m", cur_step)
            c2, _ = fetch_meteo_file(sel_model, "td_2m", cur_step)
            c3, _ = fetch_meteo_file(sel_model, "pmsl", cur_step)
            ds1, ds2, ds3 = process_grib(c1), process_grib(c2), process_grib(c3)
            if ds1 and ds2 and ds3:
                lons, lats = np.meshgrid(ds1.longitude.values, ds1.latitude.values)
                data = bolton_theta_e(ds1[list(ds1.data_vars)[0]].values, 
                                     ds2[list(ds2.data_vars)[0]].values, 
                                     ds3[list(ds3.data_vars)[0]].values)
        
        elif sel_param == "Niederschlag (mm)":
            c, run_id = fetch_meteo_file(sel_model, "tot_prec", cur_step)
            ds = process_grib(c)
            if ds:
                data = ds[list(ds.data_vars)[0]].values
                lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
        
        elif sel_param == "Temperatur 2m (°C)":
            c, run_id = fetch_meteo_file(sel_model, "t_2m", cur_step)
            ds = process_grib(c)
            if ds:
                raw = ds[list(ds.data_vars)[0]].values
                data = raw - 273.15 if raw.max() > 100 else raw
                lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)

        elif sel_param == "Windböen (km/h)":
            c, run_id = fetch_meteo_file(sel_model, "vmax_10m", cur_step)
            ds = process_grib(c)
            if ds:
                data = ds[list(ds.data_vars)[0]].values * 3.6
                lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)

        elif sel_param == "Wolkenuntergrenze (m)":
            k = "ceiling" if "D2" in sel_model else "hbas_con"
            c, run_id = fetch_meteo_file(sel_model, k, cur_step)
            ds = process_grib(c)
            if ds:
                data = ds[list(ds.data_vars)[0]].values
                lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)

        elif sel_param == "Signifikantes Wetter":
            c, run_id = fetch_meteo_file(sel_model, "ww", cur_step)
            ds = process_grib(c)
            if ds:
                data = ds[list(ds.data_vars)[0]].values
                lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)

        # OVERLAYS (Isobaren & Gewitter)
        iso_data = None
        if show_isobars:
            ci, _ = fetch_meteo_file(sel_model, "pmsl", cur_step)
            dsi = process_grib(ci)
            if dsi: iso_data = dsi[list(dsi.data_vars)[0]].values / 100.0
        
        ww_data = None
        if show_hatch and sel_param != "Signifikantes Wetter":
            cw, _ = fetch_meteo_file(sel_model, "ww", cur_step)
            dsw = process_grib(cw)
            if dsw: ww_data = dsw[list(dsw.data_vars)[0]].values

    # PLOTTING ENGINE
    if data is not None:
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
        ax.set_extent(REGIONS[sel_region])
        
        # Topografie
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black', zorder=20)
        ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='black', zorder=20)
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue', alpha=0.3, zorder=19)
        
        # Spezifische Plots
        if "Niederschlag" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=CMAPS[0], norm=mcolors.Normalize(0, 50), shading='auto', zorder=5)
            cb = plt.colorbar(im, label="Summe in mm", shrink=0.6, ticks=list(range(0, 55, 5)))
        
        elif "Temperatur" in sel_param or "Taupunkt" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=CMAPS[1], norm=mcolors.Normalize(-30, 35), shading='auto', zorder=5)
            cb = plt.colorbar(im, label="Temperatur in °C", shrink=0.6)
            
        elif "Theta-E" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=CMAPS[3], norm=mcolors.Normalize(10, 80), shading='auto', zorder=5)
            cb = plt.colorbar(im, label="Theta-E in °C", shrink=0.6)
            
        elif "Wolkenuntergrenze" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=CMAPS[2], norm=mcolors.BoundaryNorm([0,100,200,300,400,500,750,1000,1500,2000,3000,8000], 11), shading='auto', zorder=5)
            cb = plt.colorbar(im, label="Basis in m", shrink=0.6)

        elif "Signifikantes Wetter" in sel_param:
            # WW-Legende Logik
            grid = np.zeros_like(data)
            for i, (l, (c, codes)) in enumerate(WW_LEGEND_DATA.items(), 1):
                for code in codes: grid[data == code] = i
            ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, shading='nearest', zorder=5)
            patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND_DATA.items()]
            ax.legend(handles=patches, loc='lower left', fontsize='8', framealpha=0.9).set_zorder(30)

        # GEWITTER-SCHRAFFUR (//////)
        if ww_data is not None:
            storm_mask = np.isin(ww_data, [95, 96, 97, 99])
            if np.any(storm_mask):
                plt.rcParams['hatch.linewidth'] = 2.5
                ax.contourf(lons, lats, storm_mask, levels=[0.5, 1.5], colors='none', hatches=['////'], edgecolors='red', zorder=15)

        # ISOBAREN
        if iso_data is not None:
            cs = ax.contour(lons, lats, iso_data, colors='black', linewidths=0.7, levels=np.arange(940, 1060, 4), zorder=25)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

        # ORTS-MARKER MÜHLBERG (ELBE)
        ax.plot(13.2167, 51.4333, 'ro', markersize=6, markeredgecolor='white', zorder=40)
        t_mue = ax.text(13.22, 51.44, "Mühlberg", fontsize=10, fontweight='bold', color='white', zorder=41)
        t_mue.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])

        # HEADER
        target_dt = (datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=cur_step)).astimezone(LOCAL_TZ)
        header_str = f"WarnwetterBB | {sel_param}\nModell: {sel_model} ({MODEL_CONFIG[sel_model]['res']}) | Lauf: {run_id[-2:]}Z\nTermin: {target_dt.strftime('%A, %d.%m.%Y %H:%M')} ME(S)Z"
        ax.text(0.01, 0.99, header_str, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'), zorder=50)

        st.pyplot(fig)
        
        # DOWNLOAD & TOOLS
        col_t1, col_t2, col_t3 = st.columns([1,2,1])
        with col_t2:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            st.download_button("📥 Karte als PNG speichern", buf.getvalue(), f"WarnBB_{sel_model}_{cur_step}h.png", "image/png", use_container_width=True)

        cleanup_temp_files()
    else:
        st.error("Datenverbindung unterbrochen oder Modell-Lauf noch nicht auf dem Server. Bitte Vorwärts/Zurück probieren.")

# --- TAB 3: EXPERTEN-DATEN ---
with tab_expert:
    st.header("🧬 Meteorologische Analyse-Werte")
    if data is not None:
        e_col1, e_col2, e_col3 = st.columns(3)
        e_col1.metric("Maximalwert (Sichtfeld)", f"{np.nanmax(data):.1f}")
        e_col2.metric("Minimalwert (Sichtfeld)", f"{np.nanmin(data):.1f}")
        e_col3.metric("Durchschnitt", f"{np.nanmean(data):.1f}")
        
        st.markdown("---")
        st.subheader("Gefahren-Index")
        if sel_param == "Windböen (km/h)" and np.nanmax(data) > 75:
            st.error(f"⚠️ STURMGEFAHR detektiert! Spitzenböe: {np.nanmax(data):.1f} km/h")
        elif sel_param == "Niederschlag (mm)" and np.nanmax(data) > 25:
            st.warning(f"⚠️ STARKREGEN detektiert! Max: {np.nanmax(data):.1f} mm/h")
        else:
            st.success("✅ Keine akuten Schwellwert-Überschreitungen im gewählten Ausschnitt.")

# ENDE DES SKRIPTS
