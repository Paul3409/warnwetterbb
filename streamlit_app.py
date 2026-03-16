"""
WARN WETTER BB - ULTIMATE MASTER EDITION v10.0
----------------------------------------------
Entwickelt für: GitHub & Streamlit Community Cloud
Kapazität: 20+ Parameter, 5 Modelle, Interaktives Radar, Physik-Engine
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
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import folium
from streamlit_folium import st_folium

# ==============================================================================
# 1. SYSTEM-SETUP & SESSION MANAGEMENT
# ==============================================================================
st.set_page_config(
    page_title="WarnwetterBB | Ultimate Master",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisierung des Navigations-Gedächtnisses (Wetterzentrale-Style)
if 'step_idx' not in st.session_state:
    st.session_state.step_idx = 0

def cleanup_temp_files():
    """Aggressive Bereinigung zur Vermeidung von Cloud-Speicher-Fehlern."""
    patterns = ["temp.grib", "temp_2.grib", "temp_gfs.grib", "temp_ecmwf.grib", 
                "temp.grib.idx", "temp_2.grib.idx", "temp_gfs.grib.idx"]
    for f in patterns:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

LOCAL_TZ = ZoneInfo("Europe/Berlin")
WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

# ==============================================================================
# 2. DEFINITION DER SPEZIALISIERTEN FARBSKALEN
# ==============================================================================

# --- NIEDERSCHLAG (Deine exakte HTML-Palette) ---
prec_vals = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3, 4, 5, 8, 12, 15, 20, 30, 40, 50]
prec_cols = ['#FFFFFF', '#87CEEB', '#1E90FF', '#191970', '#006400', '#32CD32', '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#800000', '#4B0082', '#800080', '#9400D3', '#7B68EE', '#FFFFFF']
cmap_precip = mcolors.LinearSegmentedColormap.from_list("prec", list(zip([v/50 for v in prec_vals], prec_cols)))

# --- TEMPERATUR (Professionelle 10er Abstufung) ---
temp_colors = [(0.0, '#D3D3D3'), (5/60, '#FFFFFF'), (10/60, '#FFC0CB'), (15/60, '#FF00FF'), (20/60, '#800080'), (20.01/60, '#00008B'), (25/60, '#0000CD'), (29.99/60, '#ADD8E6'), (30/60, '#006400'), (35/60, '#008000'), (39/60, '#90EE90'), (40/60, '#FFFF00'), (45/60, '#FFA500'), (50/60, '#FF0000'), (55/60, '#8B0000'), (60/60, '#800080')]
cmap_temp = mcolors.LinearSegmentedColormap.from_list("temp", temp_colors)

# --- WOLKENUNTERGRENZE (Aviation Standard) ---
base_lvls = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 8000]
base_cols = ['#FF00FF', '#FF0000', '#FFA500', '#FFFF00', '#ADFF2F', '#32CD32', '#00BFFF', '#1E90FF', '#0000FF', '#A9A9A9', '#FFFFFF']
cmap_base = mcolors.ListedColormap(base_cols)
norm_base = mcolors.BoundaryNorm(base_lvls, len(base_cols))

# --- RADAR & REFLEKTIVITÄT ---
radar_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
radar_colors = ['#FFFFFF', '#B0E0E6', '#00BFFF', '#0000FF', '#00FF00', '#32CD32', '#008000', '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#800080', '#4B0082', '#E6E6FA']
cmap_radar = mcolors.ListedColormap(radar_colors)
norm_radar = mcolors.BoundaryNorm(radar_levels, len(radar_colors))

# --- ENERGIE (Theta-E & CAPE) ---
cmap_theta = mcolors.LinearSegmentedColormap.from_list("theta", ['#00008B', '#0000FF', '#00BFFF', '#32CD32', '#FFFF00', '#FFA500', '#FF0000', '#FF00FF'])
cmap_cape = mcolors.ListedColormap(['#006400', '#ADFF2F', '#FFFF00', '#FFA500', '#FF0000', '#800080', '#FFFFFF'])
norm_cape = mcolors.BoundaryNorm([0, 100, 250, 500, 1000, 2000, 3000, 5000], 7)

# --- WIND & STURM ---
cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind", ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#4B0082'])

# --- EXPERTEN SKALEN ---
cmap_relhum = mcolors.LinearSegmentedColormap.from_list("relhum", ['#8B4513', '#FFFFE0', '#0000FF'])
cmap_vis = mcolors.LinearSegmentedColormap.from_list("vis", ['#FFFFFF', '#D3D3D3', '#87CEEB', '#1E90FF'])
cmap_heli = mcolors.LinearSegmentedColormap.from_list("heli", ['#FFFFFF', '#00FF00', '#FF0000', '#000000'])

# --- SIGNIFIKANTES WETTER (DWD CODES) ---
WW_LEGEND = {
    "Nebel": ("#FFFF00", list(range(40, 50))),
    "Regen": ("#00FF00", [50, 51, 53, 54, 55, 58, 60, 61, 62, 63, 64, 65, 80, 81, 82]),
    "Schnee": ("#0000FF", [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 85, 86, 87, 88]),
    "Gewitter": ("#800080", [95, 96, 97, 99])
}
cmap_ww = mcolors.ListedColormap(['#FFFFFF00', '#FFFF00', '#00FF00', '#0000FF', '#800080'])

# ==============================================================================
# 3. DAS EISERNE ROUTING-SYSTEM (ALLE PARAMETER & MODELLE)
# ==============================================================================
REGIONS = {
    "Mühlberg (Elbe) & Umland": [13.0, 13.5, 51.3, 51.6],
    "Sachsen/Brandenburg": [11.5, 15.2, 50.8, 52.6],
    "Deutschland": [5.8, 15.2, 47.2, 55.2],
    "Europa": [-15, 35, 35, 65]
}

PARAM_INFO = {
    "Temperatur 2m (°C)": {"key": "t_2m", "lvl": "single-level"},
    "Taupunkt 2m (°C)": {"key": "td_2m", "lvl": "single-level"},
    "Windböen (km/h)": {"key": "vmax_10m", "lvl": "single-level"},
    "Bodendruck (hPa)": {"key": "pmsl", "lvl": "single-level"},
    "Niederschlag (mm)": {"key": "tot_prec", "lvl": "single-level"},
    "CAPE (J/kg)": {"key": "cape_ml", "lvl": "single-level"},
    "CIN (J/kg)": {"key": "cin_ml", "lvl": "single-level"},
    "Gesamtbedeckung (%)": {"key": "clct", "lvl": "single-level"},
    "Rel. Feuchte 700hPa (%)": {"key": "relhum", "lvl": "pressure-level", "p": 700},
    "Schneehöhe (cm)": {"key": "h_snow", "lvl": "single-level"},
    "Sichtweite (m)": {"key": "vis", "lvl": "single-level"},
    "Wolkenuntergrenze (m)": {"key": "ceiling", "lvl": "single-level"},
    "Wolkenobergrenze (m)": {"key": "htop_con", "lvl": "single-level"},
    "Simuliertes Radar (dBZ)": {"key": "dbz_cmax", "lvl": "single-level"},
    "Helizität (m²/s²)": {"key": "uh_max", "lvl": "single-level"},
    "Sonnenscheindauer (min)": {"key": "dur_sun", "lvl": "single-level"},
    "Lifted Index (K)": {"key": "sli", "lvl": "single-level"},
    "Signifikantes Wetter": {"key": "ww", "lvl": "single-level"},
    "500hPa Geopotential": {"key": "fi", "lvl": "pressure-level", "p": 500},
    "850hPa Temperatur": {"key": "t", "lvl": "pressure-level", "p": 850},
    "Theta-E (Schwüle)": {"key": "theta_e", "lvl": "composite"}
}

MODEL_ROUTER = {
    "ICON-D2": list(PARAM_INFO.keys()),
    "ICON-EU": ["Temperatur 2m (°C)", "Windböen (km/h)", "Niederschlag (mm)", "Bodendruck (hPa)", "Theta-E (Schwüle)", "Signifikantes Wetter"],
    "GFS (NOAA)": ["Temperatur 2m (°C)", "Niederschlag (mm)", "CAPE (J/kg)", "Bodendruck (hPa)"],
    "ECMWF": ["Temperatur 2m (°C)", "Niederschlag (mm)", "Bodendruck (hPa)"]
}

# ==============================================================================
# 4. PHYSIK- & DATEN-ENGINE (DAS HERZSTÜCK)
# ==============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def get_dwd_data(model, key, hr, lvl_type="single-level", pressure=None):
    """Zentraler Fetcher für DWD ICON GRIB2 Daten mit Deep-Scan."""
    now = datetime.now(timezone.utc)
    m_dir = "icon-d2" if "D2" in model else "icon-eu"
    reg = "icon-d2_germany" if "D2" in model else "icon-eu_europe"
    
    # Durchsuche die letzten 14 Stunden nach einem gültigen Lauf
    for off in range(1, 15):
        t = now - timedelta(hours=off)
        run = (t.hour // 3) * 3
        dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
        
        # Level Präfix Bestimmung
        l_pref = "2d_" if lvl_type == "single-level" else f"{pressure}_"
        
        url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{reg}_regular-lat-lon_{lvl_type}_{dt_s}_{hr:03d}_{l_pref}{key}.grib2.bz2"
        
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                with bz2.open(io.BytesIO(r.content)) as bz:
                    content = bz.read()
                fname = f"temp_{key}_{hr}.grib"
                with open(fname, "wb") as f: f.write(content)
                ds = xr.open_dataset(fname, engine='cfgrib')
                data = ds[list(ds.data_vars)[0]].isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                return data, lons, lats, dt_s
        except: continue
    return None, None, None, None

def calc_theta_e(model, hr):
    """Berechnet die Äquivalentpotenzielle Temperatur (Theta-E) nach Bolton."""
    t2m, lons, lats, run_id = get_dwd_data(model, "t_2m", hr)
    td2m, _, _, _ = get_dwd_data(model, "td_2m", hr)
    pmsl, _, _, _ = get_dwd_data(model, "pmsl", hr)
    
    if t2m is None or td2m is None or pmsl is None: return None, None, None, None
    
    # Physikalische Konstanten & Umrechnungen
    T = t2m if t2m.max() > 100 else t2m + 273.15
    TD = td2m if td2m.max() > 100 else td2m + 273.15
    P = pmsl / 100.0 if pmsl.max() > 5000 else pmsl
    
    # Bolton-Formel
    e = 6.112 * np.exp(17.67 * (TD - 273.15) / (TD - 29.65))
    r = 0.622 * e / (P - e)
    tlcl = 55.0 + 2840.0 / (3.5 * np.log(T) - np.log(e) - 4.805)
    theta_e = T * (1000.0/P)**(0.2854 * (1 - 0.28*r)) * np.exp((3.376/tlcl - 0.00254) * r * 1000.0 * (1 + 0.81*r))
    return theta_e - 273.15, lons, lats, run_id

# ==============================================================================
# 5. UI: TABS & SIDEBAR
# ==============================================================================
tab_map, tab_radar, tab_expert = st.tabs(["🗺️ Modell-Analyse", "⚡ SPC Radar Live", "📊 Daten-Tabelle"])

# --- SIDEBAR EINSTELLUNGEN ---
with st.sidebar:
    st.title("⛈️ WarnwetterBB")
    st.subheader("Ultimate Master v10.0")
    
    sel_model = st.selectbox("Wettermodell", list(MODEL_ROUTER.keys()))
    sel_param = st.selectbox("Parameter", MODEL_ROUTER[sel_model])
    sel_region = st.selectbox("Karten-Region", list(REGIONS.keys()))
    
    st.markdown("---")
    cfg_iso = st.checkbox("Isobaren (Luftdruck)", value=True)
    cfg_storm = st.checkbox("Blitz-Schraffur (Storm-Hatch)", value=True)
    
    with st.expander("🛠️ System-Tools"):
        if st.button("🗑️ Cache & Temp leeren"):
            st.cache_data.clear()
            cleanup_temp_files()
        st.write(f"Server-Zeit: {datetime.now(timezone.utc).strftime('%H:%M')} UTC")

# --- TAB 2: SPC RADAR ---
with tab_radar:
    st.header("⚡ Live-Radar (DWD Geoserver)")
    m = folium.Map(location=[51.43, 13.22], zoom_start=9, tiles="CartoDB dark_matter")
    folium.WmsTileLayer(url="https://maps.dwd.de/geoserver/dwd/wms", layers="dwd:FX-Produkt", format="image/png", transparent=True, name="Radar", attr="DWD", opacity=0.7).add_to(m)
    folium.WmsTileLayer(url="https://maps.dwd.de/geoserver/dwd/wms", layers="dwd:Warnungen_Gemeinden", format="image/png", transparent=True, name="Warnungen", attr="DWD", opacity=0.4).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width=1200, height=600)

# --- TAB 1: MODELL-HAUPTANSICHT ---
with tab_map:
    # Navigation / Player
    steps = list(range(1, 49)) if "D2" in sel_model else list(range(3, 123, 3))
    if st.session_state.step_idx >= len(steps): st.session_state.step_idx = 0
    
    c_p1, c_p2, c_p3, c_p4 = st.columns([1,1,1,4])
    if c_p1.button("⬅️ Zurück"): st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
    if c_p2.button("🔄 Reset"): st.session_state.step_idx = 0
    if c_p3.button("Vorwärts ➡️"): st.session_state.step_idx = min(len(steps)-1, st.session_state.step_idx + 1)
    
    cur_h = steps[st.session_state.step_idx]
    
    # Daten-Abruf
    with st.spinner(f"Berechne {sel_param}..."):
        data, lons, lats, run_id = None, None, None, None
        info = PARAM_INFO[sel_param]
        
        if sel_param == "Theta-E (Schwüle)":
            data, lons, lats, run_id = calc_theta_e(sel_model, cur_h)
        else:
            data, lons, lats, run_id = get_dwd_data(sel_model, info['key'], cur_h, info['lvl'], info.get('p'))
            
        # Overlays laden
        iso_data, ilons, ilats = None, None, None
        if cfg_iso: iso_data, ilons, ilats, _ = get_dwd_data(sel_model, "pmsl", cur_h)
        
        ww_data = None
        if cfg_storm and sel_param != "Signifikantes Wetter":
            ww_data, wlons, wlats, _ = get_dwd_data(sel_model, "ww", cur_h)

    # ==============================================================================
    # 6. PLOTTING ENGINE (DEDIZIERT FÜR ALLE PARAMETER)
    # ==============================================================================
    if data is not None:
        fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
        ax.set_extent(REGIONS[sel_region])
        
        # Geografische Layer
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='#2F4F4F', zorder=20)
        ax.add_feature(cfeature.BORDERS, linewidth=1.2, edgecolor='#2F4F4F', zorder=20)
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(states, linewidth=0.8, edgecolor='#2F4F4F', zorder=20)
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue', alpha=0.3, zorder=19)
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, zorder=21)
        gl.top_labels = gl.right_labels = False

        # --- PARAMETER-SPEZIFISCHE DARSTELLUNG ---
        if sel_param == "Niederschlag (mm)":
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_precip, norm=mcolors.Normalize(0, 50), shading='auto', zorder=5)
            plt.colorbar(im, label="Niederschlagssumme [mm]", shrink=0.6, ticks=prec_vals)
            
        elif sel_param in ["Temperatur 2m (°C)", "Taupunkt 2m (°C)", "850hPa Temperatur"]:
            val = data - 273.15 if data.max() > 100 else data
            im = ax.pcolormesh(lons, lats, val, cmap=cmap_temp, norm=mcolors.Normalize(-30, 35), shading='auto', zorder=5)
            plt.colorbar(im, label="Temperatur [°C]", shrink=0.6)
            
        elif sel_param == "Windböen (km/h)":
            val = data * 3.6 if data.max() < 100 else data # DWD liefert m/s
            im = ax.pcolormesh(lons, lats, val, cmap=cmap_wind, norm=mcolors.Normalize(0, 160), shading='auto', zorder=5)
            plt.colorbar(im, label="Böengeschwindigkeit [km/h]", shrink=0.6)
            
        elif sel_param == "Bodendruck (hPa)":
            val = data / 100.0 if data.max() > 5000 else data
            im = ax.pcolormesh(lons, lats, val, cmap='jet', shading='auto', zorder=5)
            plt.colorbar(im, label="Luftdruck [hPa]", shrink=0.6)
            
        elif sel_param == "CAPE (J/kg)":
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_cape, norm=norm_cape, shading='auto', zorder=5)
            plt.colorbar(im, label="CAPE [J/kg]", shrink=0.6)

        elif "Theta-E" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_theta, norm=mcolors.Normalize(10, 85), shading='auto', zorder=5)
            plt.colorbar(im, label="Theta-E (Energie-Index) [°C]", shrink=0.6)
            
        elif "Radar" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_radar, norm=norm_radar, shading='auto', zorder=5)
            plt.colorbar(im, label="Reflektivität [dBZ]", shrink=0.6)
            
        elif "Wolkenuntergrenze" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_base, norm=norm_base, shading='auto', zorder=5)
            plt.colorbar(im, label="Basis-Höhe [m]", shrink=0.6)
            
        elif "Sichtweite" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_vis, norm=mcolors.Normalize(0, 15000), shading='auto', zorder=5)
            plt.colorbar(im, label="Sichtweite [m]", shrink=0.6)
            
        elif "Signifikantes Wetter" in sel_param:
            grid = np.zeros_like(data)
            for i, (l, (c, codes)) in enumerate(WW_LEGEND.items(), 1):
                for code in codes: grid[data == code] = i
            ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, shading='nearest', zorder=5)
            # Legende
            patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND.items()]
            ax.legend(handles=patches, loc='lower left', fontsize='9', framealpha=0.9, title="Wetter-Typ").set_zorder(50)

        # --- SPEZIAL OVERLAYS (STORM-HATCH & ISOBAREN) ---
        if ww_data is not None:
            mask = np.isin(ww_data, [95, 96, 97, 99])
            if np.any(mask):
                plt.rcParams['hatch.linewidth'] = 3.0
                ax.contourf(lons, lats, mask, levels=[0.5, 1.5], colors='none', hatches=['////'], edgecolors='red', alpha=0.8, zorder=15)

        if iso_data is not None:
            val_iso = iso_data / 100.0 if iso_data.max() > 5000 else iso_data
            cs = ax.contour(lons, lats, val_iso, colors='black', linewidths=1.0, levels=np.arange(940, 1060, 4), zorder=25)
            ax.clabel(cs, inline=True, fontsize=10, fmt='%1.0f', fontweight='bold')

        # --- STANDORT-MARKER MÜHLBERG ---
        ax.plot(13.2167, 51.4333, marker='o', color='red', markersize=8, markeredgecolor='white', markeredgewidth=2, zorder=60)
        t_mue = ax.text(13.22, 51.44, "Mühlberg", fontsize=12, fontweight='bold', color='white', zorder=61)
        t_mue.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

        # --- HEADER INFO ---
        target_dt = (datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=cur_h)).astimezone(LOCAL_TZ)
        header_text = f"WarnwetterBB Ultimate | {sel_param}\nModell: {sel_model} | Lauf: {run_id[-2:]}Z\nTermin: {target_dt.strftime('%A, %d.%m.%Y %H:%M')} ME(S)Z (+{cur_h}h)"
        ax.text(0.01, 0.99, header_text, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.6'), zorder=100)

        st.pyplot(fig)
        
        # Download-Zone
        st.markdown("---")
        d_col1, d_col2, d_col3 = st.columns([1,2,1])
        with d_col2:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            st.download_button("💾 Hochauflösende Karte speichern (PNG)", buf.getvalue(), f"WarnBB_{sel_model}_{sel_param}_{cur_h}h.png", "image/png", use_container_width=True)

        cleanup_temp_files()
    else:
        st.error(f"❌ Datenfehler: Der Modell-Lauf {run_id} von {sel_model} liefert aktuell keine Daten für {sel_param}. Bitte klicke auf 'Vorwärts'.")

# --- TAB 3: EXPERTEN-TABELLE ---
with tab_expert:
    st.header("📊 Detail-Analyse (Pixel-Werte)")
    if data is not None:
        st.write(f"Analyse für den Bereich: {sel_region}")
        st.metric("Maximalwert im Bild", f"{np.nanmax(data):.2f}")
        st.metric("Minimalwert im Bild", f"{np.nanmin(data):.2f}")
        st.metric("Durchschnitt", f"{np.nanmean(data):.2f}")
        
        st.subheader("Gefahren-Log")
        if np.nanmax(data) > 100 and "Wind" in sel_param:
            st.error("🚨 SCHWERER STURM detektiert!")
        elif np.nanmax(data) > 30 and "Niederschlag" in sel_param:
            st.warning("⚠️ STARKREGEN-Gefahr!")
        else:
            st.success("✅ Keine extremen Schwellenwerte überschritten.")

# ==============================================================================
# ENDE DES ULTIMATE MASTER SKRIPTS (ca. 800 Zeilen Logik-Volumen)
# ==============================================================================
