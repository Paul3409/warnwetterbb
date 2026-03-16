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

# ==============================================================================
# 1. SETUP, KONFIGURATION & AGGRESSIVES CACHE-MANAGEMENT
# ==============================================================================
st.set_page_config(page_title="WarnwetterBB | Pro-Zentrale", layout="wide", initial_sidebar_state="expanded")

def cleanup_temp_files():
    temp_files = ["temp.grib", "temp_gfs.grib", "temp_ecmwf.grib", "temp.grib.idx", "temp_gfs.grib.idx"]
    for file in temp_files:
        if os.path.exists(file):
            try: os.remove(file)
            except: pass

LOCAL_TZ = ZoneInfo("Europe/Berlin")
WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

# ==============================================================================
# 2. MASTER-FARBSKALEN
# ==============================================================================
temp_colors = [
    (0.0, '#D3D3D3'), (5/60, '#FFFFFF'), (10/60, '#FFC0CB'), (15/60, '#FF00FF'),
    (20/60, '#800080'), (20.01/60, '#00008B'), (25/60, '#0000CD'), (29.99/60, '#ADD8E6'),
    (30/60, '#006400'), (35/60, '#008000'), (39/60, '#90EE90'), (39.99/60, '#90EE90'),
    (40/60, '#FFFF00'), (45/60, '#FFA500'), (50/60, '#FF0000'), (55/60, '#8B0000'), (60/60, '#800080')
]
cmap_temp = mcolors.LinearSegmentedColormap.from_list("custom_temp", temp_colors)

# DEINE EXAKTE HTML-FARBPALETTE
precip_values = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3, 4, 5, 8, 12, 15, 20, 30, 40, 50]
precip_colors = [
    '#FFFFFF', '#87CEEB', '#1E90FF', '#191970', '#006400', '#32CD32', '#FFFF00', 
    '#FFA500', '#FF0000', '#8B0000', '#800000', '#4B0082', '#800080', '#9400D3', 
    '#7B68EE', '#FFFFFF'  
]
vmax_precip = 50.0
precip_anchors = [v / vmax_precip for v in precip_values]
cmap_precip = mcolors.LinearSegmentedColormap.from_list("custom_precip", list(zip(precip_anchors, precip_colors)))
norm_precip = mcolors.Normalize(vmin=0, vmax=vmax_precip)

# AVIATION FLUGSICHERHEITS-SKALA
base_levels = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 8000]
base_colors = [
    '#FF00FF', '#FF0000', '#FFA500', '#FFFF00', '#ADFF2F', '#32CD32', '#00BFFF', 
    '#1E90FF', '#0000FF', '#A9A9A9', '#FFFFFF'
]
cmap_base = mcolors.ListedColormap(base_colors)
norm_base = mcolors.BoundaryNorm(base_levels, cmap_base.N)

cape_levels = [0, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]
cape_colors = [
    '#006400', '#2E8B57', '#ADFF2F', '#FFFF00', '#FFB347', '#FFA500', 
    '#FF4500', '#FF0000', '#8B0000', '#800080', '#FF00FF', '#FFFFFF', '#808080', '#404040'
]
cmap_cape = mcolors.ListedColormap(cape_colors)
norm_cape = mcolors.BoundaryNorm(cape_levels, cmap_cape.N)

radar_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
radar_colors = [
    '#FFFFFF', '#B0E0E6', '#00BFFF', '#0000FF', '#00FF00', '#32CD32', '#008000', 
    '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#800080', '#4B0082', '#E6E6FA'
]
cmap_radar = mcolors.ListedColormap(radar_colors)
norm_radar = mcolors.BoundaryNorm(radar_levels, cmap_radar.N)

cmap_cin = mcolors.LinearSegmentedColormap.from_list("cin", ['#FFFFFF', '#ADD8E6', '#0000FF', '#00008B', '#000000'], N=256)
cmap_clouds = mcolors.LinearSegmentedColormap.from_list("clouds", ['#1E90FF', '#87CEEB', '#D3D3D3', '#FFFFFF'], N=256)
cmap_relhum = mcolors.LinearSegmentedColormap.from_list("relhum", ['#8B4513', '#F4A460', '#FFFFE0', '#90EE90', '#008000', '#0000FF'], N=256)
cmap_snow = mcolors.LinearSegmentedColormap.from_list("snow", ['#CCFFCC', '#FFFFFF', '#ADD8E6', '#0000FF', '#800080'], N=256)
cmap_vis = mcolors.LinearSegmentedColormap.from_list("vis", ['#FFFFFF', '#D3D3D3', '#87CEEB', '#1E90FF'], N=256)
W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']
cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind", W_COLORS, N=256)
cmap_heli = mcolors.LinearSegmentedColormap.from_list("heli", ['#FFFFFF', '#00FF00', '#FFFF00', '#FF0000', '#800080', '#000000'], N=256)
cmap_lifted = mcolors.LinearSegmentedColormap.from_list("lifted", ['#FF00FF', '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#0000FF'], N=256)
cmap_sun = mcolors.LinearSegmentedColormap.from_list("sun", ['#808080', '#FFD700', '#FFA500', '#FF8C00'], N=256)

WW_LEGEND_DATA = {
    "Nebel": ("#FFFF00", list(range(40, 50))),
    "Regen leicht": ("#00FF00", [50, 51, 58, 60, 80]),
    "Regen mäßig": ("#228B22", [53, 61, 62, 81]),
    "Regen stark": ("#006400", [54, 55, 63, 64, 65, 82]),
    "gefr. Regen leicht": ("#FF7F7F", [56, 66]),
    "gefr. Regen mäßig/stark": ("#FF0000", [57, 67]),
    "Schneeregen leicht": ("#FFB347", [68, 83]),
    "Schneeregen mäßig/stark": ("#FFA500", [69, 84]),
    "Schnee leicht": ("#87CEEB", [70, 71, 85]),
    "Schnee mäßig": ("#0000FF", [72, 73, 86]),
    "Schnee stark": ("#00008B", [74, 75, 76, 77, 78, 79, 87, 88]),
    "Gewitter leicht": ("#FF00FF", [95]),
    "Gewitter mäßig/stark": ("#800080", [96, 97, 99])
}
cmap_ww = mcolors.ListedColormap(['#FFFFFF00'] + [c for l, (c, codes) in WW_LEGEND_DATA.items()])

# ==============================================================================
# 3. DAS EISERNE ROUTING-SYSTEM
# ==============================================================================
MODEL_ROUTER = {
    "ICON-D2": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum"],
        "params": ["Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
                   "500 hPa Geopot. Höhe", "850 hPa Temp.", "Niederschlag (mm)", "CAPE (J/kg)", 
                   "CIN (J/kg)", "Gesamtbedeckung (%)", "Rel. Feuchte 700 hPa (%)", "Schneehöhe (cm)",
                   "Signifikantes Wetter", "Sichtweite (m)", "Wolkenuntergrenze (m)", "Wolkenobergrenze (m)", 
                   "Spezifische Feuchte (g/kg)", "Simuliertes Radar (dBZ)", "Helizität / SRH (m²/s²)", 
                   "Sonnenscheindauer (Min)"]
    },
    "ICON-EU": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"],
        "params": ["Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
                   "500 hPa Geopot. Höhe", "850 hPa Temp.", "Niederschlag (mm)", "CAPE (J/kg)", 
                   "CIN (J/kg)", "Gesamtbedeckung (%)", "Rel. Feuchte 700 hPa (%)", "Schneehöhe (cm)",
                   "Signifikantes Wetter", "Sichtweite (m)", "Wolkenuntergrenze (m)", "Wolkenobergrenze (m)"]
    },
    "ICON (Global)": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"],
        "params": ["Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
                   "500 hPa Geopot. Höhe", "850 hPa Temp.", "Niederschlag (mm)", "CAPE (J/kg)", 
                   "CIN (J/kg)", "Gesamtbedeckung (%)", "Rel. Feuchte 700 hPa (%)", "Schneehöhe (cm)"]
    },
    "GFS (NOAA)": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"],
        "params": ["Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
                   "500 hPa Geopot. Höhe", "850 hPa Temp.", "Niederschlag (mm)", "CAPE (J/kg)", 
                   "CIN (J/kg)", "Gesamtbedeckung (%)", "Rel. Feuchte 700 hPa (%)", "Schneehöhe (cm)",
                   "0-Grad-Grenze (m)", "Lifted Index (K)", "Sichtweite (m)", "Wolkenuntergrenze (m)"]
    },
    "ECMWF": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"],
        "params": ["Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
                   "500 hPa Geopot. Höhe", "850 hPa Temp.", "Niederschlag (mm)", "Gesamtbedeckung (%)"] 
    },
    "ECMWF-AIFS (KI-Modell)": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"],
        "params": ["Temperatur 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
                   "500 hPa Geopot. Höhe", "850 hPa Temp.", "Niederschlag (mm)"]
    }
}

def estimate_latest_run(model, now_utc):
    if "D2" in model or "EU" in model:
        run_h = ((now_utc.hour - 3) // 3) * 3
        if run_h < 0: return (now_utc - timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)
        return now_utc.replace(hour=run_h, minute=0, second=0, microsecond=0)
    elif "GFS" in model or "Global" in model:
        run_h = ((now_utc.hour - 6) // 6) * 6
        if run_h < 0: return (now_utc - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        return now_utc.replace(hour=run_h, minute=0, second=0, microsecond=0)
    elif "ECMWF" in model:
        run_h = ((now_utc.hour - 10) // 12) * 12
        if run_h < 0: return (now_utc - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
        return now_utc.replace(hour=run_h, minute=0, second=0, microsecond=0)

# ==============================================================================
# 4. DYNAMISCHE SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("🛰️ Modell-Zentrale")
    
    with st.expander("🌍 1. Modell wählen", expanded=True):
        sel_model = st.radio("Wettermodell", list(MODEL_ROUTER.keys()), label_visibility="collapsed")
    
    with st.expander("🗺️ 2. Karten-Ausschnitt", expanded=False):
        valid_regions = MODEL_ROUTER[sel_model]["regions"]
        sel_region = st.radio("Region", valid_regions, label_visibility="collapsed")
    
    with st.expander("🌪️ 3. Parameter wählen", expanded=True):
        valid_params = MODEL_ROUTER[sel_model]["params"]
        sel_param = st.radio("Parameter", valid_params, label_visibility="collapsed")
    
    with st.expander("⏱️ 4. Vorhersage-Stunde (MEZ/MESZ)", expanded=True):
        if "EU" in sel_model: hours = list(range(1, 79))
        elif "GFS" in sel_model or "Global" in sel_model: hours = list(range(3, 123, 3))
        elif "ECMWF" in sel_model: hours = list(range(3, 147, 3))
        else: hours = list(range(1, 49))
        
        now_utc = datetime.now(timezone.utc)
        base_run = estimate_latest_run(sel_model, now_utc)
        
        hour_labels = []
        for h in hours:
            target_dt_utc = base_run + timedelta(hours=h)
            target_dt_loc = target_dt_utc.astimezone(LOCAL_TZ)
            tz_str = "MESZ" if target_dt_loc.dst() else "MEZ"
            wt = WOCHENTAGE[target_dt_loc.weekday()]
            time_str = f"+{h}h  ({wt}, {target_dt_loc.strftime('%d.%m. %H:%M')} {tz_str})"
            hour_labels.append(time_str)
            
        sel_hour_str = st.radio("Zeit", hour_labels, label_visibility="collapsed")
        sel_hour = int(sel_hour_str.split("h")[0].replace("+", ""))
    
    st.markdown("---")
    show_isobars = st.checkbox("Isobaren (Luftdruck) einblenden", value=True)
    
    # NEU: Der Gewitter-Schalter!
    show_storms = st.checkbox("⚡ Gewitter-Risiko rot schraffieren", value=True, help="Zieht rote, diagonale Linien über Gebiete, in denen das Modell Gewitter berechnet.")
    
    st.markdown("---")
    generate = st.button("🚀 Profi-Karte generieren", use_container_width=True)
    
    with st.expander("🛠️ Entwickler-Konsole"):
        debug_mode = st.checkbox("URL-Ping aktivieren (Zeigt Live-Serveranfragen)")

# ==============================================================================
# 5. DATA FETCH ENGINE
# ==============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_meteo_data(model, param, hr, debug=False):
    dyn_cloud_base = "ceiling" if "D2" in model else "hbas_con"
    
    p_map = {
        "Temperatur 2m (°C)": "t_2m", "Taupunkt 2m (°C)": "td_2m", "Windböen (km/h)": "vmax_10m", 
        "Bodendruck (hPa)": "sp", "500 hPa Geopot. Höhe": "fi", "850 hPa Temp.": "t", 
        "Signifikantes Wetter": "ww", "Isobaren": "pmsl", "CAPE (J/kg)": "cape_ml", 
        "CIN (J/kg)": "cin_ml", "Niederschlag (mm)": "tot_prec", "Simuliertes Radar (dBZ)": "dbz_cmax",
        "0-Grad-Grenze (m)": "hgt_0c", "Gesamtbedeckung (%)": "clct", "Rel. Feuchte 700 hPa (%)": "relhum", 
        "Schneehöhe (cm)": "h_snow", "Sichtweite (m)": "vis", "Wolkenuntergrenze (m)": dyn_cloud_base,
        "Wolkenobergrenze (m)": "htop_con", "Spezifische Feuchte (g/kg)": "qv",
        "Helizität / SRH (m²/s²)": "uh_max", "Sonnenscheindauer (Min)": "dur_sun", "Lifted Index (K)": "sli"
    }
    key = p_map.get(param, "t_2m")
    now = datetime.now(timezone.utc)
    headers = {'User-Agent': 'Mozilla/5.0'}
    debug_logs = []

    if "ICON" in model:
        is_global = "Global" in model
        if is_global: m_dir, reg_str = "icon", "icon_global"
        elif "D2" in model: m_dir, reg_str = "icon-d2", "icon-d2_germany"
        else: m_dir, reg_str = "icon-eu", "icon-eu_europe"
        
        for off in range(1, 18):
            t = now - timedelta(hours=off)
            if is_global: run = (t.hour // 6) * 6
            else: run = (t.hour // 3) * 3
            
            dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
            
            l_type = "pressure-level" if key in ["fi", "t", "relhum", "qv"] else "single-level"
            if l_type == "pressure-level": lvl_str = f"{'500' if '500' in param else '700' if '700' in param else '850'}_"
            else: lvl_str = "2d_"
            
            url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{reg_str}_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{lvl_str}{key}.grib2.bz2"
            debug_logs.append(url)
            
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    with bz2.open(io.BytesIO(r.content)) as f_bz2:
                        with open("temp.grib", "wb") as f_out: f_out.write(f_bz2.read())
                    ds = xr.open_dataset("temp.grib", engine='cfgrib')
                    ds_var = ds[list(ds.data_vars)[0]]
                    if 'isobaricInhPa' in ds_var.dims:
                        target_p = 500 if "500" in param else 700 if "700" in param else 850
                        ds_var = ds_var.sel(isobaricInhPa=target_p)
                    data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                    lons, lats = ds.longitude.values, ds.latitude.values
                    if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                    return data, lons, lats, dt_s, debug_logs
            except Exception: continue

    elif "GFS" in model:
        gfs_map = {
            "t_2m": "&var_TMP=on&lev_2_m_above_ground=on", "td_2m": "&var_DPT=on&lev_2_m_above_ground=on",
            "vmax_10m": "&var_GUST=on&lev_surface=on", "fi": "&var_HGT=on&lev_500_mb=on",
            "t": "&var_TMP=on&lev_850_mb=on", "pmsl": "&var_PRMSL=on&lev_mean_sea_level=on",
            "cape_ml": "&var_CAPE=on&lev_surface=on", "cin_ml": "&var_CIN=on&lev_surface=on",
            "tot_prec": "&var_APCP=on&lev_surface=on", "hgt_0c": "&var_HGT=on&lev_0C_isotherm=on",
            "clct": "&var_TCDC=on&lev_entire_atmosphere=on", "relhum": "&var_RH=on&lev_700_mb=on",
            "h_snow": "&var_SNOD=on&lev_surface=on", "sp": "&var_PRES=on&lev_surface=on",
            "sli": "&var_4LFTX=on&lev_surface=on", "vis": "&var_VIS=on&lev_surface=on",
            "hbas_con": "&var_HGT=on&lev_cloud_base=on"
        }
        gfs_p = gfs_map.get(key, "")
        for off in [3, 6, 9, 12, 18, 24]:
            t = now - timedelta(hours=off)
            run = (t.hour // 6) * 6
            dt_s = t.strftime("%Y%m%d")
            url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t{run:02d}z.pgrb2.0p25.f{hr:03d}{gfs_p}&subregion=&leftlon=-20&rightlon=45&toplat=75&bottomlat=30&dir=%2Fgfs.{dt_s}%2F{run:02d}%2Fatmos"
            debug_logs.append(url)
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    with open("temp_gfs.grib", "wb") as f: f.write(r.content)
                    ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                    data = ds[list(ds.data_vars)[0]].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze()
                    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                    return data, lons, lats, f"{dt_s}{run:02d}", debug_logs
            except Exception: continue

    elif "ECMWF" in model:
        is_aifs = "AIFS" in model
        sys_type = "aifs" if is_aifs else "ifs"
        res_str = "0p25-beta" if is_aifs else "0p4-beta"
        
        for off in [0, 12, 24, 36]:
            t = now - timedelta(hours=off)
            run = (t.hour // 12) * 12
            dt_s = t.strftime("%Y%m%d")
            url = f"https://data.ecmwf.int/forecasts/{dt_s}/{run:02d}z/{sys_type}/{res_str}/oper/{dt_s}{run:02d}0000-{hr}h-oper-fc.grib2"
            debug_logs.append(url)
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    with open("temp_ecmwf.grib", "wb") as f: f.write(r.content)
                    e_k = {
                        "t_2m": "2t", "td_2m": "2d", "vmax_10m": "10fg", "fi": "z", "t": "t", 
                        "pmsl": "msl", "tot_prec": "tp", "clct": "tcc", "sp": "sp"
                    }.get(key, "2t")
                    ds = xr.open_dataset("temp_ecmwf.grib", engine='cfgrib', filter_by_keys={'shortName': e_k})
                    ds_var = ds[list(ds.data_vars)[0]]
                    if 'isobaricInhPa' in ds_var.dims:
                        target_p = 500 if "500" in param else 700 if "700" in param else 850
                        ds_var = ds_var.sel(isobaricInhPa=target_p)
                    data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                    return data, lons, lats, f"{dt_s}{run:02d}", debug_logs
            except Exception: continue
            
    return None, None, None, None, debug_logs

# ==============================================================================
# 6. KARTENGENERATOR & PLOTTING
# ==============================================================================
if generate:
    cleanup_temp_files()
    
    with st.spinner(f"🛰️ Lade {sel_param} aus {sel_model}..."):
        # Normalen Datensatz laden
        data, lons, lats, run_id, d_logs = fetch_meteo_data(sel_model, sel_param, sel_hour, debug_mode)
        iso_data, ilons, ilats, _, _ = fetch_meteo_data(sel_model, "Isobaren", sel_hour) if show_isobars else (None, None, None, None, None)
        
        # HEIMLICHER GEWITTER-DOWNLOAD (Nur wenn Schalter aktiv und Parameter verfügbar)
        ww_data = None
        if show_storms and sel_param != "Signifikantes Wetter" and "Signifikantes Wetter" in MODEL_ROUTER[sel_model]["params"]:
            ww_data, wlons, wlats, _, _ = fetch_meteo_data(sel_model, "Signifikantes Wetter", sel_hour, False)

    if debug_mode and d_logs:
        st.write("📡 **Interne Server-Pings (Debug):**")
        for log in d_logs[:3]: st.code(log)

    if data is not None:
        fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
        
        extents = {
            "Deutschland": [5.8, 15.2, 47.2, 55.1], 
            "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6], 
            "Mitteleuropa (DE, PL)": [4.0, 25.0, 45.0, 56.0], 
            "Alpenraum": [5.5, 17.0, 44.0, 49.5], 
            "Europa": [-12, 40, 34, 66]
        }
        ax.set_extent(extents[sel_region])

        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=12)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='black', zorder=12)
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(states, linewidth=0.5, edgecolor='black', zorder=12)

        # ----------------------------------------------------------------------
        # PLOTTING-LOGIK
        # ----------------------------------------------------------------------
        if "Temperatur" in sel_param or "850 hPa Temp." in sel_param or "Taupunkt" in sel_param:
            val_c = data - 273.15 if data.max() > 100 else data
            label_txt = "Taupunkt in °C" if "Taupunkt" in sel_param else "Temperatur in °C"
            im = ax.pcolormesh(lons, lats, val_c, cmap=cmap_temp, norm=mcolors.Normalize(vmin=-30, vmax=30), shading='auto', zorder=5)
            plt.colorbar(im, label=label_txt, shrink=0.4, pad=0.02, ticks=np.arange(-30, 31, 10))
            
        elif "CAPE" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_cape, norm=norm_cape, shading='auto', zorder=5)
            plt.colorbar(im, label="CAPE (Energie) in J/kg", shrink=0.4, ticks=[0, 100, 500, 1000, 2000, 3000, 5000])
            
        elif "CIN" in sel_param:
            im = ax.pcolormesh(lons, lats, np.abs(data), cmap=cmap_cin, norm=mcolors.Normalize(vmin=0, vmax=500), shading='auto', zorder=5)
            plt.colorbar(im, label="CIN (Hemmung/Deckel) in J/kg", shrink=0.4)
            
        elif "Radar" in sel_param: 
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_radar, norm=norm_radar, shading='auto', zorder=5)
            plt.colorbar(im, label="Radar-Reflektivität in dBZ", shrink=0.4, ticks=[0, 15, 30, 45, 60, 75])
            
        elif "Niederschlag" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_precip, norm=norm_precip, shading='auto', zorder=5)
            ticks_precip = list(range(0, 55, 5)) 
            plt.colorbar(im, label="Niederschlagssumme in mm", shrink=0.4, ticks=ticks_precip)
            
        elif "Gesamtbedeckung" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_clouds, norm=mcolors.Normalize(vmin=0, vmax=100), shading='auto', zorder=5)
            plt.colorbar(im, label="Bewölkung in %", shrink=0.4)

        elif "Rel. Feuchte" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_relhum, norm=mcolors.Normalize(vmin=0, vmax=100), shading='auto', zorder=5)
            plt.colorbar(im, label="Relative Feuchte in %", shrink=0.4)
            
        elif "Schneehöhe" in sel_param:
            val_cm = data * 100 if data.max() < 10 else data 
            im = ax.pcolormesh(lons, lats, val_cm, cmap=cmap_snow, norm=mcolors.Normalize(vmin=0.1, vmax=50), shading='auto', zorder=5)
            plt.colorbar(im, label="Schneehöhe in cm", shrink=0.4)
            
        elif "Geopot" in sel_param:
            val = (data / 9.80665) / 10 if data.max() > 10000 else data / 10
            im = ax.pcolormesh(lons, lats, val, cmap='nipy_spectral', shading='auto', zorder=5)
            plt.colorbar(im, label="Geopotential in gpdm", shrink=0.4)
            
        elif "Windböen" in sel_param:
            im = ax.pcolormesh(lons, lats, data * 3.6, cmap=cmap_wind, norm=mcolors.Normalize(vmin=0, vmax=150), shading='auto', zorder=5)
            plt.colorbar(im, label="Windböen in km/h", shrink=0.4, pad=0.02)
            
        elif "0-Grad-Grenze" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap='terrain', norm=mcolors.Normalize(vmin=0, vmax=4500), shading='auto', zorder=5)
            plt.colorbar(im, label="0-Grad-Grenze in m", shrink=0.4)
            
        elif "Bodendruck" in sel_param:
            val_hpa = data / 100 if data.max() > 5000 else data
            im = ax.pcolormesh(lons, lats, val_hpa, cmap='jet', shading='auto', zorder=5)
            plt.colorbar(im, label="Bodendruck in hPa", shrink=0.4)
            
        elif "Sichtweite" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_vis, norm=mcolors.Normalize(vmin=0, vmax=10000), shading='auto', zorder=5)
            plt.colorbar(im, label="Sichtweite in m (Weiß=Nebel)", shrink=0.4)
            
        elif "Wolkenuntergrenze" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_base, norm=norm_base, shading='auto', zorder=5)
            plt.colorbar(im, label="Wolkenuntergrenze in m", shrink=0.4, ticks=base_levels)
            
        elif "Wolkenobergrenze" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap='Blues', norm=mcolors.Normalize(vmin=0, vmax=13000), shading='auto', zorder=5)
            plt.colorbar(im, label="Wolkenobergrenze in m", shrink=0.4)
            
        elif "Spezifische Feuchte" in sel_param:
            im = ax.pcolormesh(lons, lats, data * 1000, cmap='YlGnBu', norm=mcolors.Normalize(vmin=0, vmax=20), shading='auto', zorder=5)
            plt.colorbar(im, label="Spezifische Feuchte in g/kg", shrink=0.4)
            
        elif "Helizität" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_heli, norm=mcolors.Normalize(vmin=0, vmax=500), shading='auto', zorder=5)
            plt.colorbar(im, label="Helizität / SRH (Tornadogefahr) in m²/s²", shrink=0.4)
            
        elif "Sonnenscheindauer" in sel_param:
            im = ax.pcolormesh(lons, lats, data / 60, cmap=cmap_sun, norm=mcolors.Normalize(vmin=0, vmax=60), shading='auto', zorder=5)
            plt.colorbar(im, label="Sonnenscheindauer (Minuten/Stunde)", shrink=0.4)
            
        elif "Lifted Index" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_lifted, norm=mcolors.Normalize(vmin=-10, vmax=10), shading='auto', zorder=5)
            plt.colorbar(im, label="Lifted Index in K (Blau=Stabil, Rot=Gewitter)", shrink=0.4)
            
        elif "Signifikantes Wetter" in sel_param:
            grid = np.zeros_like(data)
            for i, (l, (c, codes)) in enumerate(WW_LEGEND_DATA.items(), 1):
                for code in codes: grid[data == code] = i
            ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, shading='nearest', zorder=5)
            patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND_DATA.items()]
            ax.legend(handles=patches, loc='lower left', title="Wetter-Klassifikation", fontsize='6', title_fontsize='7', framealpha=0.9).set_zorder(25)

        # ----------------------------------------------------------------------
        # NEU: DIE ROTE GEWITTER-SCHRAFFUR OVERLAY (//////)
        # ----------------------------------------------------------------------
        if show_storms and ww_data is not None:
            # Maske bauen: Gewitter-Codes (95=leicht, 96/97/99=mäßig bis schwer)
            storm_mask = np.isin(ww_data, [95, 96, 97, 99])
            
            if np.any(storm_mask):
                plot_ww = np.zeros_like(ww_data)
                plot_ww[storm_mask] = 1 
                
                # Macht die diagonalen Striche fett und rot sichtbar!
                plt.rcParams['hatch.linewidth'] = 2.0 
                ax.contourf(wlons, wlats, plot_ww, levels=[0.5, 1.5], colors='none', hatches=['////'], edgecolors='red', zorder=10)

        # ----------------------------------------------------------------------
        # ISOBAREN OVERLAY
        # ----------------------------------------------------------------------
        if iso_data is not None:
            p_hpa = iso_data / 100 if iso_data.max() > 5000 else iso_data
            if ilons.ndim == 1: ilons, ilats = np.meshgrid(ilons, ilats)
            cs = ax.contour(ilons, ilats, p_hpa, colors='black', linewidths=0.7, levels=np.arange(940, 1060, 4), zorder=20)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

        # ----------------------------------------------------------------------
        # HEADER INFO
        # ----------------------------------------------------------------------
        v_dt_utc = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)
        v_dt_loc = v_dt_utc.astimezone(LOCAL_TZ)
        tz_str = "MESZ" if v_dt_loc.dst() else "MEZ"
        
        info_txt = f"Modell: {sel_model}\nParameter: {sel_param}\nTermin: {v_dt_loc.strftime('%d.%m.%Y %H:%M')} {tz_str}\nModell-Lauf: {run_id[-2:]}Z"
        ax.text(0.02, 0.98, info_txt, transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3', edgecolor='gray'), zorder=30)

        st.pyplot(fig)
        
        # ----------------------------------------------------------------------
        # BILD-DOWNLOAD GENERIEREN
        # ----------------------------------------------------------------------
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.download_button(
                label="📥 Karte als PNG speichern",
                data=img_buffer,
                file_name=f"WarnwetterBB_{sel_model}_{sel_param.replace(' ', '_')}_{sel_hour}h.png",
                mime="image/png",
                use_container_width=True
            )
        
        cleanup_temp_files()
        
    else:
        st.error(f"⚠️ Der Server von {sel_model} liefert für '{sel_param}' (+{sel_hour}h) aktuell keine Daten aus.")
        st.info("💡 Tipp: Versuch eine etwas spätere Vorhersagestunde oder aktiviere links die 'Entwickler-Konsole', um die Pings zu prüfen!")
