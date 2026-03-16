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
import numpy as np

# ==============================================================================
# 1. SETUP & KONFIGURATION DER METEO-ZENTRALE
# ==============================================================================
st.set_page_config(page_title="WarnwetterBB | Unwetter-Zentrale", layout="wide", initial_sidebar_state="expanded")

def cleanup_temp_files():
    """Löscht temporäre GRIB-Dateien restlos vom Server, um Memory Leaks zu verhindern."""
    for file in ["temp.grib", "temp_gfs.grib", "temp_ecmwf.grib"]:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception:
                pass

# ==============================================================================
# 2. METEOROLOGISCHE FARBSKALEN (HOCHPRÄZISE & ORIGINAL DWD)
# ==============================================================================

# --- TEMPERATUR & TAUPUNKT (Die schlagartigen 10er-Sprünge) ---
temp_colors = [
    (0.0, '#D3D3D3'), (5/60, '#FFFFFF'), (10/60, '#FFC0CB'), (15/60, '#FF00FF'),
    (20/60, '#800080'), (20.01/60, '#00008B'), (25/60, '#0000CD'), (29.99/60, '#ADD8E6'),
    (30/60, '#006400'), (35/60, '#008000'), (39/60, '#90EE90'), (39.99/60, '#90EE90'),
    (40/60, '#FFFF00'), (45/60, '#FFA500'), (50/60, '#FF0000'), (55/60, '#8B0000'), (60/60, '#800080')
]
cmap_temp = mcolors.LinearSegmentedColormap.from_list("custom_temp", temp_colors)

# --- CAPE (EXAKTE GRENZWERTE) ---
cape_levels = [0, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]
cape_colors = [
    '#006400', '#2E8B57', '#ADFF2F', '#FFFF00', '#FFB347', '#FFA500', 
    '#FF4500', '#FF0000', '#8B0000', '#800080', '#FF00FF', '#FFFFFF', '#808080', '#404040'
]
cmap_cape = mcolors.ListedColormap(cape_colors)
norm_cape = mcolors.BoundaryNorm(cape_levels, cmap_cape.N)

# --- ORIGINAL DWD RADAR-SKALA (HARTE SCHWELLENWERTE) ---
radar_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
radar_colors = [
    '#FFFFFF', # 0-5: Nichts
    '#B0E0E6', # 5-10: Sehr helles Blau (Nieselregen)
    '#00BFFF', # 10-15: Hellblau
    '#0000FF', # 15-20: Blau (Leichter Regen)
    '#00FF00', # 20-25: Hellgrün
    '#32CD32', # 25-30: Grün (Mäßiger Regen)
    '#008000', # 30-35: Dunkelgrün
    '#FFFF00', # 35-40: Gelb (Starker Landregen)
    '#FFA500', # 40-45: Orange (Konvektion beginnt)
    '#FF0000', # 45-50: Rot (Gewitter)
    '#8B0000', # 50-55: Dunkelrot (Starkgewitter)
    '#FF00FF', # 55-60: Magenta (Hagel / extremer Starkregen)
    '#800080', # 60-65: Lila
    '#4B0082', # 65-70: Dunkellila (Superzelle)
    '#E6E6FA'  # 70+: Weiß/Hellgrau (Extremer Hagel / Tornado-Signatur)
]
cmap_radar = mcolors.ListedColormap(radar_colors)
norm_radar = mcolors.BoundaryNorm(radar_levels, cmap_radar.N)

# --- UNWETTER-INDIZES & PARAMETER ---
cmap_cin = mcolors.LinearSegmentedColormap.from_list("cin", ['#FFFFFF', '#ADD8E6', '#0000FF', '#00008B', '#000000'], N=256)
cmap_precip = mcolors.LinearSegmentedColormap.from_list("precip", ['#FFFFFF', '#ADD8E6', '#0000FF', '#800080', '#8B0000'], N=256)
cmap_clouds = mcolors.LinearSegmentedColormap.from_list("clouds", ['#1E90FF', '#87CEEB', '#D3D3D3', '#FFFFFF'], N=256)
cmap_relhum = mcolors.LinearSegmentedColormap.from_list("relhum", ['#8B4513', '#F4A460', '#FFFFE0', '#90EE90', '#008000', '#0000FF'], N=256)
cmap_snow = mcolors.LinearSegmentedColormap.from_list("snow", ['#CCFFCC', '#FFFFFF', '#ADD8E6', '#0000FF', '#800080'], N=256)
cmap_vis = mcolors.LinearSegmentedColormap.from_list("vis", ['#FFFFFF', '#D3D3D3', '#87CEEB', '#1E90FF'], N=256)
cmap_base = mcolors.LinearSegmentedColormap.from_list("base", ['#808080', '#A9A9A9', '#ADD8E6', '#FFFFFF'], N=256)
W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']
cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind", W_COLORS, N=256)

# Spezielle Skalen für die neuen Parameter
cmap_heli = mcolors.LinearSegmentedColormap.from_list("heli", ['#FFFFFF', '#00FF00', '#FFFF00', '#FF0000', '#800080', '#000000'], N=256)
cmap_lifted = mcolors.LinearSegmentedColormap.from_list("lifted", ['#FF00FF', '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#0000FF'], N=256)
cmap_sun = mcolors.LinearSegmentedColormap.from_list("sun", ['#808080', '#FFD700', '#FFA500', '#FF8C00'], N=256)

# --- SIGNIFIKANTES WETTER (DWD CODES) ---
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
# 3. DYNAMISCHE SIDEBAR (MOBILE-OPTIMIERT: KEINE TASTATUR!)
# ==============================================================================
with st.sidebar:
    st.header("🛰️ Modell-Zentrale")
    
    # 1. Modellauswahl (Mit dem neuen ECMWF-AIFS KI-Modell!)
    with st.expander("🌍 1. Modell wählen", expanded=True):
        model_list = ["ICON-D2", "ICON-D2-RUC", "ICON-EU", "ICON (Global)", "GFS (NOAA)", "ECMWF", "ECMWF-AIFS (KI-Modell)"]
        sel_model = st.radio("Wettermodell", model_list, label_visibility="collapsed")
    
    # 2. Regionen (Dynamischer Schutz vor leeren Karten)
    with st.expander("🗺️ 2. Karten-Ausschnitt", expanded=False):
        reg_opts = ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum"]
        if sel_model not in ["ICON-D2", "ICON-D2-RUC"]: 
            reg_opts.append("Europa")
        sel_region = st.radio("Region", reg_opts, label_visibility="collapsed")
    
    # 3. Parameter-Baum (Passt sich extrem präzise dem gewählten Modell an!)
    with st.expander("🌪️ 3. Parameter wählen", expanded=True):
        p_opts = [
            "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", 
            "Bodendruck (hPa)", "500 hPa Geopot. Höhe", "850 hPa Temp.", 
            "Niederschlag (mm)"
        ]
        
        # Konvektions-Parameter (Fast überall verfügbar)
        if "ICON" in sel_model or "GFS" in sel_model or "ECMWF" in sel_model:
            p_opts.extend(["CAPE (J/kg)", "CIN (J/kg)", "Gesamtbedeckung (%)", "Rel. Feuchte 700 hPa (%)", "Schneehöhe (cm)"])
            
        # DWD Exklusiv (Die ganz tiefen Detail-Parameter)
        if "ICON" in sel_model:
            p_opts.extend(["Signifikantes Wetter", "Sichtweite (m)", "Wolkenuntergrenze (m)", "Wolkenobergrenze (m)", "Spezifische Feuchte (g/kg)"])
        
        # ICON-D2 Exklusiv (High-Res)
        if "ICON-D2" in sel_model:
            p_opts.extend(["Simuliertes Radar (dBZ)", "Helizität / SRH (m²/s²)", "Sonnenscheindauer (Min)"])
            
        # GFS & ECMWF Exklusiv
        if "GFS" in sel_model or "ECMWF" in sel_model:
            p_opts.extend(["0-Grad-Grenze (m)", "Lifted Index (K)"])
            
        sel_param = st.radio("Parameter", p_opts, label_visibility="collapsed")
    
    # 4. Vorhersage-Stunde (Scrollbare Liste ohne Tastatur-Popup)
    with st.expander("⏱️ 4. Vorhersage-Stunde", expanded=True):
        if "RUC" in sel_model: hours = list(range(1, 28))
        elif "EU" in sel_model: hours = list(range(1, 79))
        elif "GFS" in sel_model: hours = list(range(3, 123, 3))
        elif "ECMWF" in sel_model: hours = list(range(3, 147, 3))
        elif "Global" in sel_model: hours = list(range(3, 123, 3))
        else: hours = list(range(1, 49))
            
        sel_hour_str = st.radio("Zeit", [f"+{h}h" for h in hours], label_visibility="collapsed")
        sel_hour = int(sel_hour_str.replace("+", "").replace("h", ""))
    
    show_isobars = st.checkbox("Isobaren (Luftdruck) einblenden", value=True)
    st.markdown("---")
    generate = st.button("🚀 Profi-Karte generieren", use_container_width=True)


# ==============================================================================
# 4. DATA FETCH ENGINE (ABSOLUT FEHLERTOLERANT & UMFASSEND)
# ==============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_meteo_data(model, param, hr):
    # Mapping auf die GRIB ShortNames der Server
    p_map = {
        "Temperatur 2m (°C)": "t_2m", "Taupunkt 2m (°C)": "td_2m", "Windböen (km/h)": "vmax_10m", 
        "Bodendruck (hPa)": "sp", "500 hPa Geopot. Höhe": "fi", "850 hPa Temp.": "t", 
        "Signifikantes Wetter": "ww", "Isobaren": "pmsl", "CAPE (J/kg)": "cape_ml", 
        "CIN (J/kg)": "cin_ml", "Niederschlag (mm)": "tot_prec", "Simuliertes Radar (dBZ)": "dbz_cmax",
        "0-Grad-Grenze (m)": "hgt_0c", "Gesamtbedeckung (%)": "clct", "Rel. Feuchte 700 hPa (%)": "relhum", 
        "Schneehöhe (cm)": "h_snow", "Sichtweite (m)": "vis", "Wolkenuntergrenze (m)": "hbas_con",
        "Wolkenobergrenze (m)": "htop_con", "Spezifische Feuchte (g/kg)": "qv",
        "Helizität / SRH (m²/s²)": "uh_max", "Sonnenscheindauer (Min)": "dur_sun", "Lifted Index (K)": "sli"
    }
    
    key = p_map.get(param, "t_2m")
    now = datetime.now(timezone.utc)
    headers = {'User-Agent': 'Mozilla/5.0'}

    # --- A: DWD LOGIK (D2, RUC, EU, Global) ---
    if "ICON" in model:
        is_ruc = "RUC" in model
        is_global = "Global" in model
        
        if is_global: m_dir, reg_str = "icon", "icon_global"
        elif is_ruc: m_dir, reg_str = "icon-d2-ruc", "icon-d2-ruc_germany"
        elif "D2" in model: m_dir, reg_str = "icon-d2", "icon-d2_germany"
        else: m_dir, reg_str = "icon-eu", "icon-eu_europe"
        
        for off in range(1, 15):
            t = now - timedelta(hours=off)
            if is_ruc: run = t.hour
            elif is_global: run = (t.hour // 6) * 6
            else: run = (t.hour // 3) * 3
            
            dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
            
            # Bestimmung der Druckfläche für das GRIB-File
            l_type = "pressure-level" if key in ["fi", "t", "relhum", "qv"] else "single-level"
            if l_type == "pressure-level": 
                lvl_str = f"{'500' if '500' in param else '700' if '700' in param else '850'}_"
            else: 
                lvl_str = "2d_"
            
            url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{reg_str}_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{lvl_str}{key}.grib2.bz2"
            
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with bz2.open(io.BytesIO(r.content)) as f_bz2:
                        with open("temp.grib", "wb") as f_out: 
                            f_out.write(f_bz2.read())
                    ds = xr.open_dataset("temp.grib", engine='cfgrib')
                    ds_var = ds[list(ds.data_vars)[0]]
                    
                    if 'isobaricInhPa' in ds_var.dims:
                        target_p = 500 if "500" in param else 700 if "700" in param else 850
                        ds_var = ds_var.sel(isobaricInhPa=target_p)
                        
                    data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                    lons, lats = ds.longitude.values, ds.latitude.values
                    if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                    return data, lons, lats, dt_s
            except Exception: continue

    # --- B: GFS LOGIK ---
    elif "GFS" in model:
        gfs_map = {
            "t_2m": "&var_TMP=on&lev_2_m_above_ground=on", "td_2m": "&var_DPT=on&lev_2_m_above_ground=on",
            "vmax_10m": "&var_GUST=on&lev_surface=on", "fi": "&var_HGT=on&lev_500_mb=on",
            "t": "&var_TMP=on&lev_850_mb=on", "pmsl": "&var_PRMSL=on&lev_mean_sea_level=on",
            "cape_ml": "&var_CAPE=on&lev_surface=on", "cin_ml": "&var_CIN=on&lev_surface=on",
            "tot_prec": "&var_APCP=on&lev_surface=on", "hgt_0c": "&var_HGT=on&lev_0C_isotherm=on",
            "clct": "&var_TCDC=on&lev_entire_atmosphere=on", "relhum": "&var_RH=on&lev_700_mb=on",
            "h_snow": "&var_SNOD=on&lev_surface=on", "sp": "&var_PRES=on&lev_surface=on",
            "sli": "&var_4LFTX=on&lev_surface=on"
        }
        gfs_p = gfs_map.get(key, "")
        
        for off in [3, 6, 9, 12, 18]:
            t = now - timedelta(hours=off)
            run = (t.hour // 6) * 6
            dt_s = t.strftime("%Y%m%d")
            url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t{run:02d}z.pgrb2.0p25.f{hr:03d}{gfs_p}&subregion=&leftlon=-20&rightlon=45&toplat=75&bottomlat=30&dir=%2Fgfs.{dt_s}%2F{run:02d}%2Fatmos"
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200:
                    with open("temp_gfs.grib", "wb") as f: f.write(r.content)
                    ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                    data = ds[list(ds.data_vars)[0]].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze()
                    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                    return data, lons, lats, f"{dt_s}{run:02d}"
            except Exception: continue

    # --- C: ECMWF & AIFS LOGIK ---
    elif "ECMWF" in model:
        # Check ob das normale ECMWF oder das neue KI-Modell (AIFS) gewählt wurde
        sys_type = "aifs" if "AIFS" in model else "ifs"
        
        for off in [0, 12, 24, 36]:
            t = now - timedelta(hours=off)
            run = (t.hour // 12) * 12
            dt_s = t.strftime("%Y%m%d")
            url = f"https://data.ecmwf.int/forecasts/{dt_s}/{run:02d}z/{sys_type}/0p25-beta/oper/{dt_s}{run:02d}0000-{hr}h-oper-fc.grib2"
            
            try:
                r = requests.get(url, timeout=25)
                if r.status_code == 200:
                    with open("temp_ecmwf.grib", "wb") as f: f.write(r.content)
                    
                    e_k = {
                        "t_2m": "2t", "td_2m": "2d", "vmax_10m": "10fg", "fi": "z", "t": "t", 
                        "pmsl": "msl", "cape_ml": "cape", "cin_ml": "cin", "tot_prec": "tp",
                        "clct": "tcc", "relhum": "r", "h_snow": "sd", "sp": "sp", "sli": "li"
                    }.get(key, "2t")
                    
                    ds = xr.open_dataset("temp_ecmwf.grib", engine='cfgrib', filter_by_keys={'shortName': e_k})
                    ds_var = ds[list(ds.data_vars)[0]]
                    if 'isobaricInhPa' in ds_var.dims:
                        target_p = 500 if "500" in param else 700 if "700" in param else 850
                        ds_var = ds_var.sel(isobaricInhPa=target_p)
                    data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                    return data, lons, lats, f"{dt_s}{run:02d}"
            except Exception: continue
    return None, None, None, None

# ==============================================================================
# 5. KARTENGENERATOR & PLOTTING (DIE ENGINE)
# ==============================================================================
if generate:
    cleanup_temp_files()
    with st.spinner(f"🛰️ Kontaktiere Wetterrechner... Lade {sel_param} aus {sel_model}"):
        data, lons, lats, run_id = fetch_meteo_data(sel_model, sel_param, sel_hour)
        iso_data, ilons, ilats, _ = fetch_meteo_data(sel_model, "Isobaren", sel_hour) if show_isobars else (None, None, None, None)

    if data is not None:
        fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=120)
        
        extents = {
            "Deutschland": [5.8, 15.2, 47.2, 55.1], 
            "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6], 
            "Mitteleuropa (DE, PL)": [4.0, 25.0, 45.0, 56.0], 
            "Alpenraum": [5.5, 17.0, 44.0, 49.5], 
            "Europa": [-12, 40, 34, 66]
        }
        ax.set_extent(extents[sel_region])

        # Scharfe topografische Overlays
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=12)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='black', zorder=12)
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(states, linewidth=0.5, edgecolor='black', zorder=12)

        # ----------------------------------------------------------------------
        # PLOTTING-LOGIK FÜR 22 PARAMETER
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
            # DWD Radar-Skala mit exakten, harten Kanten!
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_radar, norm=norm_radar, shading='auto', zorder=5)
            plt.colorbar(im, label="Radar-Reflektivität in dBZ", shrink=0.4, ticks=[0, 15, 30, 45, 60, 75])
            
        elif "Niederschlag" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_precip, norm=mcolors.Normalize(vmin=0.1, vmax=50), shading='auto', zorder=5)
            plt.colorbar(im, label="Niederschlagssumme in mm", shrink=0.4)
            
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
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_base, norm=mcolors.Normalize(vmin=0, vmax=3000), shading='auto', zorder=5)
            plt.colorbar(im, label="Wolkenuntergrenze in m", shrink=0.4)
            
        elif "Wolkenobergrenze" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap='Blues', norm=mcolors.Normalize(vmin=0, vmax=13000), shading='auto', zorder=5)
            plt.colorbar(im, label="Wolkenobergrenze in m (Je höher, desto stärker das Gewitter)", shrink=0.4)
            
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
            # Lifted Index: Negativ = Gewitter!
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_lifted, norm=mcolors.Normalize(vmin=-10, vmax=10), shading='auto', zorder=5)
            plt.colorbar(im, label="Lifted Index in K (Blau=Stabil, Rot/Magenta=Gewitter)", shrink=0.4)
            
        elif "Signifikantes Wetter" in sel_param:
            grid = np.zeros_like(data)
            for i, (l, (c, codes)) in enumerate(WW_LEGEND_DATA.items(), 1):
                for code in codes: grid[data == code] = i
            ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, shading='nearest', zorder=5)
            patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND_DATA.items()]
            ax.legend(handles=patches, loc='lower left', title="Wetter-Klassifikation", fontsize='6', title_fontsize='7', framealpha=0.9).set_zorder(25)

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
        v_dt = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)
        info_txt = f"Modell: {sel_model}\nParameter: {sel_param}\nTermin: {v_dt.strftime('%d.%m.%Y %H:00')} UTC\nModell-Lauf: {run_id[-2:]}Z"
        ax.text(0.02, 0.98, info_txt, transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3', edgecolor='gray'), zorder=30)

        st.pyplot(fig)
        cleanup_temp_files()
        
    else:
        st.error(f"⚠️ Schwerer Datenfehler: {sel_model} liefert für '{sel_param}' (+{sel_hour}h) aktuell keine Daten aus. Versuch eine andere Uhrzeit.")
else:
    st.info("ℹ️ Keine Tastatur mehr! Wähle links ganz entspannt deine Parameter und klicke auf Karte generieren.")
