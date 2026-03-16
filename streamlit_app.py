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
# 1. SETUP & KONFIGURATION
# ==============================================================================
st.set_page_config(page_title="WarnwetterBB | Unwetter-Zentrale", layout="wide")

def cleanup_temp_files():
    """Löscht temporäre GRIB-Dateien, um den Server-Speicher sauber zu halten."""
    for file in ["temp.grib", "temp_gfs.grib", "temp_ecmwf.grib"]:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception:
                pass

# ==============================================================================
# 2. FARBSKALEN DEFINITIONEN (HOCHPRÄZISE)
# ==============================================================================

# --- TEMPERATUR (Die bewährten schlagartigen 10er-Sprünge) ---
temp_colors = [
    (0.0, '#D3D3D3'),       # -30: hellgrau
    (5/60, '#FFFFFF'),      # -25: weiß
    (10/60, '#FFC0CB'),     # -20: rosa
    (15/60, '#FF00FF'),     # -15: magenta
    (20/60, '#800080'),     # -10: lila
    (20.01/60, '#00008B'),  # SCHLAGARTIG zu dunkelblau
    (25/60, '#0000CD'),     # -5: dunkles blau
    (29.99/60, '#ADD8E6'),  # kurz vor 0: hellblau
    (30/60, '#006400'),     # +0: SCHLAGARTIG zu dunkelgrün
    (35/60, '#008000'),     # +5: grün
    (39/60, '#90EE90'),     # +9: hellgrün
    (39.99/60, '#90EE90'),  # kurz vor 10: hellgrün
    (40/60, '#FFFF00'),     # +10: SCHLAGARTIG zu gelb
    (45/60, '#FFA500'),     # +15: orange
    (50/60, '#FF0000'),     # +20: rot
    (55/60, '#8B0000'),     # +25: dunkelrot
    (60/60, '#800080')      # +30: lila
]
cmap_temp = mcolors.LinearSegmentedColormap.from_list("custom_temp", temp_colors)

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
# Indexbasierte Colormap für Wetter
cmap_ww = mcolors.ListedColormap(['#FFFFFF00'] + [c for l, (c, codes) in WW_LEGEND_DATA.items()])

# --- CAPE (DEINE EXAKTE SKALA) ---
# Mit BoundaryNorm springt die Farbe exakt bei diesen Grenzwerten um!
cape_levels = [0, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]
cape_colors = [
    '#006400', # 0-25: dunkles grün
    '#008000', # 25-50: grün
    '#ADFF2F', # 50-100: gelbgrün
    '#FFFF00', # 100-250: gelb
    '#FFB347', # 250-500: hellorange
    '#FFA500', # 500-750: orange
    '#FF4500', # 750-1000: rotorange
    '#FF0000', # 1000-1500: rot
    '#8B0000', # 1500-2000: dunkelrot
    '#800080', # 2000-2500: lila
    '#FF00FF', # 2500-3000: magenta
    '#FFFFFF', # 3000-4000: weiß
    '#808080', # 4000-5000: grau
    '#404040'  # 5000+: dunkelgrau
]
cmap_cape = mcolors.ListedColormap(cape_colors)
norm_cape = mcolors.BoundaryNorm(cape_levels, cmap_cape.N)

# --- WEITERE UNWETTER-SKALEN ---
# CIN (Gewitterhemmung / Deckel)
cmap_cin = mcolors.LinearSegmentedColormap.from_list("cin", ['#FFFFFF', '#ADD8E6', '#0000FF', '#00008B', '#000000'], N=256)
# Radar-Reflektivität (dBZ)
cmap_radar = mcolors.LinearSegmentedColormap.from_list("radar", ['#FFFFFF', '#00FFFF', '#0000FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#FF00FF', '#E6E6FA'], N=256)
# Niederschlag (mm)
cmap_precip = mcolors.LinearSegmentedColormap.from_list("precip", ['#FFFFFF', '#ADD8E6', '#0000FF', '#800080', '#8B0000'], N=256)
# Wind (km/h)
W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']
cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind", W_COLORS, N=256)


# ==============================================================================
# 3. SIDEBAR & BENUTZEROBERFLÄCHE
# ==============================================================================
with st.sidebar:
    st.header("🛰️ Modell-Zentrale")
    
    # 1. Modellauswahl
    sel_model = st.selectbox("Wettermodell", ["ICON-D2", "ICON-D2-RUC", "ICON-EU", "GFS (NOAA)", "ECMWF"])
    
    # 2. Regionen (Fokus auf präzise Analyse)
    sel_region = st.selectbox("Karten-Ausschnitt", ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"])
    
    # 3. Dynamischer Parameter-Baum
    p_opts = ["Temperatur 2m (°C)", "Windböen (km/h)", "500 hPa Geopot. Höhe", "850 hPa Temp.", "Niederschlag (mm)", "CAPE (J/kg)", "CIN (J/kg)"]
    if "ICON" in sel_model:
        p_opts.append("Signifikantes Wetter")
    if "ICON-D2" in sel_model:
        p_opts.append("Simuliertes Radar (dBZ)")
        
    sel_param = st.selectbox("Meteorologischer Parameter", p_opts)
    
    # 4. Zeit-Slider (Passt sich der Modellauslösung an)
    max_h, step_h = 48, 1
    if "RUC" in sel_model:
        max_h = 27
    elif "EU" in sel_model or "GFS" in sel_model:
        max_h = 120
    elif "ECMWF" in sel_model:
        max_h, step_h = 144, 3
        
    sel_hour = st.slider("Vorhersagezeit (+h)", step_h, max_h, step_h, step=step_h)
    
    # 5. Overlays
    show_isobars = st.checkbox("Isobaren (Bodendruck) einblenden", value=True)
    
    st.markdown("---")
    generate = st.button("🚀 Profi-Karte generieren", use_container_width=True)
    st.caption("Daten: DWD OpenData | NOAA NOMADS | ECMWF IFS")


# ==============================================================================
# 4. MULTI-MODEL DATA FETCH ENGINE (ROBUST & FEHLERFREI)
# ==============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_meteo_data(model, param, hr):
    """
    Lädt GRIB-Daten vom Server des jeweiligen Wetterdienstes.
    Ignoriert überflüssige Dimensionen (Lila-Schutz) und mappt Parameter.
    """
    # Parameter-Zuordnung der GRIB ShortNames
    p_map = {
        "Temperatur 2m (°C)": "t_2m",
        "Windböen (km/h)": "vmax_10m", 
        "500 hPa Geopot. Höhe": "fi",
        "850 hPa Temp.": "t", 
        "Signifikantes Wetter": "ww",
        "Isobaren": "pmsl",
        "CAPE (J/kg)": "cape_ml",
        "CIN (J/kg)": "cin_ml",
        "Niederschlag (mm)": "tot_prec",
        "Simuliertes Radar (dBZ)": "dbz_cmax"
    }
    key = p_map[param]
    now = datetime.now(timezone.utc)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    # --- A: DWD LOGIK (ICON-D2, RUC, EU) ---
    if "ICON" in model:
        is_ruc = "RUC" in model
        m_dir = "icon-d2-ruc" if is_ruc else ("icon-d2" if "D2" in model else "icon-eu")
        reg_str = "europe" if "eu" in m_dir else "germany"
        
        # Iteration rückwärts, falls der neueste Lauf noch rechnet
        for off in range(1, 15):
            t = now - timedelta(hours=off)
            run = t.hour if is_ruc else (t.hour // 3) * 3
            dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
            
            l_type = "pressure-level" if key in ["fi", "t"] else "single-level"
            lvl_str = f"{'500' if '500' in param else '850'}_" if l_type == "pressure-level" else "2d_"
            
            url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{m_dir}_{reg_str}_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{lvl_str}{key}.grib2.bz2"
            
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with bz2.open(io.BytesIO(r.content)) as f_bz2:
                        with open("temp.grib", "wb") as f_out: 
                            f_out.write(f_bz2.read())
                            
                    ds = xr.open_dataset("temp.grib", engine='cfgrib')
                    ds_var = ds[list(ds.data_vars)[0]]
                    
                    # Expliziter Höhen-/Druck-Filter gegen den 'Lila-Fehler'
                    if 'isobaricInhPa' in ds_var.dims:
                        target_p = 500 if "500" in param else 850
                        ds_var = ds_var.sel(isobaricInhPa=target_p)
                        
                    data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                    lons, lats = ds.longitude.values, ds.latitude.values
                    if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                    return data, lons, lats, dt_s
            except Exception:
                continue

    # --- B: GFS LOGIK (NOAA NOMADS Server) ---
    elif "GFS" in model:
        # GFS braucht exakte Level-Filter für den Download
        if key == "t_2m": gfs_p = "&var_TMP=on&lev_2_m_above_ground=on"
        elif key == "vmax_10m": gfs_p = "&var_GUST=on&lev_surface=on"
        elif key == "fi": gfs_p = "&var_HGT=on&lev_500_mb=on"
        elif key == "t": gfs_p = "&var_TMP=on&lev_850_mb=on"
        elif key == "pmsl": gfs_p = "&var_PRMSL=on&lev_mean_sea_level=on"
        elif key == "cape_ml": gfs_p = "&var_CAPE=on&lev_surface=on"
        elif key == "cin_ml": gfs_p = "&var_CIN=on&lev_surface=on"
        elif key == "tot_prec": gfs_p = "&var_APCP=on&lev_surface=on"
        
        for off in [3, 6, 9, 12, 18]:
            t = now - timedelta(hours=off)
            run = (t.hour // 6) * 6
            dt_s = t.strftime("%Y%m%d")
            url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t{run:02d}z.pgrb2.0p25.f{hr:03d}{gfs_p}&subregion=&leftlon=-20&rightlon=45&toplat=75&bottomlat=30&dir=%2Fgfs.{dt_s}%2F{run:02d}%2Fatmos"
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200:
                    with open("temp_gfs.grib", "wb") as f: 
                        f.write(r.content)
                    ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                    data = ds[list(ds.data_vars)[0]].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze()
                    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                    return data, lons, lats, f"{dt_s}{run:02d}"
            except Exception:
                continue

    # --- C: ECMWF LOGIK (IFS Open Data) ---
    elif "ECMWF" in model:
        for off in [0, 12, 24, 36]:
            t = now - timedelta(hours=off)
            run = (t.hour // 12) * 12
            dt_s = t.strftime("%Y%m%d")
            url = f"https://data.ecmwf.int/forecasts/{dt_s}/{run:02d}z/ifs/0p4-beta/oper/{dt_s}{run:02d}0000-{hr}h-oper-fc.grib2"
            try:
                r = requests.get(url, timeout=25)
                if r.status_code == 200:
                    with open("temp_ecmwf.grib", "wb") as f: 
                        f.write(r.content)
                    
                    e_k = {"t_2m": "2t", "vmax_10m": "10fg", "fi": "z", "t": "t", "pmsl": "msl", "cape_ml": "cape", "cin_ml": "cin", "tot_prec": "tp"}.get(key, "2t")
                    ds = xr.open_dataset("temp_ecmwf.grib", engine='cfgrib', filter_by_keys={'shortName': e_k})
                    ds_var = ds[list(ds.data_vars)[0]]
                    
                    if 'isobaricInhPa' in ds_var.dims:
                        ds_var = ds_var.sel(isobaricInhPa=500 if "500" in param else 850)
                        
                    data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                    return data, lons, lats, f"{dt_s}{run:02d}"
            except Exception:
                continue
                
    return None, None, None, None


# ==============================================================================
# 5. KARTENGENERATOR & PLOTTING-ROUTINEN
# ==============================================================================
if generate:
    cleanup_temp_files()
    with st.spinner(f"🛰️ Kontaktiere Wetterrechner und lade Daten für {sel_param}..."):
        data, lons, lats, run_id = fetch_meteo_data(sel_model, sel_param, sel_hour)
        iso_data, ilons, ilats, _ = fetch_meteo_data(sel_model, "Isobaren", sel_hour) if show_isobars else (None, None, None, None)

    if data is not None:
        # Layout & Projektion
        fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=120)
        
        # Kartengrenzen festlegen
        extents = {
            "Deutschland": [5.8, 15.2, 47.2, 55.1], 
            "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6], 
            "Mitteleuropa (DE, PL)": [4.0, 25.0, 45.0, 56.0], 
            "Alpenraum": [5.5, 17.0, 44.0, 49.5], 
            "Europa": [-12, 40, 34, 66]
        }
        ax.set_extent(extents[sel_region])

        # Scharfe topografische Grenzen & Bundesländer
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=12)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='black', zorder=12)
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(states, linewidth=0.5, edgecolor='black', zorder=12)

        # ----------------------------------------------------------------------
        # FARB- & WERT-ZUWEISUNGEN (PARAMETER SPEZIFISCH)
        # ----------------------------------------------------------------------
        
        # Temperatur
        if "Temperatur" in sel_param or "850 hPa Temp." in sel_param:
            val_c = data - 273.15 if data.max() > 100 else data
            im = ax.pcolormesh(lons, lats, val_c, cmap=cmap_temp, norm=mcolors.Normalize(vmin=-30, vmax=30), shading='auto', zorder=5)
            plt.colorbar(im, label="Temperatur in °C", shrink=0.4, pad=0.02, ticks=np.arange(-30, 31, 10))
            
        # CAPE (Gewitterenergie mit exakten Grenz-Sprüngen)
        elif "CAPE" in sel_param:
            # norm_cape erzwingt, dass die Farbe genau an deinen gewünschten Werten (25, 50, 100...) umspringt
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_cape, norm=norm_cape, shading='auto', zorder=5)
            plt.colorbar(im, label="CAPE (Energie) in J/kg", shrink=0.4, ticks=[0, 100, 500, 1000, 2000, 3000, 5000])
            
        # CIN (Gewitterhemmung)
        elif "CIN" in sel_param:
            im = ax.pcolormesh(lons, lats, np.abs(data), cmap=cmap_cin, norm=mcolors.Normalize(vmin=0, vmax=500), shading='auto', zorder=5)
            plt.colorbar(im, label="CIN (Hemmung/Deckel) in J/kg", shrink=0.4)
            
        # Simuliertes Radar
        elif "Radar" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_radar, norm=mcolors.Normalize(vmin=0, vmax=65), shading='auto', zorder=5)
            plt.colorbar(im, label="Radar-Reflektivität in dBZ", shrink=0.4)
            
        # Niederschlag
        elif "Niederschlag" in sel_param:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_precip, norm=mcolors.Normalize(vmin=0.1, vmax=50), shading='auto', zorder=5)
            plt.colorbar(im, label="Niederschlagssumme in mm", shrink=0.4)
            
        # Geopotentialhöhe
        elif "Geopot" in sel_param:
            val = (data / 9.80665) / 10 if data.max() > 10000 else data / 10
            im = ax.pcolormesh(lons, lats, val, cmap='nipy_spectral', shading='auto', zorder=5)
            plt.colorbar(im, label="Geopotential in gpdm", shrink=0.4)
            
        # Windböen
        elif "Windböen" in sel_param:
            im = ax.pcolormesh(lons, lats, data * 3.6, cmap=cmap_wind, norm=mcolors.Normalize(vmin=0, vmax=150), shading='auto', zorder=5)
            plt.colorbar(im, label="Windböen in km/h", shrink=0.4, pad=0.02)
            
        # Signifikantes Wetter (DWD Mapping)
        elif "Signifikantes Wetter" in sel_param:
            grid = np.zeros_like(data)
            for i, (l, (c, codes)) in enumerate(WW_LEGEND_DATA.items(), 1):
                for code in codes: 
                    grid[data == code] = i
                    
            ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, shading='nearest', zorder=5)
            patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND_DATA.items()]
            ax.legend(handles=patches, loc='lower left', title="Wetter-Klassifikation", fontsize='6', title_fontsize='7', framealpha=0.9).set_zorder(25)

        # ----------------------------------------------------------------------
        # ISOBAREN (LUFTDRUCK-LINIEN)
        # ----------------------------------------------------------------------
        if iso_data is not None:
            p_hpa = iso_data / 100 if iso_data.max() > 5000 else iso_data
            if ilons.ndim == 1: 
                ilons, ilats = np.meshgrid(ilons, ilats)
            cs = ax.contour(ilons, ilats, p_hpa, colors='black', linewidths=0.7, levels=np.arange(940, 1060, 4), zorder=20)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

        # ----------------------------------------------------------------------
        # DATEN-STEMPEL (HEADER)
        # ----------------------------------------------------------------------
        v_dt = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)
        info_txt = f"Modell: {sel_model}\nParameter: {sel_param}\nTermin: {v_dt.strftime('%d.%m.%Y %H:00')} UTC\nModell-Lauf: {run_id[-2:]}Z"
        ax.text(0.02, 0.98, info_txt, transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='gray'), zorder=30)

        # Plot in Streamlit ausgeben
        st.pyplot(fig)
        cleanup_temp_files()
        
    else:
        st.error(f"Schwerer Datenfehler: {sel_model} liefert für '{sel_param}' (Stunde +{sel_hour}) aktuell keine bereinigten GRIB-Daten aus.")
else:
    st.info("ℹ️ Wähle links deine Parameter (Tipp: ICON-D2 mit CAPE oder Radar für Gewitter!) und klicke auf Karte generieren.")
