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

# --- 2. FARBSKALEN ---
# Temperatur (Deine schlagartige 10er Skala)
temp_colors = [
    (0.0, '#D3D3D3'),       # -30: hellgrau
    (5/60, '#FFFFFF'),      # -25: weiß
    (10/60, '#FFC0CB'),     # -20: rosa
    (15/60, '#FF00FF'),     # -15: magenta
    (20/60, '#800080'),     # -10: lila
    (20.01/60, '#00008B'),  # SCHLAGARTIG zu dunkelblau
    (21/60, '#00008B'),     # -9: dunkelblau
    (25/60, '#0000CD'),     # -5: dunkles blau
    (29.99/60, '#ADD8E6'),  # -0: hellblau
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

# Wind
W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']
cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind", W_COLORS, N=256)

# --- NEU: DEINE DETAILLIERTE WETTER-SKALA ---
# Mapping der DWD/WMO Wettercodes auf deine exakten Farbwünsche
WW_LEGEND_DATA = {
    "Nebel": ("#FFFF00", list(range(40, 50))),                       # gelb
    "Regen leicht": ("#00FF00", [50, 51, 58, 60, 80]),               # grün
    "Regen mäßig": ("#228B22", [53, 61, 62, 81]),                    # dunkleres grün (ForestGreen)
    "Regen stark": ("#006400", [54, 55, 63, 64, 65, 82]),            # dunkelgrün
    "gefr. Regen leicht": ("#FF7F7F", [56, 66]),                     # hellrot
    "gefr. Regen mäßig/stark": ("#FF0000", [57, 67]),                # rot
    "Schneeregen leicht": ("#FFB347", [68, 83]),                     # hellorange
    "Schneeregen mäßig/stark": ("#FFA500", [69, 84]),                # orange
    "Schnee leicht": ("#87CEEB", [70, 71, 85]),                      # himmelblau
    "Schnee mäßig": ("#0000FF", [72, 73, 86]),                       # blau
    "Schnee stark": ("#00008B", [74, 75, 76, 77, 78, 79, 87, 88]),   # dunkleres blau
    "Gewitter leicht": ("#FF00FF", [95]),                            # magenta
    "Gewitter mäßig/stark": ("#800080", [96, 97, 99])                # lila
}
# Farbe für "Kein signifikantes Wetter" ist transparent (#FFFFFF00), danach folgen deine Farben
cmap_ww = mcolors.ListedColormap(['#FFFFFF00'] + [c for l, (c, codes) in WW_LEGEND_DATA.items()])


# --- 3. SIDEBAR (DYNAMISCH) ---
with st.sidebar:
    st.header("🛰️ Konfiguration")
    sel_model = st.selectbox("Modell", ["ICON-D2", "ICON-D2-RUC", "ICON-EU", "GFS (NOAA)", "ECMWF"])
    sel_region = st.selectbox("Region", ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"])
    
    p_opts = ["Temperatur 2m (°C)", "Windböen (km/h)", "500 hPa Geopot. Höhe", "850 hPa Temp."]
    if "ICON-D2" in sel_model: p_opts.append("Signifikantes Wetter")
    sel_param = st.selectbox("Parameter", p_opts)
    
    max_h, step_h = 48, 1
    if "ICON-D2-RUC" in sel_model: max_h, step_h = 27, 1
    elif "ICON-EU" in sel_model: max_h, step_h = 120, 1
    elif "GFS" in sel_model: max_h, step_h = 120, 3
    elif "ECMWF" in sel_model: max_h, step_h = 144, 3
        
    sel_hour = st.slider("Stunde (+h)", step_h, max_h, step_h, step=step_h)
    show_isobars = st.checkbox("Isobaren (Luftdruck) anzeigen", value=True)
    
    st.markdown("---")
    generate = st.button("🚀 Karte generieren", use_container_width=True)

# --- 4. ROBUSTER MULTI-FETCH ---
@st.cache_data(ttl=600)
def fetch_any_model(model, param, hr):
    p_map = {"Temperatur 2m (°C)": "t_2m", "Windböen (km/h)": "vmax_10m", "500 hPa Geopot. Höhe": "fi", "850 hPa Temp.": "t", "Signifikantes Wetter": "ww", "Isobaren": "pmsl"}
    key = p_map[param]
    now = datetime.now(timezone.utc)
    headers = {'User-Agent': 'Mozilla/5.0'}

    if "ICON" in model:
        is_ruc = "RUC" in model
        m_dir = "icon-d2-ruc" if is_ruc else ("icon-d2" if "D2" in model else "icon-eu")
        reg_str = "europe" if "icon-eu" in m_dir else "germany"
        step = 1 if is_ruc else 3
        
        for off in range(1, 10):
            t = now - timedelta(hours=off)
            run = t.hour if is_ruc else (t.hour // 3) * 3
            dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
            
            l_type = "pressure-level" if key in ["fi", "t"] else "single-level"
            file_end = f"{'500' if '500' in param else '850'}_{key}" if l_type == "pressure-level" else f"2d_{key}"
            
            url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{m_dir}_{reg_str}_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{file_end}.grib2.bz2"
            
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with bz2.open(io.BytesIO(r.content)) as f:
                        with open("temp.grib", "wb") as out: out.write(f.read())
                    ds = xr.open_dataset("temp.grib", engine='cfgrib')
                    
                    var = list(ds.data_vars)[0]
                    ds_var = ds[var]
                    
                    if 'isobaricInhPa' in ds_var.dims:
                        lvl_val = 500 if "500" in param else 850
                        data_flat = ds_var.sel(isobaricInhPa=lvl_val)
                    else: data_flat = ds_var
                    
                    drop_dims = {d: 0 for d in ['step', 'height', 'time', 'valid_time', 'surface'] if d in data_flat.dims}
                    data = data_flat.isel(**drop_dims).values.squeeze()
                    
                    lons, lats = ds.longitude.values, ds.latitude.values
                    if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                    return data, lons, lats, dt_s
            except: continue

    elif "GFS" in model:
        for off in [3, 6, 9, 12]:
            t = now - timedelta(hours=off)
            run = (t.hour // 6) * 6
            dt_s = t.strftime("%Y%m%d")
            url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t{run:02d}z.pgrb2.0p25.f{hr:03d}&lev_2_m_above_ground=on&lev_10_m_above_ground=on&lev_500_mb=on&lev_850_mb=on&lev_mean_sea_level=on&var_TMP=on&var_HGT=on&var_GUST=on&var_PRMSL=on&subregion=&leftlon=-20&rightlon=45&toplat=75&bottomlat=30&dir=%2Fgfs.{dt_s}%2F{run:02d}%2Fatmos"
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200:
                    with open("temp_gfs.grib", "wb") as out: out.write(r.content)
                    ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                    var = list(ds.data_vars)[0]
                    ds_var = ds[var]
                    
                    if 'isobaricInhPa' in ds_var.dims:
                        lvl_val = 500 if "500" in param else 850
                        data_flat = ds_var.sel(isobaricInhPa=lvl_val)
                    else: data_flat = ds_var
                    
                    drop_dims = {d: 0 for d in ['step', 'height', 'time', 'valid_time', 'meanSea'] if d in data_flat.dims}
                    data = data_flat.isel(**drop_dims).values.squeeze()
                    
                    lons, lats = ds.longitude.values, ds.latitude.values
                    if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                    return data, lons, lats, f"{dt_s}{run:02d}"
            except: continue

    elif "ECMWF" in model:
        for off in [0, 12, 24]:
            t = now - timedelta(hours=off)
            run = (t.hour // 12) * 12
            dt_s = t.strftime("%Y%m%d")
            url = f"https://data.ecmwf.int/forecasts/{dt_s}/{run:02d}z/ifs/0p4-beta/oper/{dt_s}{run:02d}0000-{hr}h-oper-fc.grib2"
            try:
                r = requests.get(url, headers=headers, timeout=25)
                if r.status_code == 200:
                    with open("temp_ecmwf.grib", "wb") as out: out.write(r.content)
                    ecmwf_keys = {"t_2m": "2t", "vmax_10m": "10fg", "fi": "z", "t": "t", "pmsl": "msl"}
                    e_key = ecmwf_keys.get(key, "2t")
                    ds = xr.open_dataset("temp_ecmwf.grib", engine='cfgrib', filter_by_keys={'shortName': e_key})
                    var = list(ds.data_vars)[0]
                    ds_var = ds[var]
                    
                    if 'isobaricInhPa' in ds_var.dims:
                        lvl_val = 500 if "500" in param else 850
                        data_flat = ds_var.sel(isobaricInhPa=lvl_val)
                    else: data_flat = ds_var
                    
                    drop_dims = {d: 0 for d in ['step', 'height', 'time', 'valid_time', 'surface', 'meanSea'] if d in data_flat.dims}
                    data = data_flat.isel(**drop_dims).values.squeeze()
                    
                    lons, lats = ds.longitude.values, ds.latitude.values
                    if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                    return data, lons, lats, f"{dt_s}{run:02d}"
            except: continue

    return None, None, None, None

# --- 5. KARTEN-GENERIERUNG ---
if generate:
    with st.spinner(f"📡 Lade Daten für {sel_model}..."):
        data, lons, lats, run_id = fetch_any_model(sel_model, sel_param, sel_hour)
        iso_data = None
        if show_isobars:
            iso_data, ilons, ilats, _ = fetch_any_model(sel_model, "Isobaren", sel_hour)

    if data is not None:
        fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=120)
        
        ext = {"Deutschland": [5.8, 15.2, 47.2, 55.1], "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6], "Mitteleuropa (DE, PL)": [4.0, 25.0, 45.0, 56.0], "Alpenraum": [5.5, 17.0, 44.0, 49.5], "Europa": [-12, 40, 34, 66]}
        ax.set_extent(ext[sel_region])

        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=12)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='black', zorder=12)

        # PLOTTING LOGIK
        if "Temperatur" in sel_param or "850 hPa Temp." in sel_param:
            val_c = data - 273.15 if data.max() > 100 else data
            im = ax.pcolormesh(lons, lats, val_c, cmap=cmap_temp, norm=mcolors.Normalize(vmin=-30, vmax=30), shading='auto', zorder=5)
            plt.colorbar(im, label="°C", shrink=0.4, pad=0.02, ticks=np.arange(-30, 31, 10))

        elif "Geopot" in sel_param:
            val = (data / 9.80665) / 10 if data.max() > 10000 else data / 10
            im = ax.pcolormesh(lons, lats, val, cmap='nipy_spectral', shading='auto', zorder=5)
            plt.colorbar(im, label="gpdm", shrink=0.4)
            
        elif "Windböen" in sel_param:
            im = ax.pcolormesh(lons, lats, data * 3.6, cmap=cmap_wind, norm=mcolors.Normalize(vmin=0, vmax=150), shading='auto', zorder=5)
            plt.colorbar(im, label="km/h", shrink=0.4, pad=0.02)
            
        elif "Signifikantes Wetter" in sel_param:
            # MATRIX FÜR DAS WETTER BAUEN (Ordnet die DWD Codes deinen Farben zu)
            grid = np.zeros_like(data)
            for i, (l, (c, codes)) in enumerate(WW_LEGEND_DATA.items(), 1):
                for code in codes:
                    grid[data == code] = i
                    
            # shading='nearest' sorgt dafür, dass die Wetter-Blöcke scharfkantig bleiben (kein Verwischen)
            ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, shading='nearest', zorder=5)
            
            # Legende extrem kompakt generieren, damit sie nicht die Karte verdeckt
            patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND_DATA.items()]
            leg = ax.legend(handles=patches, loc='lower left', title="Signifikantes Wetter", fontsize='6', title_fontsize='7', framealpha=0.9)
            leg.set_zorder(25)

        # ISOBAREN LAYER
        if iso_data is not None:
            p_hpa = iso_data / 100 if iso_data.max() > 5000 else iso_data
            if ilons.ndim == 1: ilons, ilats = np.meshgrid(ilons, ilats)
            cs = ax.contour(ilons, ilats, p_hpa, colors='black', linewidths=0.7, levels=np.arange(940, 1060, 4), zorder=20)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

        # HEADER INFO
        valid = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)
        info = f"{sel_model} | {sel_param}\nTermin: {valid.strftime('%d.%m. %H:00')} UTC\nLauf: {run_id[-2:]}Z"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2', edgecolor='none'), zorder=30)

        st.pyplot(fig)
    else:
        st.error(f"Modell {sel_model} ist für diesen Zeitschritt (+{sel_hour}h) gerade nicht erreichbar. Versuche eine andere Stunde oder ein anderes Modell.")
else:
    st.info("Einstellungen wählen und auf 'Karte generieren' klicken.")
