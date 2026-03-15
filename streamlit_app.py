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
st.set_page_config(page_title="WarnwetterBB | KI-Ensemble", layout="wide")

# --- 2. FARBSKALEN ---
# Deine schlagartige 10er-Temperatur-Skala
temp_colors = [
    (0.0, '#D3D3D3'), (5/60, '#FFFFFF'), (10/60, '#FFC0CB'), (15/60, '#FF00FF'),
    (20/60, '#800080'), (20.01/60, '#00008B'), (21/60, '#00008B'), (25/60, '#0000CD'),
    (29.99/60, '#ADD8E6'), (30/60, '#006400'), (35/60, '#008000'), (39/60, '#90EE90'),
    (39.99/60, '#90EE90'), (40/60, '#FFFF00'), (45/60, '#FFA500'), (50/60, '#FF0000'),
    (55/60, '#8B0000'), (60/60, '#800080')
]
cmap_temp = mcolors.LinearSegmentedColormap.from_list("custom_temp", temp_colors)

# Deine detaillierte Wetter-Skala
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

W_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']
cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind", W_COLORS, N=256)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🛰️ Modell-Konfiguration")
    sel_model = st.selectbox("Modell", ["ICON-D2", "ICON-D2-RUC", "ICON-EU", "GFS (NOAA)", "ECMWF", "WarnwetterBB-KI"])
    sel_region = st.selectbox("Region", ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"])
    
    p_opts = ["Temperatur 2m (°C)", "Windböen (km/h)", "500 hPa Geopot. Höhe", "850 hPa Temp."]
    if "ICON" in sel_model or "WarnwetterBB" in sel_model:
        p_opts.append("Signifikantes Wetter")
    sel_param = st.selectbox("Parameter", p_opts)
    
    max_h, step_h = 48, 1
    if "RUC" in sel_model: max_h = 27
    elif "EU" in sel_model or "GFS" in sel_model: max_h = 120
    elif "ECMWF" in sel_model: max_h, step_h = 144, 3
        
    sel_hour = st.slider("Stunde (+h)", step_h, max_h, step_h, step=step_h)
    show_isobars = st.checkbox("Isobaren (Luftdruck) anzeigen", value=True)
    st.markdown("---")
    generate = st.button("🚀 Karte generieren", use_container_width=True)

# --- 4. DATA-ENGINE ---
@st.cache_data(ttl=600)
def fetch_any_model(model, param, hr):
    p_map = {"Temperatur 2m (°C)": "t_2m", "Windböen (km/h)": "vmax_10m", "500 hPa Geopot. Höhe": "fi", "850 hPa Temp.": "t", "Signifikantes Wetter": "ww", "Isobaren": "pmsl"}
    key = p_map[param]
    now = datetime.now(timezone.utc)
    
    # Helfer für DWD-Downloads
    def dl_icon(m_name, run_h, dt_str, lead_h, p_key):
        reg = "europe" if "eu" in m_name else "germany"
        l_type = "pressure-level" if p_key in ["fi", "t"] else "single-level"
        f_end = f"{'500' if '500' in param else '850'}_{p_key}" if l_type == "pressure-level" else f"2d_{p_key}"
        url = f"https://opendata.dwd.de/weather/nwp/{m_name}/grib/{run_h:02d}/{p_key}/{m_name}_{reg}_regular-lat-lon_{l_type}_{dt_str}_{lead_h:03d}_{f_end}.grib2.bz2"
        try:
            r = requests.get(url, timeout=7)
            if r.status_code == 200:
                with bz2.open(io.BytesIO(r.content)) as f:
                    ds = xr.open_dataset(f, engine='cfgrib')
                    v = list(ds.data_vars)[0]
                    return ds[v].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze(), ds.longitude.values, ds.latitude.values, dt_str
        except: return None, None, None, None
        return None, None, None, None

    # --- ENSEMBLE LOGIK (WarnwetterBB-KI) ---
    if "WarnwetterBB" in model:
        all_runs = []
        lons, lats, run_id = None, None, None
        target_dt = now + timedelta(hours=hr)
        found, off = 0, 1
        
        while found < 4 and off < 20:
            t_check = now - timedelta(hours=off)
            run = (t_check.hour // 3) * 3
            dt_s = t_check.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
            
            # Lead-Time exakt berechnen
            run_dt = datetime.strptime(dt_s, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            lead = int((target_dt - run_dt).total_seconds() // 3600)
            
            if lead > 0:
                d, lo, la, rid = dl_icon("icon-d2", run, dt_s, lead, key)
                if d is not None:
                    all_runs.append(d)
                    if found == 0: lons, lats, run_id = lo, la, rid
                    found += 1
            off += 1
            
        if len(all_runs) == 4:
            avg = np.mean(all_runs, axis=0)
            final = avg + (all_runs[0] - avg) * 0.25 # Deine KI-Trend-Formel
            if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
            return final, lons, lats, run_id
        return None, None, None, None

    # --- STANDARD MODELLE ---
    if "ICON" in model:
        is_ruc = "RUC" in model
        m_path = "icon-d2-ruc" if is_ruc else ("icon-d2" if "D2" in model else "icon-eu")
        step = 1 if is_ruc else 3
        for o in range(1, 15):
            t = now - timedelta(hours=o)
            r = t.hour if is_ruc else (t.hour // 3) * 3
            dt_s = t.replace(hour=r, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
            d, lo, la, rid = dl_icon(m_path, r, dt_s, hr, key)
            if d is not None:
                if lo.ndim == 1: lo, la = np.meshgrid(lo, la)
                return d, lo, la, rid
                
    elif "GFS" in model:
        for off in [3, 6, 9, 12]:
            t = now - timedelta(hours=off)
            run = (t.hour // 6) * 6
            dt_s = t.strftime("%Y%m%d")
            g_f = f"&var_{'TMP' if 'Temp' in param else 'HGT' if 'Geopot' in param else 'GUST' if 'Wind' in param else 'PRMSL'}=on"
            l_f = f"&lev_{'2_m_above_ground' if '2m' in param else '850_mb' if '850' in param else '500_mb' if '500' in param else 'surface' if 'Wind' in param else 'mean_sea_level'}=on"
            url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t{run:02d}z.pgrb2.0p25.f{hr:03d}{g_f}{l_f}&subregion=&leftlon=-20&rightlon=45&toplat=75&bottomlat=30&dir=%2Fgfs.{dt_s}%2F{run:02d}%2Fatmos"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    ds = xr.open_dataset(io.BytesIO(r.content), engine='cfgrib')
                    d = ds[list(ds.data_vars)[0]].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze()
                    lo, la = np.meshgrid(ds.longitude.values, ds.latitude.values)
                    return d, lo, la, f"{dt_s}{run:02d}"
            except: continue

    elif "ECMWF" in model:
        for off in [0, 12, 24]:
            t = now - timedelta(hours=off)
            run = (t.hour // 12) * 12
            dt_s = t.strftime("%Y%m%d")
            url = f"https://data.ecmwf.int/forecasts/{dt_s}/{run:02d}z/ifs/0p4-beta/oper/{dt_s}{run:02d}0000-{hr}h-oper-fc.grib2"
            try:
                r = requests.get(url, timeout=25)
                if r.status_code == 200:
                    e_k = {"t_2m":"2t", "vmax_10m":"10fg", "fi":"z", "t":"t", "pmsl":"msl"}
                    ds = xr.open_dataset(io.BytesIO(r.content), engine='cfgrib', filter_by_keys={'shortName': e_keys.get(key, "2t")})
                    d = ds[list(ds.data_vars)[0]].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze()
                    lo, la = np.meshgrid(ds.longitude.values, ds.latitude.values)
                    return d, lo, la, f"{dt_s}{run:02d}"
            except: continue
    return None, None, None, None

# --- 5. HAUPTTEIL ---
if generate:
    with st.spinner(f"📡 Berechne {sel_model}..."):
        data, lons, lats, run_id = fetch_any_model(sel_model, sel_param, sel_hour)
        iso_data, ilons, ilats, _ = fetch_any_model(sel_model, "Isobaren", sel_hour) if show_isobars else (None, None, None, None)

    if data is not None:
        fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=120)
        ext = {"Deutschland": [5.8, 15.2, 47.2, 55.1], "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6], "Mitteleuropa (DE, PL)": [4.0, 25.0, 45.0, 56.0], "Alpenraum": [5.5, 17.0, 44.0, 49.5], "Europa": [-12, 40, 34, 66]}
        ax.set_extent(ext[sel_region])

        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=12)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='black', zorder=12)
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(states, linewidth=0.5, edgecolor='black', zorder=12)

        if "Temperatur" in sel_param or "850 hPa Temp." in sel_param:
            val_c = data - 273.15 if data.max() > 100 else data
            im = ax.pcolormesh(lons, lats, val_c, cmap=cmap_temp, norm=mcolors.Normalize(vmin=-30, vmax=30), shading='auto', zorder=5)
            plt.colorbar(im, label="°C", shrink=0.4, pad=0.02, ticks=np.arange(-30, 31, 10))
        elif "Geopot" in sel_param:
            val = (data / 9.81) / 10 if data.max() > 1000 else data / 10
            im = ax.pcolormesh(lons, lats, val, cmap='nipy_spectral', shading='auto', zorder=5)
            plt.colorbar(im, label="gpdm", shrink=0.4)
        elif "Windböen" in sel_param:
            im = ax.pcolormesh(lons, lats, data * 3.6, cmap=cmap_wind, norm=mcolors.Normalize(vmin=0, vmax=150), shading='auto', zorder=5)
            plt.colorbar(im, label="km/h", shrink=0.4, pad=0.02)
        elif "Signifikantes Wetter" in sel_param:
            grid = np.zeros_like(data)
            for i, (l, (c, codes)) in enumerate(WW_LEGEND_DATA.items(), 1):
                for code in codes: grid[np.isclose(data, code, atol=0.4)] = i
            ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, shading='nearest', zorder=5)
            patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND_DATA.items()]
            leg = ax.legend(handles=patches, loc='lower left', title="Wetter", fontsize='6', title_fontsize='7', framealpha=0.9)
            leg.set_zorder(25)

        if iso_data is not None:
            p_hpa = iso_data / 100 if iso_data.max() > 5000 else iso_data
            if ilons.ndim == 1: ilons, ilats = np.meshgrid(ilons, ilats)
            cs = ax.contour(ilons, ilats, p_hpa, colors='black', linewidths=0.7, levels=np.arange(940, 1060, 4), zorder=20)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

        info = f"{sel_model} | {sel_param}\nTermin: {(datetime.strptime(run_id, '%Y%m%d%H').replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)).strftime('%d.%m. %H:00')} UTC\nLauf: {run_id[-2:]}Z"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2', edgecolor='none'), zorder=30)
        st.pyplot(fig)
    else:
        st.error(f"Daten für {sel_model} konnten nicht kombiniert werden. Versuche es in 5 Min noch einmal.")
else:
    st.info("🚀 Konfiguration wählen und Starten.")
