import streamlit as st
import xarray as xr
import requests, bz2, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
import numpy as np

# --- SEITEN-KONFIGURATION ---
st.set_page_config(page_title="WarnwetterBB | Profi", layout="wide")

# --- DEFINITION DER WETTER-CODES & FARBEN ---
WW_LEGEND = {
    "Nebel": ("#FFFF00", range(40, 50)),
    "Regen leicht": ("#90EE90", [50, 51, 60, 80]),
    "Regen mäßig": ("#00FF00", [53, 61, 81]),
    "Regen stark": ("#006400", [55, 63, 65, 82]),
    "Schneeregen": ("#FFA500", [68, 69, 83, 84]),
    "Schnee leicht": ("#ADD8E6", [70, 71, 85]),
    "Schnee mäßig": ("#0000FF", [73, 86]),
    "Schnee stark": ("#00008B", [75, 87]),
    "gefr. Regen": ("#FF0000", [66, 67]),
    "Gewitter": ("#800080", [95, 96, 97, 99])
}

WIND_LEVELS = [0, 10, 20, 30, 40, 50, 75, 100, 125, 150]
WIND_COLORS = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']

# --- AUSWAHLMENÜ (SIDEBAR) ---
with st.sidebar:
    st.header("🔍 Steuerung")
    sel_region = st.selectbox("Region", ["Deutschland", "Brandenburg/Berlin", "Europa"])
    sel_model = st.selectbox("Modell", ["ICON-D2 (DWD)"])
    sel_param = st.selectbox("Parameter", ["Signifikantes Wetter", "Windböen (km/h)"])
    # Geändert: Slider für 1h Schritte
    sel_hour = st.slider("Zeitschritt (Vorhersage +h)", min_value=1, max_value=48, value=1, step=1)
    st.markdown("---")
    st.write("Die App prüft alle 30 Min. auf neue Daten.")

# --- DATEN-DOWNLOAD LOGIK ---
@st.cache_data(ttl=1800) # Cache wird alle 30 Min. ungültig
def fetch_dwd_data(param_key, hr):
    for offset in [3, 4, 6, 9]:
        now = datetime.now(timezone.utc) - timedelta(hours=offset)
        run_h = (now.hour // 3) * 3
        dt_obj = now.replace(hour=run_h, minute=0, second=0, microsecond=0)
        dt_str = dt_obj.strftime("%Y%m%d%H")
        
        url = f"https://opendata.dwd.de/weather/nwp/icon-d2/grib/{run_h:02d}/{param_key}/icon-d2_germany_regular-lat-lon_single-level_{dt_str}_{hr:03d}_2d_{param_key}.grib2.bz2"
        
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with bz2.open(requests.get(url, stream=True).raw) as f_in:
                    with open(f"{param_key}.grib", "wb") as f_out: f_out.write(f_in.read())
                ds = xr.open_dataset(f"{param_key}.grib", engine='cfgrib')
                return ds, dt_obj
        except: continue
    return None, None

# --- HAUPTTEIL ---
p_map = {"Signifikantes Wetter": "ww", "Windböen (km/h)": "vmax_10m"}
ds_cl, run_dt = fetch_dwd_data("clct", sel_hour) # Bewölkung im Hintergrund
ds_main, _ = fetch_dwd_data(p_map[sel_param], sel_hour)

if ds_cl and ds_main:
    # Berechnung des Vorhersagezeitpunkts
    valid_time = run_dt + timedelta(hours=sel_hour)
    
    # Geändert: Schmaleres figsize (7x9 statt 10x11)
    fig, ax = plt.subplots(figsize=(7, 9), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Regionen-Zoom
    extents = {
        "Deutschland": [5.8, 15.2, 47.2, 55.1],
        "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6],
        "Europa": [-10, 30, 35, 65]
    }
    ax.set_extent(extents[sel_region])
    
    # Geändert: Wolken-alpha erhöht (0.6 statt 0.3)
    ax.pcolormesh(ds_cl.longitude, ds_cl.latitude, ds_cl[list(ds_cl.data_vars)[0]], cmap='Greys', alpha=0.6, shading='auto', zorder=2)
    
    # Daten-Layer
    vals = ds_main[list(ds_main.data_vars)[0]].values
    if sel_param == "Windböen (km/h)":
        im = ax.pcolormesh(ds_main.longitude, ds_main.latitude, vals * 3.6, 
                           cmap=mcolors.ListedColormap(W_COLORS), 
                           norm=mcolors.BoundaryNorm(WIND_LEVELS, ncolors=len(W_COLORS)), 
                           alpha=0.7, shading='auto', zorder=5)
        plt.colorbar(im, label="km/h", shrink=0.6, pad=0.02)
    else:
        # Signifikantes Wetter Mapping
        c_grid = np.zeros_like(vals)
        color_list = ['#FFFFFF00']
        for i, (label, (color, codes)) in enumerate(WW_LEGEND.items(), 1):
            for code in codes:
                c_grid[vals == code] = i
            color_list.append(color)
        
        ax.pcolormesh(ds_main.longitude, ds_main.latitude, c_grid, 
                      cmap=mcolors.ListedColormap(color_list), alpha=0.8, shading='auto', zorder=5)
        
        # Legende
        legend_patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in WW_LEGEND.items()]
        ax.legend(handles=legend_patches, loc='lower left', title="Sign. Wetter", fontsize='small', framealpha=0.8, zorder=12)

    # Grenzen
    ax.add_feature(cfeature.BORDERS, linewidth=1, zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=10)

    # Neu: Text-Infos direkt in der Karte
    info_text = f"Region: {sel_region}\nDatum: {valid_time.strftime('%d.%m.%Y %H:00')} UTC\nModell: {sel_model} (Lauf: {run_dt.strftime('%H')}Z)"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=15)
    
    st.pyplot(fig)
else:
    st.info("🔄 Suche nach aktuellsten Wetterdaten vom DWD-Server...")

