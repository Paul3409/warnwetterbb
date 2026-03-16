"""
=========================================================================================
WARNWETTER BB - PROFESSIONAL METEOROLOGICAL WORKSTATION
=========================================================================================
Version: Stable Release (Rollback)
Fokus: Maximale Ausfallsicherheit, GRIB2 Parsing, RainViewer API, Google Maps Satellite.

Architektur-Highlights:
- Vollständige Trennung von Datenbeschaffung (Fetcher) und Visualisierung (Plotter).
- Dedizierte Render-Funktionen für JEDEN meteorologischen Parameter.
- Speicherschonendes Garbage-Collection-System für Streamlit Cloud.
- Dynamische Cartopy Tile-Generierung für hochauflösende Satellitenbilder.
=========================================================================================
"""

import streamlit as st
import xarray as xr
import requests
import bz2
import os
import io
import time
import logging
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from typing import Tuple, List, Dict, Optional, Any
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# ==============================================================================
# 1. SYSTEM-SETUP & LOGGING
# ==============================================================================
# Logging konfigurieren, damit Fehler in der Streamlit-Konsole abgefangen werden
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WarnwetterBB_Core")

# Streamlit Autorefresh (Optional für Radar-Live-Modus)
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None
    logger.warning("streamlit_autorefresh nicht installiert. Auto-Update deaktiviert.")

# Streamlit Page Configuration (Muss als Erstes aufgerufen werden)
st.set_page_config(
    page_title="WarnwetterBB | Pro-Zentrale", 
    page_icon="🛰️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Konstanten für das Zeitzonen-Management
LOCAL_TZ = ZoneInfo("Europe/Berlin")
WOCHENTAGE = [
    "Mo", 
    "Di", 
    "Mi", 
    "Do", 
    "Fr", 
    "Sa", 
    "So"
]

# ==============================================================================
# 2. SYSTEM UTILITIES (SPEICHERVERWALTUNG)
# ==============================================================================
class SystemManager:
    """Verwaltet serverseitige Ressourcen und temporäre Dateien."""
    
    @staticmethod
    def cleanup_temp_files(directory: str = ".") -> None:
        """
        Sucht und löscht GRIB-Reste und Index-Dateien.
        Verhindert das Vollmüllen des Containers nach vielen Kartengenerierungen.
        """
        temp_extensions = [
            ".grib", 
            ".grib2", 
            ".bz2", 
            ".idx", 
            "temp_gfs.grib"
        ]
        
        for filename in os.listdir(directory):
            if any(filename.endswith(ext) for ext in temp_extensions) and "temp" in filename:
                filepath = os.path.join(directory, filename)
                try:
                    os.remove(filepath)
                    logger.debug(f"Temporäre Datei gelöscht: {filename}")
                except Exception as e:
                    logger.debug(f"Datei im Zugriff, übersprungen: {filename}")


# ==============================================================================
# 3. KARTEN-HINTERGRÜNDE (CARTOPY TILES)
# ==============================================================================
class GoogleSatelliteTiles(cimgt.GoogleWTS):
    """
    Greift direkt auf Google Maps Satellite Tiles zu.
    Garantiert fotorealistische, fehlerfreie Kartenhintergründe.
    """
    def _image_url(self, tile: Tuple[int, int, int]) -> str:
        x, y, z = tile
        # 's' Parameter in der URL steht für Satellite
        return f'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

class RainViewerTiles(cimgt.GoogleWTS):
    """
    Greift auf das hochauflösende Mosaik der RainViewer Radar API zu.
    Wird nahtlos über die GoogleSatelliteTiles gelegt.
    """
    def __init__(self, host: str, path: str):
        self.host = host
        self.path = path
        super().__init__()

    def _image_url(self, tile: Tuple[int, int, int]) -> str:
        x, y, z = tile
        # Parameter: Farbschema 2 (Meteo), Smooth=1, Snow=1
        return f"{self.host}{self.path}/256/{z}/{x}/{y}/2/1_1.png"


# ==============================================================================
# 4. METEOROLOGISCHE FARBSKALEN (COLORMAP REGISTRY)
# ==============================================================================
class ColormapRegistry:
    """
    Zentrale Definition aller meteorologischen Farbskalen.
    Ausführlich definiert für maximale Präzision bei der Darstellung.
    """
    
    @staticmethod
    def get_temperature() -> mcolors.LinearSegmentedColormap:
        colors = [
            (0.0, '#D3D3D3'), 
            (5/60, '#FFFFFF'), 
            (10/60, '#FFC0CB'), 
            (15/60, '#FF00FF'),
            (20/60, '#800080'), 
            (20.01/60, '#00008B'), 
            (25/60, '#0000CD'), 
            (29.99/60, '#ADD8E6'),
            (30/60, '#006400'), 
            (35/60, '#008000'), 
            (39/60, '#90EE90'), 
            (39.99/60, '#90EE90'),
            (40/60, '#FFFF00'), 
            (45/60, '#FFA500'), 
            (50/60, '#FF0000'), 
            (55/60, '#8B0000'), 
            (60/60, '#800080')
        ]
        return mcolors.LinearSegmentedColormap.from_list("custom_temp", colors)

    @staticmethod
    def get_precipitation() -> Tuple[mcolors.LinearSegmentedColormap, mcolors.Normalize]:
        precip_values = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3, 4, 5, 8, 12, 15, 20, 30, 40, 50]
        precip_colors = [
            '#FFFFFF', '#87CEEB', '#1E90FF', '#191970', '#006400', '#32CD32', '#FFFF00', 
            '#FFA500', '#FF0000', '#8B0000', '#800000', '#4B0082', '#800080', '#9400D3', 
            '#7B68EE', '#FFFFFF'  
        ]
        vmax = 50.0
        anchors = [v / vmax for v in precip_values]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_precip", list(zip(anchors, precip_colors)))
        # Mache Null-Werte transparent
        cmap.set_bad(color='black', alpha=0.0)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        return cmap, norm

    @staticmethod
    def get_radar() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
        colors = [
            '#FFFFFF', '#B0E0E6', '#00BFFF', '#0000FF', '#00FF00', '#32CD32', '#008000', 
            '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#800080', '#4B0082', '#E6E6FA'
        ]
        cmap = mcolors.ListedColormap(colors)
        # WICHTIG: Setzt alle leeren (NaN) Radar-Werte auf transparent, 
        # damit das Google-Satellitenbild fehlerfrei durchscheint!
        cmap.set_bad(color='black', alpha=0.0)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm

    @staticmethod
    def get_cape() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        levels = [0, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]
        colors = [
            '#006400', '#2E8B57', '#ADFF2F', '#FFFF00', '#FFB347', '#FFA500', 
            '#FF4500', '#FF0000', '#8B0000', '#800080', '#FF00FF', '#FFFFFF', '#808080', '#404040'
        ]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm
        
    @staticmethod
    def get_cloud_base() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        levels = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 8000]
        colors = [
            '#FF00FF', '#FF0000', '#FFA500', '#FFFF00', '#ADFF2F', '#32CD32', '#00BFFF', 
            '#1E90FF', '#0000FF', '#A9A9A9', '#FFFFFF'
        ]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm

    @staticmethod
    def get_wind() -> mcolors.LinearSegmentedColormap:
        colors = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']
        return mcolors.LinearSegmentedColormap.from_list("custom_wind", colors, N=256)

    @staticmethod
    def get_generic_cmap(name: str) -> mcolors.LinearSegmentedColormap:
        """Hilfsfunktion für diverse Standard-Farbskalen."""
        mapping = {
            "cin": ['#FFFFFF', '#ADD8E6', '#0000FF', '#00008B', '#000000'],
            "clouds": ['#1E90FF', '#87CEEB', '#D3D3D3', '#FFFFFF'],
            "relhum": ['#8B4513', '#F4A460', '#FFFFE0', '#90EE90', '#008000', '#0000FF'],
            "snow": ['#CCFFCC', '#FFFFFF', '#ADD8E6', '#0000FF', '#800080'],
            "vis": ['#FFFFFF', '#D3D3D3', '#87CEEB', '#1E90FF'],
            "heli": ['#FFFFFF', '#00FF00', '#FFFF00', '#FF0000', '#800080', '#000000'],
            "lifted": ['#FF00FF', '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#0000FF'],
            "sun": ['#808080', '#FFD700', '#FFA500', '#FF8C00']
        }
        return mcolors.LinearSegmentedColormap.from_list(f"custom_{name}", mapping.get(name, ['#FFFFFF', '#000000']), N=256)

    @staticmethod
    def get_significant_weather() -> Tuple[mcolors.ListedColormap, Dict[str, Tuple[str, List[int]]]]:
        """Definition der DWD ww-Keys für die Signifikantes-Wetter-Karte."""
        legend_data = {
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
        # Die erste Farbe ist zu 100% transparent für alle Werte ohne besonderes Wetter
        cmap = mcolors.ListedColormap(['#FFFFFF00'] + [color for label, (color, codes) in legend_data.items()])
        return cmap, legend_data


# ==============================================================================
# 5. MODELL-ROUTING & KONFIGURATION
# ==============================================================================
# Fest definierte Regionen für sauberes Clipping
EXTENTS = {
    "Deutschland": [5.8, 15.2, 47.2, 55.1], 
    "Brandenburg/Berlin": [11.2, 14.8, 51.2, 53.6], 
    "Mitteleuropa (DE, PL)": [4.0, 25.0, 45.0, 56.0], 
    "Alpenraum": [5.5, 17.0, 44.0, 49.5], 
    "Europa": [-12, 40, 34, 66]
}

# Master-Routing Dictionary (DWD Radar entfernt, RainViewer bleibt exklusiv)
MODEL_ROUTER = {
    "RainViewer Echtzeit-Radar": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Europa"],
        "params": ["Echtzeit-Radar (Reflektivität)"]
    },
    "ICON-D2": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum"],
        "params": [
            "Temperatur 2m (°C)", 
            "Taupunkt 2m (°C)", 
            "Windböen (km/h)", 
            "Bodendruck (hPa)", 
            "500 hPa Geopot. Höhe", 
            "850 hPa Temp.", 
            "Niederschlag (mm)", 
            "CAPE (J/kg)", 
            "CIN (J/kg)", 
            "Gesamtbedeckung (%)", 
            "Rel. Feuchte 700 hPa (%)", 
            "Schneehöhe (cm)",
            "Signifikantes Wetter", 
            "Sichtweite (m)", 
            "Wolkenuntergrenze (m)", 
            "Wolkenobergrenze (m)", 
            "Spezifische Feuchte (g/kg)", 
            "Simuliertes Radar (dBZ)", 
            "Helizität / SRH (m²/s²)", 
            "Sonnenscheindauer (Min)"
        ]
    },
    "ICON-EU": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"],
        "params": [
            "Temperatur 2m (°C)", 
            "Taupunkt 2m (°C)", 
            "Windböen (km/h)", 
            "Bodendruck (hPa)", 
            "500 hPa Geopot. Höhe", 
            "850 hPa Temp.", 
            "Niederschlag (mm)", 
            "CAPE (J/kg)", 
            "CIN (J/kg)", 
            "Gesamtbedeckung (%)", 
            "Rel. Feuchte 700 hPa (%)", 
            "Schneehöhe (cm)",
            "Signifikantes Wetter", 
            "Sichtweite (m)", 
            "Wolkenuntergrenze (m)", 
            "Wolkenobergrenze (m)"
        ]
    },
    "ICON (Global)": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"],
        "params": [
            "Temperatur 2m (°C)", 
            "Taupunkt 2m (°C)", 
            "Windböen (km/h)", 
            "Bodendruck (hPa)", 
            "500 hPa Geopot. Höhe", 
            "850 hPa Temp.", 
            "Niederschlag (mm)", 
            "CAPE (J/kg)", 
            "CIN (J/kg)", 
            "Gesamtbedeckung (%)", 
            "Rel. Feuchte 700 hPa (%)", 
            "Schneehöhe (cm)"
        ]
    },
    "GFS (NOAA)": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"],
        "params": [
            "Temperatur 2m (°C)", 
            "Taupunkt 2m (°C)", 
            "Windböen (km/h)", 
            "Bodendruck (hPa)", 
            "500 hPa Geopot. Höhe", 
            "850 hPa Temp.", 
            "Niederschlag (mm)", 
            "CAPE (J/kg)", 
            "CIN (J/kg)", 
            "Gesamtbedeckung (%)", 
            "Rel. Feuchte 700 hPa (%)", 
            "Schneehöhe (cm)",
            "0-Grad-Grenze (m)", 
            "Lifted Index (K)", 
            "Sichtweite (m)", 
            "Wolkenuntergrenze (m)"
        ]
    },
    "ECMWF": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"],
        "params": [
            "Temperatur 2m (°C)", 
            "Taupunkt 2m (°C)", 
            "Windböen (km/h)", 
            "Bodendruck (hPa)", 
            "500 hPa Geopot. Höhe", 
            "850 hPa Temp.", 
            "Niederschlag (mm)", 
            "Gesamtbedeckung (%)"
        ] 
    },
    "ECMWF-AIFS (KI-Modell)": {
        "regions": ["Deutschland", "Brandenburg/Berlin", "Mitteleuropa (DE, PL)", "Alpenraum", "Europa"],
        "params": [
            "Temperatur 2m (°C)", 
            "Windböen (km/h)", 
            "Bodendruck (hPa)", 
            "500 hPa Geopot. Höhe", 
            "850 hPa Temp.", 
            "Niederschlag (mm)"
        ]
    }
}


# ==============================================================================
# 6. DATA FETCH ENGINE (GRIB & RADAR API)
# ==============================================================================
class DataFetcher:
    """Zentrale Klasse für den Bezug sämtlicher Wetterdaten."""
    
    @staticmethod
    def estimate_latest_run(model: str, now_utc: datetime) -> datetime:
        """Kalkuliert den neuesten Modelllauf unter Berücksichtigung der Server-Delays."""
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
        return now_utc

    @staticmethod
    def fetch_rainviewer_metadata() -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """Zieht das aktuellste Radar-Mosaik von der RainViewer API."""
        logs = ["Starte RainViewer API Request..."]
        try:
            r = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=10)
            r.raise_for_status()
            data = r.json()
            host = data.get("host", "https://tilecache.rainviewer.com")
            past_radar = data.get("radar", {}).get("past", [])
            
            if past_radar:
                latest = past_radar[-1]
                logs.append(f"Erfolgreich: Pfad {latest['path']} gefunden.")
                return host, latest["path"], str(latest["time"]), logs
        except Exception as e:
            logs.append(f"RainViewer API Fehler: {e}")
        return None, None, None, logs

    @classmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_model_data(cls, model: str, param: str, hr: int, debug: bool = False) -> Tuple[Any, Any, Any, Any, List[str]]:
        """
        Haupt-Dispatcher: Entscheidet, ob Radar oder GRIB2 geladen werden muss.
        """
        if "RainViewer" in model:
            host, path, ts, logs = cls.fetch_rainviewer_metadata()
            return host, path, None, ts, logs

        # Mapping der UI-Namen auf GRIB-ShortNames
        dyn_cloud_base = "ceiling" if "D2" in model else "hbas_con"
        p_map = {
            "Temperatur 2m (°C)": "t_2m", 
            "Taupunkt 2m (°C)": "td_2m", 
            "Windböen (km/h)": "vmax_10m", 
            "Bodendruck (hPa)": "sp", 
            "500 hPa Geopot. Höhe": "fi", 
            "850 hPa Temp.": "t", 
            "Signifikantes Wetter": "ww", 
            "Isobaren": "pmsl", 
            "CAPE (J/kg)": "cape_ml", 
            "CIN (J/kg)": "cin_ml", 
            "Niederschlag (mm)": "tot_prec", 
            "Simuliertes Radar (dBZ)": "dbz_cmax",
            "0-Grad-Grenze (m)": "hgt_0c", 
            "Gesamtbedeckung (%)": "clct", 
            "Rel. Feuchte 700 hPa (%)": "relhum", 
            "Schneehöhe (cm)": "h_snow", 
            "Sichtweite (m)": "vis", 
            "Wolkenuntergrenze (m)": dyn_cloud_base,
            "Wolkenobergrenze (m)": "htop_con", 
            "Spezifische Feuchte (g/kg)": "qv",
            "Helizität / SRH (m²/s²)": "uh_max", 
            "Sonnenscheindauer (Min)": "dur_sun", 
            "Lifted Index (K)": "sli"
        }
        
        key = p_map.get(param, "t_2m")
        now = datetime.now(timezone.utc)
        debug_logs = []

        if "ICON" in model:
            is_global = "Global" in model
            if is_global: 
                m_dir, reg_str = "icon", "icon_global"
            elif "D2" in model: 
                m_dir, reg_str = "icon-d2", "icon-d2_germany"
            else: 
                m_dir, reg_str = "icon-eu", "icon-eu_europe"
            
            # Rückwärtssuche für höchste Stabilität
            for off in range(1, 18):
                t = now - timedelta(hours=off)
                if is_global: 
                    run = (t.hour // 6) * 6
                else: 
                    run = (t.hour // 3) * 3
                
                dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
                
                l_type = "pressure-level" if key in ["fi", "t", "relhum", "qv"] else "single-level"
                if l_type == "pressure-level": 
                    lvl_str = f"{'500' if '500' in param else '700' if '700' in param else '850'}_"
                else: 
                    lvl_str = "2d_"
                
                url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{reg_str}_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{lvl_str}{key}.grib2.bz2"
                debug_logs.append(url)
                
                try:
                    r = requests.get(url, timeout=5)
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
                        if lons.ndim == 1: 
                            lons, lats = np.meshgrid(lons, lats)
                        return data, lons, lats, dt_s, debug_logs
                except Exception: 
                    continue

        elif "GFS" in model:
            headers = {'User-Agent': 'Mozilla/5.0'}
            gfs_map = {
                "t_2m": "&var_TMP=on&lev_2_m_above_ground=on", 
                "td_2m": "&var_DPT=on&lev_2_m_above_ground=on",
                "vmax_10m": "&var_GUST=on&lev_surface=on", 
                "fi": "&var_HGT=on&lev_500_mb=on",
                "t": "&var_TMP=on&lev_850_mb=on", 
                "pmsl": "&var_PRMSL=on&lev_mean_sea_level=on",
                "cape_ml": "&var_CAPE=on&lev_surface=on", 
                "cin_ml": "&var_CIN=on&lev_surface=on",
                "tot_prec": "&var_APCP=on&lev_surface=on", 
                "hgt_0c": "&var_HGT=on&lev_0C_isotherm=on",
                "clct": "&var_TCDC=on&lev_entire_atmosphere=on", 
                "relhum": "&var_RH=on&lev_700_mb=on",
                "h_snow": "&var_SNOD=on&lev_surface=on", 
                "sp": "&var_PRES=on&lev_surface=on",
                "sli": "&var_4LFTX=on&lev_surface=on", 
                "vis": "&var_VIS=on&lev_surface=on",
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
                        with open("temp_gfs.grib", "wb") as f: 
                            f.write(r.content)
                        ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                        data = ds[list(ds.data_vars)[0]].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze()
                        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                        return data, lons, lats, f"{dt_s}{run:02d}", debug_logs
                except Exception: 
                    continue

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
                        with open("temp_ecmwf.grib", "wb") as f: 
                            f.write(r.content)
                        e_k = {
                            "t_2m": "2t", 
                            "td_2m": "2d", 
                            "vmax_10m": "10fg", 
                            "fi": "z", 
                            "t": "t", 
                            "pmsl": "msl", 
                            "tot_prec": "tp", 
                            "clct": "tcc", 
                            "sp": "sp"
                        }.get(key, "2t")
                        ds = xr.open_dataset("temp_ecmwf.grib", engine='cfgrib', filter_by_keys={'shortName': e_k})
                        ds_var = ds[list(ds.data_vars)[0]]
                        if 'isobaricInhPa' in ds_var.dims:
                            target_p = 500 if "500" in param else 700 if "700" in param else 850
                            ds_var = ds_var.sel(isobaricInhPa=target_p)
                        data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                        return data, lons, lats, f"{dt_s}{run:02d}", debug_logs
                except Exception: 
                    continue
                
        return None, None, None, None, debug_logs


# ==============================================================================
# 7. VISUALISIERUNG (RENDER ENGINE KLASSEN)
# ==============================================================================
class PlottingEngine:
    """
    Diese Klasse enthält dedizierte Plot-Methoden für alle Parameter.
    Dieses Design verhindert die ValueError-Crashes bei plt.colorbar().
    """
    
    @staticmethod
    def plot_temperature(ax, fig, lons, lats, data, param_name):
        """Kühlt oder heizt: Rendert Temperatur und Taupunkt."""
        val_c = data - 273.15 if data.max() > 100 else data
        label_txt = "Taupunkt in °C" if "Taupunkt" in param_name else "Temperatur in °C"
        cmap = ColormapRegistry.get_temperature()
        norm = mcolors.Normalize(vmin=-30, vmax=30)
        im = ax.pcolormesh(lons, lats, val_c, cmap=cmap, norm=norm, shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label=label_txt, shrink=0.45, pad=0.03, ticks=np.arange(-30, 31, 10))

    @staticmethod
    def plot_precipitation(ax, fig, lons, lats, data):
        """Zeichnet Regen und Schnee."""
        cmap, norm = ColormapRegistry.get_precipitation()
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto', zorder=5, alpha=0.85)
        ticks_precip = list(range(0, 55, 5)) 
        fig.colorbar(im, ax=ax, label="Niederschlagssumme in mm", shrink=0.45, pad=0.03, ticks=ticks_precip)

    @staticmethod
    def plot_wind(ax, fig, lons, lats, data):
        """Zeichnet Windböen und Sturm."""
        val_kmh = data * 3.6 if data.max() < 100 else data # Fallback für m/s
        cmap = ColormapRegistry.get_wind()
        norm = mcolors.Normalize(vmin=0, vmax=150)
        im = ax.pcolormesh(lons, lats, val_kmh, cmap=cmap, norm=norm, shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label="Windböen in km/h", shrink=0.45, pad=0.03)

    @staticmethod
    def plot_cape(ax, fig, lons, lats, data):
        """Zeichnet Konvektive Verfügbare Potentielle Energie (Gewitterpotenzial)."""
        cmap, norm = ColormapRegistry.get_cape()
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label="CAPE (Energie) in J/kg", shrink=0.45, pad=0.03, ticks=[0, 100, 500, 1000, 2000, 3000, 5000])

    @staticmethod
    def plot_radar_simulated(ax, fig, lons, lats, data):
        """Zeichnet simuliertes Radar aus den Modellen (dBZ)."""
        cmap, norm = ColormapRegistry.get_radar()
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label="Simuliertes Radar (Reflektivität in dBZ)", shrink=0.45, pad=0.03, ticks=[0, 15, 30, 45, 60, 75])

    @staticmethod
    def plot_clouds(ax, fig, lons, lats, data):
        """Zeichnet die Gesamtbedeckung."""
        cmap = ColormapRegistry.get_generic_cmap("clouds")
        norm = mcolors.Normalize(vmin=0, vmax=100)
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label="Bewölkung in %", shrink=0.45, pad=0.03)

    @staticmethod
    def plot_significant_weather(ax, fig, lons, lats, data):
        """Wandelt WMO ww-Codes in eine farbige Legende um."""
        cmap_ww, legend_dict = ColormapRegistry.get_significant_weather()
        grid = np.zeros_like(data)
        
        for i, (l, (c, codes)) in enumerate(legend_dict.items(), 1):
            for code in codes: 
                grid[data == code] = i
                
        ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, shading='nearest', zorder=5, alpha=0.9)
        patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in legend_dict.items()]
        ax.legend(handles=patches, loc='lower left', title="Wetter-Klassifikation", fontsize='6', title_fontsize='7', framealpha=0.9).set_zorder(25)

    @staticmethod
    def plot_rainviewer_radar(ax, fig, host, path, region):
        """
        Zieht das RainViewer Mosaik in die Karte und erstellt eine
        erklärende Legende, da die Kacheln bereits farbig vom Server kommen.
        """
        rv_tiles = RainViewerTiles(host=host, path=path)
        zoom_l = {"Deutschland": 6, "Brandenburg/Berlin": 8, "Mitteleuropa (DE, PL)": 6, "Europa": 5}
        
        # Lege das Radar auf die Karte (Zorder=5 liegt über Satellit)
        ax.add_image(rv_tiles, zoom_l.get(region, 6), zorder=5, alpha=0.85)
        
        # Generiere die Legende (colorbar) für das UI
        cmap_radar, norm_radar = ColormapRegistry.get_radar()
        sm = plt.cm.ScalarMappable(cmap=cmap_radar, norm=norm_radar)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Radar-Reflektivität in dBZ (RainViewer)", shrink=0.45, pad=0.03)

    @staticmethod
    def add_isobars(ax, isobars_data, ilons, ilats):
        """Zeichnet Luftdrucklinien über die Karte."""
        if isobars_data is not None:
            p_hpa = isobars_data / 100 if isobars_data.max() > 5000 else isobars_data
            if ilons.ndim == 1: 
                ilons, ilats = np.meshgrid(ilons, ilats)
            cs = ax.contour(ilons, ilats, p_hpa, colors='black', linewidths=0.7, levels=np.arange(940, 1060, 4), zorder=20)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

    @staticmethod
    def add_storm_hatching(ax, ww_data, wlons, wlats):
        """Zeichnet rote Linien über vorhergesagte Gewittergebiete."""
        if ww_data is not None:
            storm_mask = np.isin(ww_data, [95, 96, 97, 99])
            if np.any(storm_mask):
                plot_ww = np.zeros_like(ww_data)
                plot_ww[storm_mask] = 1 
                plt.rcParams['hatch.linewidth'] = 2.0 
                ax.contourf(wlons, wlats, plot_ww, levels=[0.5, 1.5], colors='none', hatches=['////'], edgecolors='red', zorder=10)


# ==============================================================================
# 8. DYNAMISCHES USER INTERFACE (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.header("🛰️ Modell-Zentrale")
    
    with st.expander("🌍 1. Modell wählen", expanded=True):
        sel_model = st.radio("Wettermodell", list(MODEL_ROUTER.keys()), label_visibility="collapsed")
    
    with st.expander("🗺️ 2. Karten-Ausschnitt", expanded=False):
        valid_regions = MODEL_ROUTER[sel_model]["regions"]
        default_idx = valid_regions.index("Brandenburg/Berlin") if "Brandenburg/Berlin" in valid_regions else 0
        sel_region = st.radio("Region", valid_regions, index=default_idx, label_visibility="collapsed")
    
    with st.expander("🌪️ 3. Parameter wählen", expanded=True):
        valid_params = MODEL_ROUTER[sel_model]["params"]
        sel_param = st.radio("Parameter", valid_params, label_visibility="collapsed")
    
    with st.expander("⏱️ 4. Vorhersage-Stunde (MEZ/MESZ)", expanded=True):
        if "Radar" in sel_model:
            st.info("Echtzeit-Daten: Die Zeitauswahl ist automatisch deaktiviert.")
            sel_hour = 0
            sel_hour_str = "Live"
        else:
            if "EU" in sel_model: hours = list(range(1, 79))
            elif "GFS" in sel_model or "Global" in sel_model: hours = list(range(3, 123, 3))
            elif "ECMWF" in sel_model: hours = list(range(3, 147, 3))
            else: hours = list(range(1, 49))
            
            now_utc = datetime.now(timezone.utc)
            base_run = DataFetcher.estimate_latest_run(sel_model, now_utc)
            
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
    
    # OVERLAY STEUERUNG
    show_sat = st.checkbox("🌍 Satelliten-Hintergrund (Google Maps)", value=True, help="Rendert ein fotorealistisches Bild unter dem Wetter.")
    show_isobars = st.checkbox("Isobaren (Luftdruck) einblenden", value=True)
    show_storms = st.checkbox("⚡ Gewitter-Risiko rot schraffieren", value=True)
    
    # Auto-Update Schalter exklusiv für das Live-Radar
    enable_refresh = st.checkbox("🔄 Auto-Update (5 Min.)", value=False)
    if enable_refresh and st_autorefresh is not None:
        st_autorefresh(interval=300000, key="auto_refresh_radar")
        
    st.markdown("---")
    generate = st.button("🚀 Profi-Karte generieren", use_container_width=True)
    
    with st.expander("🛠️ Entwickler-Konsole"):
        debug_mode = st.checkbox("URL-Ping aktivieren (Zeigt Lade-Pfade)")


# ==============================================================================
# 9. MAIN EXECUTION (DATEN LADEN & KARTE RENDERN)
# ==============================================================================
if generate or (enable_refresh and "Radar" in sel_model):
    
    # Speicher freigeben bevor neue Daten geladen werden
    SystemManager.cleanup_temp_files()
    
    with st.spinner(f"🛰️ Lade {sel_param} aus {sel_model}..."):
        
        # Hauptdaten laden
        data, lons, lats, run_id, d_logs = DataFetcher.fetch_model_data(sel_model, sel_param, sel_hour, debug_mode)
        
        # Isobaren Overlay Daten laden (Wird bei Radar automatisch übersprungen)
        iso_data, ilons, ilats = None, None, None
        if show_isobars and "Radar" not in sel_model:
            iso_data, ilons, ilats, _, _ = DataFetcher.fetch_model_data(sel_model, "Isobaren", sel_hour)
            
        # Gewitter Overlay Daten laden
        ww_data, wlons, wlats = None, None, None
        if show_storms and "Radar" not in sel_model and sel_param != "Signifikantes Wetter" and "Signifikantes Wetter" in MODEL_ROUTER[sel_model]["params"]:
            ww_data, wlons, wlats, _, _ = DataFetcher.fetch_model_data(sel_model, "Signifikantes Wetter", sel_hour)

    if debug_mode and d_logs:
        st.write("📡 **Interne Server-Pings (Debug):**")
        for log in d_logs[:4]: 
            st.code(log)

    if data is not None:
        try:
            # Karte und Projektion initialisieren
            fig, ax = plt.subplots(figsize=(9, 11), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
            
            # Kartenausschnitt setzen
            current_extent = EXTENTS[sel_region]
            ax.set_extent(current_extent)

            # SATELLITEN BILD SCHICHT (Zorder=0)
            if show_sat:
                # Nutzt die fehlerfreie Google Maps Methode
                sat_tiles = GoogleSatelliteTiles()
                # Dynamischer Zoom: Brandenburg braucht mehr Zoom als Europa
                zoom_l = {"Deutschland": 6, "Brandenburg/Berlin": 8, "Mitteleuropa (DE, PL)": 6, "Alpenraum": 7, "Europa": 5}
                ax.add_image(sat_tiles, zoom_l.get(sel_region, 6), zorder=0)

            # Poltische Grenzen zeichnen
            # Weiße Linien auf Satellit, Schwarze Linien auf normalem Hintergrund
            border_color = 'white' if show_sat else 'black'
            ax.add_feature(cfeature.COASTLINE, linewidth=0.9, edgecolor=border_color, zorder=12)
            ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor=border_color, zorder=12)
            states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
            ax.add_feature(states, linewidth=0.5, edgecolor=border_color, linestyle=":", zorder=12)

            # ------------------------------------------------------------------
            # PARAMETER DISPATCHER (Das Herzstück der sauberen Architektur)
            # ------------------------------------------------------------------
            if "Radar" in sel_param:
                if "RainViewer" in sel_model:
                    # Hier sind 'data' und 'lons' Host und Path der API
                    PlottingEngine.plot_rainviewer_radar(ax, fig, data, lons, sel_region)
                else:
                    PlottingEngine.plot_radar_simulated(ax, fig, lons, lats, data)
                    
            elif "Temperatur" in sel_param or "Taupunkt" in sel_param or "850 hPa Temp." in sel_param:
                PlottingEngine.plot_temperature(ax, fig, lons, lats, data, sel_param)
                
            elif "Niederschlag" in sel_param:
                PlottingEngine.plot_precipitation(ax, fig, lons, lats, data)
                
            elif "Wind" in sel_param:
                PlottingEngine.plot_wind(ax, fig, lons, lats, data)
                
            elif "CAPE" in sel_param:
                PlottingEngine.plot_cape(ax, fig, lons, lats, data)
                
            elif "Gesamtbedeckung" in sel_param:
                PlottingEngine.plot_clouds(ax, fig, lons, lats, data)
                
            elif "Signifikantes Wetter" in sel_param:
                PlottingEngine.plot_significant_weather(ax, fig, lons, lats, data)
                
            elif "Bodendruck" in sel_param:
                val_hpa = data / 100 if data.max() > 5000 else data
                im = ax.pcolormesh(lons, lats, val_hpa, cmap='jet', shading='auto', alpha=0.85, zorder=5)
                fig.colorbar(im, ax=ax, label="Bodendruck in hPa", shrink=0.45, pad=0.03)
                
            else:
                # Generischer Fallback Plotter für alle übrigen Parameter
                im = ax.pcolormesh(lons, lats, data, cmap='viridis', shading='auto', alpha=0.85, zorder=5)
                fig.colorbar(im, ax=ax, label=sel_param, shrink=0.45, pad=0.03)

            # ------------------------------------------------------------------
            # OVERLAYS HINZUFÜGEN
            # ------------------------------------------------------------------
            PlottingEngine.add_isobars(ax, iso_data, ilons, ilats)
            PlottingEngine.add_storm_hatching(ax, ww_data, wlons, wlats)

            # ------------------------------------------------------------------
            # HEADER & META-INFORMATIONEN
            # ------------------------------------------------------------------
            if "Radar" in sel_model:
                try:
                    # RainViewer liefert UNIX-Timestamps
                    dt_obj = datetime.fromtimestamp(int(run_id), tz=timezone.utc)
                    dt_loc = dt_obj.astimezone(LOCAL_TZ)
                    time_display = dt_loc.strftime('%d.%m.%Y %H:%M') + (" MESZ" if dt_loc.dst() else " MEZ")
                except Exception:
                    time_display = run_id
                    
                info_txt = f"Modell: {sel_model}\nParameter: {sel_param}\nLive-Stand: {time_display}\n(Quelle: RainViewer API)"
            else:
                # Zeigeberechnung für klassische Prognosemodelle
                v_dt_utc = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)
                v_dt_loc = v_dt_utc.astimezone(LOCAL_TZ)
                tz_str = "MESZ" if v_dt_loc.dst() else "MEZ"
                info_txt = f"Modell: {sel_model}\nParameter: {sel_param}\nTermin: {v_dt_loc.strftime('%d.%m.%Y %H:%M')} {tz_str}\nModell-Lauf: {run_id[-2:]}Z"
                
            # Textbox auf die Karte zeichnen
            ax.text(0.02, 0.98, info_txt, transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', 
                    bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3', edgecolor='gray'), zorder=30)

            # FINALE AUSGABE IN STREAMLIT
            st.pyplot(fig)
            
            # ------------------------------------------------------------------
            # BILD-DOWNLOAD GENERATOR
            # ------------------------------------------------------------------
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                dl_suffix = "LIVE" if "Radar" in sel_model else f"{sel_hour}h"
                st.download_button(
                    label="📥 Karte als PNG speichern",
                    data=img_buffer,
                    file_name=f"WarnwetterBB_{sel_model.replace(' ', '')}_{sel_param.replace(' ', '_')}_{dl_suffix}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Bereinige den Streamlit Container von alten GRIB-Downloads
            SystemManager.cleanup_temp_files()
            
        except Exception as plot_err:
            st.error(f"⚠️ Es gab ein Problem beim Rendern der Karte: {str(plot_err)}")
            if debug_mode:
                st.code(traceback.format_exc())
            
    else:
        st.error(f"⚠️ Aktuell liefert der gewählte Server keine Daten für '{sel_param}'.")
        st.info("💡 Server-Delay: Der DWD-Lauf ist vermutlich noch im Upload. Versuche es in 5 Minuten noch einmal.")

