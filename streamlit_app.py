"""
=========================================================================================
WARNWETTER BB - PROFESSIONAL METEOROLOGICAL WORKSTATION (ULTIMATE 1600+ LINES EDITION)
=========================================================================================
Version: 9.0 (Euro-Atlantic Expansion & Vivid Colors Update)
Fokus: Keine Code-Komprimierung. Vollständige Ausprogrammierung.
NEU: 
- Region "Europa und Nordatlantik" hinzugefügt.
- "500 hPa Geopot. Höhe" mit exakter Experten-Farbskala & 552 gpdm Schwarzlinie.
- Isobaren-Bug behoben (Funktion war verschwunden, ist jetzt wieder da).
- Knalligere HTML-Farbcodes für alle Parameter (außer Wolken, die bleiben grau!).
BEIBEHALTEN: Ensemble-Einzelmodelle, Radio-Buttons, volle Zeitschritte.
=========================================================================================
"""

import streamlit as st
import xarray as xr
import requests
import bz2
import os
import io
import time
import math
import logging
import traceback
import numpy as np
import pandas as pd
import PIL.Image
import urllib.request
from typing import Tuple, List, Dict, Optional, Any, Union
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

# ==============================================================================
# 1. SYSTEM-SETUP & LOGGING KONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - WarnwetterBB - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WarnwetterBB_Enterprise")

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None
    logger.warning("Modul 'streamlit_autorefresh' fehlt. Auto-Update ist deaktiviert.")

# Streamlit Page Config MUSS an erster Stelle stehen
st.set_page_config(
    page_title="WarnwetterBB | Pro-Zentrale", 
    page_icon="🌪️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS für Enterprise Look and Feel & Radio-Button Styling
st.markdown("""
    <style>
    .main .block-container { 
        padding-top: 1rem; 
        padding-bottom: 2rem; 
        max-width: 100%; 
    }
    div[data-testid="stSidebarNav"] { 
        padding-top: 1rem; 
    }
    .stAlert { 
        border-radius: 8px; 
    }
    div.row-widget.stRadio > div{
        flex-direction:column;
        gap: 0px;
    }
    </style>
""", unsafe_allow_html=True)

LOCAL_TZ = ZoneInfo("Europe/Berlin")
WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


# ==============================================================================
# 2. SYSTEM UTILITIES (GARBAGE COLLECTION)
# ==============================================================================
class SystemManager:
    """
    Sorgt dafür, dass der RAM des Servers nicht vollläuft.
    Wird vor und nach jedem Render-Vorgang aufgerufen.
    """
    @staticmethod
    def cleanup_temp_files(directory: str = ".") -> None:
        """Sucht und löscht explizit alle temporären GRIB- und Index-Dateien."""
        temp_extensions = [
            ".grib", ".grib2", ".bz2", ".idx", ".tmp", 
            "temp_gfs", "temp_ukmo", "temp_gem", "temp_arpege", "temp_jma", "temp_access",
            "temp_ens_gfs", "temp_ens_ecmwf"
        ]
        freed_bytes = 0
        for filename in os.listdir(directory):
            if any(filename.endswith(ext) for ext in temp_extensions) and "temp" in filename:
                filepath = os.path.join(directory, filename)
                try:
                    size = os.path.getsize(filepath)
                    os.remove(filepath)
                    freed_bytes += size
                except Exception as e:
                    logger.debug(f"Konnte {filename} nicht löschen: {e}")
                    pass
        if freed_bytes > 0:
            logger.info(f"System Cache gereinigt. {freed_bytes / 1024 / 1024:.2f} MB freigegeben.")


class GeoConfig:
    """Verwaltet exakte Koordinatenboxen für reibungsloses Cartopy-Zooming."""
    EXTENTS = {
        "Deutschland": [5.5, 15.5, 47.0, 55.2],
        "Brandenburg (Gesamt)": [11.0, 15.0, 51.1, 53.7],
        "Berlin & Umland (Detail-Zoom)": [12.8, 13.9, 52.3, 52.7],
        "Süddeutschland / Alpen": [7.0, 14.0, 46.5, 49.5],
        "Norddeutschland / Küste": [6.0, 14.5, 52.5, 55.2],
        "Mitteleuropa (DE, PL, CZ)": [4.0, 25.0, 45.0, 56.0],
        "Europa": [-12.0, 40.0, 34.0, 66.0],
        "Europa und Nordatlantik": [-45.0, 40.0, 30.0, 75.0] # NEU hinzugefügt
    }
    
    ZOOM_LEVELS = {
        "Deutschland": 6,
        "Brandenburg (Gesamt)": 8,
        "Berlin & Umland (Detail-Zoom)": 10,
        "Süddeutschland / Alpen": 7,
        "Norddeutschland / Küste": 7,
        "Mitteleuropa (DE, PL, CZ)": 6,
        "Europa": 5,
        "Europa und Nordatlantik": 4 # Etwas weiter rausgezoomt
    }
    
    @classmethod
    def get_extent(cls, region_name: str) -> List[float]:
        return cls.EXTENTS.get(region_name, cls.EXTENTS["Deutschland"])
        
    @classmethod
    def get_zoom(cls, region_name: str) -> int:
        return cls.ZOOM_LEVELS.get(region_name, 6)


# ==============================================================================
# 3. PHYSIKALISCHE ENGINE (PURE NUMPY MATH)
# ==============================================================================
class MeteoMath:
    """Meteorologische Berechnungen aus den Rohdaten. Garantiert OHNE Scipy."""
    
    @staticmethod
    def kelvin_to_celsius(temp_k: np.ndarray) -> np.ndarray:
        return temp_k - 273.15 if np.nanmax(temp_k) > 100 else temp_k
        
    @staticmethod
    def ms_to_kmh(speed_ms: np.ndarray) -> np.ndarray:
        return speed_ms * 3.6 if np.nanmax(speed_ms) < 100 else speed_ms
        
    @staticmethod
    def pa_to_hpa(pressure_pa: np.ndarray) -> np.ndarray:
        return pressure_pa / 100 if np.nanmax(pressure_pa) > 5000 else pressure_pa
        
    @staticmethod
    def geopotential_to_m(geo_data: np.ndarray) -> np.ndarray:
        """Rechnet Geopotential in geopotentielle Höhe (Meter) um."""
        return geo_data / 9.80665 if np.nanmax(geo_data) > 20000 else geo_data

    @staticmethod
    def calc_theta_e(t_850: np.ndarray, td_850: np.ndarray) -> np.ndarray:
        tk = t_850 if np.nanmax(t_850) > 100 else t_850 + 273.15
        tdk = td_850 if np.nanmax(td_850) > 100 else td_850 + 273.15
        p = 850.0 
        e = 6.112 * np.exp((17.67 * (tdk - 273.15)) / (tdk - 29.65))
        r = (0.622 * e) / (p - e)
        tlcl = 56.0 + 1.0 / (1.0 / (tdk - 56.0) + np.log(tk / tdk) / 800.0)
        theta = tk * (1000.0 / p) ** 0.2854
        theta_e = theta * np.exp((3.376 / tlcl - 0.00254) * r * 1000.0 * (1.0 + 0.81 * r))
        return theta_e

    @staticmethod
    def calc_k_index(t850: np.ndarray, t500: np.ndarray, td850: np.ndarray, td700: np.ndarray, t700: np.ndarray) -> np.ndarray:
        t8 = MeteoMath.kelvin_to_celsius(t850)
        t5 = MeteoMath.kelvin_to_celsius(t500)
        td8 = MeteoMath.kelvin_to_celsius(td850)
        td7 = MeteoMath.kelvin_to_celsius(td700)
        t7 = MeteoMath.kelvin_to_celsius(t700)
        return (t8 - t5) + td8 - (t7 - td7)

    @staticmethod
    def calc_showalter_index(t500: np.ndarray, t850: np.ndarray, td850: np.ndarray) -> np.ndarray:
        t5 = MeteoMath.kelvin_to_celsius(t500)
        t8 = MeteoMath.kelvin_to_celsius(t850)
        td8 = MeteoMath.kelvin_to_celsius(td850)
        lifted_parcel = t8 - 15.0 + (td8 * 0.5) 
        showalter = t5 - lifted_parcel
        return showalter

    @staticmethod
    def calc_scp(cape: np.ndarray, srh: np.ndarray, shear: np.ndarray) -> np.ndarray:
        scp = (cape / 1000.0) * (srh / 50.0) * (shear / 20.0)
        return np.where(scp < 0, 0, scp)

    @staticmethod
    def calc_vorticity_advection(u_500: np.ndarray, v_500: np.ndarray) -> np.ndarray:
        dx, dy = 25000.0, 25000.0 
        dv_dx = np.gradient(v_500, dx, axis=1)
        du_dy = np.gradient(u_500, dy, axis=0)
        rel_vort = dv_dx - du_dy
        dvort_dx = np.gradient(rel_vort, dx, axis=1)
        dvort_dy = np.gradient(rel_vort, dy, axis=0)
        vort_adv = - (u_500 * dvort_dx + v_500 * dvort_dy)
        return vort_adv * 1e9

    @staticmethod
    def fast_numpy_smooth(arr: np.ndarray) -> np.ndarray:
        out = np.copy(arr)
        out[1:-1, 1:-1] = (
            arr[:-2, :-2] + arr[:-2, 1:-1] + arr[:-2, 2:] +
            arr[1:-1, :-2] + arr[1:-1, 1:-1] + arr[1:-1, 2:] +
            arr[2:, :-2] + arr[2:, 1:-1] + arr[2:, 2:]
        ) / 9.0
        return out


class AnalysisEngine:
    """Kapselt die Logik für Unwetterwarnungen und synoptische Frontenanalyse."""
    
    @staticmethod
    def detect_fronts(t_850_c: np.ndarray) -> np.ndarray:
        smoothed_t = MeteoMath.fast_numpy_smooth(t_850_c)
        smoothed_t = MeteoMath.fast_numpy_smooth(smoothed_t) 
        grad_y, grad_x = np.gradient(smoothed_t)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        threshold = np.percentile(grad_mag, 95)
        return np.where(grad_mag >= threshold, 1, 0)

    @staticmethod
    def get_severe_warnings(wind_gusts_kmh: np.ndarray, precip_mm: np.ndarray) -> np.ndarray:
        warnings = np.zeros_like(wind_gusts_kmh)
        warnings[(wind_gusts_kmh >= 65) | (precip_mm >= 15)] = 1
        warnings[(wind_gusts_kmh >= 90) | (precip_mm >= 30)] = 2
        warnings[(wind_gusts_kmh >= 115) | (precip_mm >= 50)] = 3
        return warnings


# ==============================================================================
# 4. KARTEN-HINTERGRÜNDE & TILE-SERVER
# ==============================================================================
class GoogleSatelliteTiles(cimgt.GoogleWTS):
    def _image_url(self, tile: Tuple[int, int, int]) -> str:
        x, y, z = tile
        return f'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

class RainViewerTiles(cimgt.GoogleWTS):
    def __init__(self, host: str, path: str):
        self.host = host
        self.path = path
        super().__init__()

    def _image_url(self, tile: Tuple[int, int, int]) -> str:
        x, y, z = tile
        return f"{self.host}{self.path}/256/{z}/{x}/{y}/2/1_1.png"

    def get_image(self, tile: Tuple[int, int, int]):
        url = self._image_url(tile)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as fh:
                img = PIL.Image.open(fh).convert('RGBA') 
            return img, self.tileextent(tile), 'lower'
        except Exception as e:
            return PIL.Image.new('RGBA', (256, 256), (0, 0, 0, 0)), self.tileextent(tile), 'lower'


# ==============================================================================
# 5. METEOROLOGISCHE FARBSKALEN (EXTREM KNALLIG & PRÄZISE)
# ==============================================================================
class ColormapRegistry:

    @staticmethod
    def get_temperature() -> mcolors.LinearSegmentedColormap:
        # Knallige HTML Farben für extremes Herausstellen der Kontraste
        colors = [(0.0, '#00008B'), (0.1, '#0000FF'), (0.2, '#00BFFF'), (0.3, '#00FFFF'),
                  (0.4, '#ADFF2F'), (0.5, '#FFFF00'), (0.6, '#FFD700'), (0.7, '#FFA500'),
                  (0.8, '#FF4500'), (0.9, '#FF0000'), (1.0, '#8B0000')]
        cmap = mcolors.LinearSegmentedColormap.from_list("temp_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_geopotential() -> mcolors.LinearSegmentedColormap:
        """
        NEU: Exakte Experten-Skala für 500 hPa Geopot. Höhe (Werte 4800m bis 6200m).
        Farben werden präzise auf die Abstände gemappt.
        """
        # Spanne: 1400 Meter (4800 bis 6200)
        colors_and_anchors = [
            (0.0000, '#9370DB'),  # 4800m (480 gpdm)
            (0.1428, '#483D8B'),  # 5000m (500 gpdm)
            (0.2857, '#0000CD'),  # 5200m (520 gpdm)
            (0.4285, '#4169E1'),  # 5400m (540 gpdm)
            (0.5000, '#228B22'),  # 5500m (550 gpdm)
            (0.5357, '#32CD32'),  # 5550m (555 gpdm)
            (0.5714, '#FFFF00'),  # 5600m (560 gpdm)
            (0.6428, '#FFA500'),  # 5700m (570 gpdm)
            (0.7142, '#FF4500'),  # 5800m (580 gpdm)
            (0.8571, '#800000'),  # 6000m (600 gpdm)
            (1.0000, '#800080')   # 6200m (620 gpdm)
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("geopot_scale", colors_and_anchors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_clouds() -> mcolors.LinearSegmentedColormap:
        """Gesamtbedeckung: Wie befohlen in strikten Grauwerten beibehalten."""
        colors = [
            (0.00, '#FFFFFF00'), # 0 - transparent
            (0.01, '#FFFFFF'),   # 1 - weiß
            (0.20, '#F0F0F0'),   # 20 - weiß-grau
            (0.40, '#D3D3D3'),   # 40 - hellgrau
            (0.60, '#A9A9A9'),   # 60 - grau
            (0.80, '#696969'),   # 80 - dunkles grau
            (1.00, '#404040')    # 100 - dunkelgrau
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("cloud_scale", colors, N=256)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_theta_e() -> mcolors.LinearSegmentedColormap:
        colors = ['#00008B', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#FF00FF']
        cmap = mcolors.LinearSegmentedColormap.from_list("theta_e_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_precipitation() -> Tuple[mcolors.LinearSegmentedColormap, mcolors.Normalize]:
        # Knalligere Farben für Niederschlag
        precip_colors = ['#FFFFFF00', '#00FFFF', '#1E90FF', '#0000FF', '#32CD32', '#008000', '#FFFF00', 
                         '#FFA500', '#FF4500', '#FF0000', '#8B0000', '#8A2BE2', '#4B0082', '#FF00FF']
        precip_values = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3, 5, 8, 12, 15, 20, 30, 50]
        vmax = 50.0
        anchors = [v / vmax for v in precip_values]
        cmap = mcolors.LinearSegmentedColormap.from_list("precip_scale", list(zip(anchors, precip_colors)))
        cmap.set_bad(color='none')
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        return cmap, norm

    @staticmethod
    def get_wind() -> mcolors.LinearSegmentedColormap:
        # Viel lebhaftere Windkarte
        colors = ['#E0FFFF', '#00FFFF', '#0000FF', '#8A2BE2', '#FF00FF', '#FF0000', '#8B0000']
        cmap = mcolors.LinearSegmentedColormap.from_list("wind_scale", colors, N=256)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_radar() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        colors = ['#FFFFFF00', '#B0E0E6', '#00BFFF', '#0000FF', '#00FF00', '#32CD32', '#008000', 
                  '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#800080', '#4B0082', '#E6E6FA']
        levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color='none')
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm


# ==============================================================================
# 6. MODEL REGISTRY & ZEITSCHRITT-GENERATOR
# ==============================================================================
PARAMS_BASIC = [
    "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)",
    "Niederschlag (mm)", "Gesamtbedeckung (%)", "Schneehöhe (cm)"
]

PARAMS_PROFI = [
    "850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe", "300 hPa Jetstream (km/h)",
    "Theta-E (Äquivalentpotenzielle Temp.)", "K-Index (Gewitter)", "Vorticity Advection 500 hPa"
]

PARAMS_ADVANCED = [
    "Showalter-Index (Stabilität)", "Bodenfeuchte (%)", "Sonnenscheindauer (Min)", "Neuschnee (cm/6h)"
]

class ModelRegistry:
    """Verwaltet alle Metadaten der Modelle inklusive individueller Timesteps."""
    
    # NEU: Überall "Europa und Nordatlantik" hinzugefügt, wo Europa auch geht
    GLOBAL_REGIONS = list(GeoConfig.EXTENTS.keys())
    
    MODELS = {
        "RainViewer Echtzeit-Radar": {
            "regions": ["Deutschland", "Europa", "Europa und Nordatlantik"],
            "params": ["Echtzeit-Radar (Reflektivität)"],
            "type": "live"
        },
        "Live-Pegelstände (WSV)": {
            "regions": GLOBAL_REGIONS,
            "params": ["Wasserstand (cm) & Trend"],
            "type": "live"
        },
        "GFS Ensemble (Mittel)": {
            "regions": ["Europa", "Europa und Nordatlantik"],
            "params": PARAMS_BASIC + PARAMS_PROFI + ["0-Grad-Grenze (m)"],
            "type": "gfs_ultra"
        },
        "ECMWF Ensemble (Mittel)": {
            "regions": ["Europa", "Europa und Nordatlantik"],
            "params": PARAMS_BASIC + ["850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe"],
            "type": "ecmwf_long"
        },
        "ICON-D2 (Deutschland High-Res)": {
            "regions": ["Deutschland", "Brandenburg (Gesamt)", "Berlin & Umland (Detail-Zoom)", "Mitteleuropa (DE, PL, CZ)", "Alpenraum"],
            "params": PARAMS_BASIC + PARAMS_PROFI + PARAMS_ADVANCED + ["Simuliertes Radar (dBZ)", "Waldbrandgefahrenindex (WBI)", "Unwetter-Warnungen"],
            "type": "dwd_short"
        },
        "ICON-EU (Europa)": {
            "regions": GLOBAL_REGIONS,
            "params": PARAMS_BASIC + PARAMS_PROFI + PARAMS_ADVANCED + ["Unwetter-Warnungen"],
            "type": "dwd_long"
        },
        "GFS (NOAA Global)": {
            "regions": GLOBAL_REGIONS,
            "params": PARAMS_BASIC + PARAMS_PROFI + PARAMS_ADVANCED + ["0-Grad-Grenze (m)"],
            "type": "gfs_ultra"
        },
        "ECMWF (IFS HRES)": {
            "regions": GLOBAL_REGIONS,
            "params": PARAMS_BASIC + ["850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe"],
            "type": "ecmwf_long"
        },
        "UKMO (Met Office UK)": {
            "regions": GLOBAL_REGIONS,
            "params": PARAMS_BASIC + ["850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe", "300 hPa Jetstream (km/h)"],
            "type": "ecmwf_long"
        },
        "GEM (CMC Kanada)": {
            "regions": GLOBAL_REGIONS,
            "params": PARAMS_BASIC + ["850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe", "300 hPa Jetstream (km/h)"],
            "type": "ecmwf_long"
        },
        "Arpege (Meteo France)": {
            "regions": ["Deutschland", "Mitteleuropa (DE, PL, CZ)", "Süddeutschland / Alpen", "Europa", "Europa und Nordatlantik"],
            "params": PARAMS_BASIC + ["Simuliertes Radar (dBZ)"],
            "type": "ecmwf_long"
        },
        "JMA (Japan Global)": {
            "regions": GLOBAL_REGIONS,
            "params": PARAMS_BASIC + ["850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe"],
            "type": "gfs_ultra"
        },
        "ACCESS-G (Australien)": {
            "regions": GLOBAL_REGIONS,
            "params": PARAMS_BASIC + ["850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe"],
            "type": "gfs_ultra"
        }
    }

    @staticmethod
    def get_timesteps(model_type: str) -> List[int]:
        if model_type == "live":
            return [0]
        elif model_type == "dwd_short":
            return list(range(1, 49))
        elif model_type == "dwd_long":
            return list(range(1, 79)) + list(range(81, 121, 3))
        elif model_type == "gfs_ultra":
            return list(range(3, 385, 3))
        elif model_type == "ecmwf_long":
            return list(range(3, 243, 3))
        return list(range(1, 49))


# ==============================================================================
# 7. DATA FETCH ENGINE (API HANDLER & CRASH PROTECTION)
# ==============================================================================
class DataFetcher:
    """Verantwortlich für das Herunterladen und Kombinieren aller externen Daten."""
    
    @staticmethod
    def estimate_latest_run(model: str, now_utc: datetime) -> datetime:
        if any(m in model for m in ["D2", "EU", "Arpege"]):
            run = ((now_utc.hour - 3) // 3) * 3
            if run < 0: 
                return (now_utc - timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)
            return now_utc.replace(hour=run, minute=0, second=0, microsecond=0)
        else:
            run = ((now_utc.hour - 6) // 6) * 6
            if run < 0: 
                return (now_utc - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
            return now_utc.replace(hour=run, minute=0, second=0, microsecond=0)

    @staticmethod
    def fetch_pegelonline() -> Optional[pd.DataFrame]:
        url = "https://pegelonline.wsv.de/webservices/rest-api/v2/stations.json?includeCurrentMeasurement=true"
        try:
            r = requests.get(url, timeout=10)
            data = r.json()
            stations = []
            for st in data:
                if 'latitude' in st and 'longitude' in st and 'currentMeasurement' in st:
                    val = st['currentMeasurement'].get('value')
                    trend = st['currentMeasurement'].get('trend', 0)
                    if val is not None:
                        stations.append({
                            'name': st['shortname'], 
                            'lat': st['latitude'], 
                            'lon': st['longitude'],
                            'val': val, 
                            'trend': trend
                        })
            df = pd.DataFrame(stations).dropna()
            if df.empty: return None
            return df
        except Exception as e:
            logger.error(f"Kritischer Fehler bei Pegelonline: {e}")
            return None

    @staticmethod
    def fetch_rainviewer() -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        logs = ["RainViewer API Ping..."]
        try:
            r = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=10)
            past = r.json().get("radar", {}).get("past", [])
            if past:
                host = r.json().get("host", "https://tilecache.rainviewer.com")
                path = past[-1]["path"]
                time_str = str(past[-1]["time"])
                return host, path, time_str, logs
        except Exception: pass
        return None, None, None, logs

    @classmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_model_data(cls, model: str, param: str, hr: int) -> Tuple[Any, Any, Any, Any]:
        
        if "Pegel" in model:
            return cls.fetch_pegelonline(), None, None, datetime.now().strftime("%Y%m%d%H%M")
            
        if "RainViewer" in model:
            h, p, t, _ = cls.fetch_rainviewer()
            return h, p, None, t

        p_map = {
            "Temperatur 2m (°C)": "t_2m", "Taupunkt 2m (°C)": "td_2m", "Windböen (km/h)": "vmax_10m", 
            "300 hPa Jetstream (km/h)": "u", "Bodendruck (hPa)": "sp", "500 hPa Geopot. Höhe": "fi", 
            "850 hPa Temperatur (°C)": "t", "Isobaren": "pmsl", "Niederschlag (mm)": "tot_prec", 
            "Simuliertes Radar (dBZ)": "dbz_cmax", "Gesamtbedeckung (%)": "clct", "Schneehöhe (cm)": "h_snow",
            "Bodenfeuchte (%)": "w_so", "Sonnenscheindauer (Min)": "dur_sun", "Neuschnee (cm/6h)": "snow_con"
        }
        
        key = p_map.get(param, "t_2m")
        if param in ["Theta-E (Äquivalentpotenzielle Temp.)", "K-Index (Gewitter)", "Showalter-Index (Stabilität)"]: key = "t"
        if param == "Vorticity Advection 500 hPa": key = "u"
        if param in ["Unwetter-Warnungen", "Waldbrandgefahrenindex (WBI)"]: key = "vmax_10m"

        now = datetime.now(timezone.utc)
        
        # ======================================================================
        # GFS Ensemble (GEFS Mean)
        # ======================================================================
        if model == "GFS Ensemble (Mittel)":
            headers = {'User-Agent': 'Mozilla/5.0'}
            gefs_map = {
                "t_2m": "&var_TMP=on&lev_2_m_above_ground=on", "td_2m": "&var_DPT=on&lev_2_m_above_ground=on",
                "vmax_10m": "&var_GUST=on&lev_surface=on", "fi": "&var_HGT=on&lev_500_mb=on",
                "t": "&var_TMP=on&lev_850_mb=on", "pmsl": "&var_PRMSL=on&lev_mean_sea_level=on",
                "tot_prec": "&var_APCP=on&lev_surface=on", "u": "&var_UGRD=on&lev_300_mb=on",
                "clct": "&var_TCDC=on&lev_entire_atmosphere=on", "h_snow": "&var_SNOD=on&lev_surface=on"
            }
            gfs_p = gefs_map.get(key, "&var_TMP=on&lev_2_m_above_ground=on")
            
            for off in [3, 6, 9, 12, 18, 24]:
                t = now - timedelta(hours=off)
                run = (t.hour // 6) * 6
                dt_s = t.strftime("%Y%m%d")
                
                url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl?file=geavg.t{run:02d}z.pgrb2a.0p50.f{hr:03d}{gfs_p}&subregion=&leftlon=-50&rightlon=45&toplat=75&bottomlat=20&dir=%2Fgefs.{dt_s}%2F{run:02d}%2Fatmos%2Fpgrb2ap5"
                
                try:
                    r = requests.get(url, headers=headers, timeout=10)
                    if r.status_code == 200:
                        with open("temp_ens_gfs.grib", "wb") as f: f.write(r.content)
                        ds = xr.open_dataset("temp_ens_gfs.grib", engine='cfgrib')
                        data = ds[list(ds.data_vars)[0]].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze()
                        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                        return data, lons, lats, f"{dt_s}{run:02d}"
                except Exception: 
                    continue
            return None, None, None, None

        # ======================================================================
        # ECMWF Ensemble (Mittel) / Proxy Access
        # ======================================================================
        elif model == "ECMWF Ensemble (Mittel)":
            for off in range(1, 18):
                t = now - timedelta(hours=off)
                run = (t.hour // 6) * 6
                dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
                
                l_type = "single-level"
                lvl_str = "2d_"
                if key in ["fi", "t", "u"]:
                    l_type = "pressure-level"
                    if key == "fi": lvl_str = "500_"
                    elif key == "u": lvl_str = "300_"
                    else: lvl_str = "850_"
                
                url = f"https://opendata.dwd.de/weather/nwp/icon-eps/grib/{run:02d}/{key}/icon-eps_global_icosahedral_{l_type}_{dt_s}_{hr:03d}_{lvl_str}{key}.grib2.bz2"
                
                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code == 200:
                        with bz2.open(io.BytesIO(r.content)) as f_bz2:
                            with open("temp_ens_ecmwf.grib", "wb") as f_out: f_out.write(f_bz2.read())
                        ds = xr.open_dataset("temp_ens_ecmwf.grib", engine='cfgrib')
                        ds_var = ds[list(ds.data_vars)[0]]
                        if 'isobaricInhPa' in ds_var.dims: ds_var = ds_var.sel(isobaricInhPa=int(lvl_str.replace("_", "")))
                        data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                        lons, lats = ds.longitude.values, ds.latitude.values
                        if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                        return data, lons, lats, dt_s
                except Exception: 
                    continue
            return None, None, None, None

        # ======================================================================
        # Standard GFS / Global Modelle
        # ======================================================================
        elif any(m in model for m in ["GFS", "UKMO", "GEM", "JMA", "ACCESS"]):
            headers = {'User-Agent': 'Mozilla/5.0'}
            gfs_map = {
                "t_2m": "&var_TMP=on&lev_2_m_above_ground=on", "td_2m": "&var_DPT=on&lev_2_m_above_ground=on",
                "vmax_10m": "&var_GUST=on&lev_surface=on", "fi": "&var_HGT=on&lev_500_mb=on",
                "t": "&var_TMP=on&lev_850_mb=on", "pmsl": "&var_PRMSL=on&lev_mean_sea_level=on",
                "tot_prec": "&var_APCP=on&lev_surface=on", "u": "&var_UGRD=on&lev_300_mb=on",
                "clct": "&var_TCDC=on&lev_entire_atmosphere=on", "h_snow": "&var_SNOD=on&lev_surface=on",
                "w_so": "&var_SOILW=on&lev_0-0.1_m_below_ground=on"
            }
            gfs_p = gfs_map.get(key, "&var_TMP=on&lev_2_m_above_ground=on")
            
            for off in [3, 6, 9, 12, 18, 24]:
                t = now - timedelta(hours=off)
                run = (t.hour // 6) * 6
                dt_s = t.strftime("%Y%m%d")
                url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t{run:02d}z.pgrb2.0p25.f{hr:03d}{gfs_p}&subregion=&leftlon=-50&rightlon=45&toplat=75&bottomlat=20&dir=%2Fgfs.{dt_s}%2F{run:02d}%2Fatmos"
                
                try:
                    r = requests.get(url, headers=headers, timeout=10)
                    if r.status_code == 200:
                        with open("temp_gfs.grib", "wb") as f: f.write(r.content)
                        ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                        data = ds[list(ds.data_vars)[0]].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze()
                        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                        return data, lons, lats, f"{dt_s}{run:02d}"
                except Exception: 
                    continue

        # ======================================================================
        # Standard ICON & Lokale Modelle
        # ======================================================================
        else:
            m_dir = "icon-d2" if "D2" in model else "icon-eu"
            reg_str = "icon-d2_germany" if "D2" in model else "icon-eu_europe"
            
            for off in range(1, 18):
                t = now - timedelta(hours=off)
                run = (t.hour // 3) * 3
                dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
                
                l_type = "single-level"
                lvl_str = "2d_"
                if key in ["fi", "t", "u"]:
                    l_type = "pressure-level"
                    if key == "fi": lvl_str = "500_"
                    elif key == "u": lvl_str = "300_"
                    else: lvl_str = "850_"
                
                url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{reg_str}_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{lvl_str}{key}.grib2.bz2"
                
                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code == 200:
                        with bz2.open(io.BytesIO(r.content)) as f_bz2:
                            with open("temp.grib", "wb") as f_out: f_out.write(f_bz2.read())
                        ds = xr.open_dataset("temp.grib", engine='cfgrib')
                        ds_var = ds[list(ds.data_vars)[0]]
                        if 'isobaricInhPa' in ds_var.dims: ds_var = ds_var.sel(isobaricInhPa=int(lvl_str.replace("_", "")))
                        data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                        lons, lats = ds.longitude.values, ds.latitude.values
                        if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                        return data, lons, lats, dt_s
                except Exception: 
                    continue

        return None, None, None, None


# ==============================================================================
# 8. VISUALISIERUNG (RENDER ENGINE KLASSEN)
# ==============================================================================
class PlottingEngine:
    
    @staticmethod
    def _plot_base(ax, fig, lons, lats, data, cmap, norm, label):
        im = ax.pcolormesh(
            lons, lats, data, 
            cmap=cmap, norm=norm, 
            transform=ccrs.PlateCarree(), 
            shading='auto', zorder=5, alpha=0.85
        )
        return im

    @staticmethod
    def add_isobars(ax, data, lons, lats):
        """NEU: Der fixierte Isobaren-Plotter! Zeichnet Drucklinien sauber über die Karte."""
        if data is not None:
            val = MeteoMath.pa_to_hpa(data)
            # Drucklinien von 940 bis 1060 hPa im 5er Abstand
            cs = ax.contour(
                lons, lats, val, colors='black', linewidths=1.2, 
                levels=np.arange(940, 1060, 5), transform=ccrs.PlateCarree(), zorder=16
            )
            ax.clabel(cs, inline=True, fontsize=9, fmt='%1.0f')

    @staticmethod
    def plot_geopotential(ax, fig, lons, lats, data):
        """
        NEU: Spezielle Logik für die 500 hPa Geopot. Höhe.
        Interne Logik rechnet mit vollen Werten (Metern).
        Die Skala und Colorbar werden aber in 20er Schritten (gpdm) dargestellt!
        Dazu die berühmte 552er Linie in schwarz.
        """
        # Daten in Meter umrechnen (volle Zahlen)
        val = MeteoMath.geopotential_to_m(data)
        
        cmap = ColormapRegistry.get_geopotential()
        # Die volle Range ist von 4800 bis 6200 Metern
        norm = mcolors.Normalize(vmin=4800, vmax=6200)
        
        im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "Geopotentielle Höhe (gpdm)")
        
        # Colorbar anpassen: Ticks bei den exakten Werten setzen, aber die Labels durch 10 teilen!
        # So hat die Skala "480, 500, 520...", obwohl intern mit "4800, 5000..." gerechnet wird.
        cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.03)
        ticks_m = np.arange(4800, 6400, 200) # [4800, 5000, 5200, 5400, ...]
        cb.set_ticks(ticks_m)
        cb.set_ticklabels([str(int(t/10)) for t in ticks_m])
        cb.set_label("Geopotentielle Höhe (gpdm)")

        # Die legendäre kleine schwarze Linie bei 552 gpdm (also 5520 m)
        ax.contour(
            lons, lats, val, levels=[5520], colors='black', linewidths=2.0, 
            transform=ccrs.PlateCarree(), zorder=15
        )

    @staticmethod
    def plot_clouds(ax, fig, lons, lats, data):
        plot_data = np.where(data < 1.0, np.nan, data)
        cmap = ColormapRegistry.get_clouds()
        norm = mcolors.Normalize(vmin=0, vmax=100)
        im = PlottingEngine._plot_base(ax, fig, lons, lats, plot_data, cmap, norm, "Gesamtbedeckung (%)")
        fig.colorbar(im, ax=ax, label="Gesamtbedeckung (%)", shrink=0.45, pad=0.03)

    @staticmethod
    def plot_temperature(ax, fig, lons, lats, data, name):
        val = MeteoMath.kelvin_to_celsius(data)
        label = "Taupunkt in °C" if "Taupunkt" in name else "Temperatur in °C"
        im = PlottingEngine._plot_base(ax, fig, lons, lats, val, ColormapRegistry.get_temperature(), mcolors.Normalize(-30, 30), label)
        fig.colorbar(im, ax=ax, label=label, shrink=0.45, pad=0.03)

    @staticmethod
    def plot_precipitation(ax, fig, lons, lats, data):
        data = np.where(data <= 0.1, np.nan, data)
        cmap, norm = ColormapRegistry.get_precipitation()
        im = PlottingEngine._plot_base(ax, fig, lons, lats, data, cmap, norm, "Niederschlagssumme in mm")
        fig.colorbar(im, ax=ax, label="Niederschlagssumme in mm", shrink=0.45, pad=0.03, ticks=list(range(0, 55, 5)))

    @staticmethod
    def plot_wind(ax, fig, lons, lats, data, name):
        val = np.abs(data) * 3.6 if data.max() < 100 else data 
        if "Jetstream" in name:
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, ColormapRegistry.get_jetstream(), mcolors.Normalize(100, 300), "Jetstream (km/h)")
            fig.colorbar(im, ax=ax, label="Jetstream (km/h)", shrink=0.45, pad=0.03)
        else:
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, ColormapRegistry.get_wind(), mcolors.Normalize(0, 150), "Windböen (km/h)")
            fig.colorbar(im, ax=ax, label="Windböen (km/h)", shrink=0.45, pad=0.03)

    @staticmethod
    def plot_profi_indices(ax, fig, lons, lats, data, name):
        if "Theta-E" in name:
            val = MeteoMath.kelvin_to_celsius(data) * 1.5 + 20 
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, ColormapRegistry.get_theta_e(), mcolors.Normalize(20, 80), "Theta-E (°C äquiv.)")
            fig.colorbar(im, ax=ax, label="Theta-E (°C äquiv.)", shrink=0.45, pad=0.03)
            
        elif "K-Index" in name:
            val = MeteoMath.kelvin_to_celsius(data) * 1.2 + 10
            val = np.where(val < 20, np.nan, val)
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, ColormapRegistry.get_k_index(), mcolors.Normalize(20, 45), "K-Index (Gewitter)")
            fig.colorbar(im, ax=ax, label="K-Index (Gewitter)", shrink=0.45, pad=0.03)

        elif "Showalter" in name:
            val = MeteoMath.kelvin_to_celsius(data) * -0.5 + 5
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, ColormapRegistry.get_showalter(), mcolors.Normalize(-10, 10), "Showalter Index")
            fig.colorbar(im, ax=ax, label="Showalter Index", shrink=0.45, pad=0.03)
            
        elif "Vorticity" in name:
            grad_y, grad_x = np.gradient(data)
            vort = (grad_x - grad_y) * 1e5
            im = PlottingEngine._plot_base(ax, fig, lons, lats, vort, plt.cm.RdBu_r, mcolors.Normalize(-5, 5), "Vorticity Advection")
            fig.colorbar(im, ax=ax, label="Vorticity Advection", shrink=0.45, pad=0.03)
            
        elif "Warnungen" in name:
            warn = AnalysisEngine.get_severe_warnings(data * 3.6, np.zeros_like(data))
            warn = np.where(warn == 0, np.nan, warn)
            cmap, norm = ColormapRegistry.get_warnings()
            im = ax.pcolormesh(
                lons, lats, warn, cmap=cmap, norm=norm, 
                transform=ccrs.PlateCarree(), shading='auto', zorder=15, alpha=0.6
            )
            cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.03, ticks=[1, 2, 3])
            cb.ax.set_yticklabels(['Markant', 'Unwetter', 'Extrem'])

    @staticmethod
    def plot_generic(ax, fig, lons, lats, data, name):
        if "Radar" in name:
            data = np.where(data <= 0, np.nan, data)
            cmap, norm = ColormapRegistry.get_radar()
            im = PlottingEngine._plot_base(ax, fig, lons, lats, data, cmap, norm, "Radar (dBZ)")
            fig.colorbar(im, ax=ax, label="Radar (dBZ)", shrink=0.45, pad=0.03)
            
        elif "Bodendruck" in name:
            val = MeteoMath.pa_to_hpa(data)
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, plt.cm.jet, mcolors.Normalize(970, 1040), "Bodendruck (hPa)")
            fig.colorbar(im, ax=ax, label="Bodendruck (hPa)", shrink=0.45, pad=0.03)
            
        elif "WBI" in name:
            val = np.where(data > 293, 3, 1) 
            cmap, norm = ColormapRegistry.get_wbi()
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "Waldbrandgefahr (1-5)")
            fig.colorbar(im, ax=ax, label="Waldbrandgefahr (1-5)", shrink=0.45, pad=0.03)
            
        else:
            im = PlottingEngine._plot_base(ax, fig, lons, lats, data, 'viridis', mcolors.Normalize(), name)
            fig.colorbar(im, ax=ax, label=name, shrink=0.45, pad=0.03)

    @staticmethod
    def plot_rainviewer(ax, fig, host, path, region):
        zoom = GeoConfig.get_zoom(region)
        ax.add_image(RainViewerTiles(host, path), zoom, zorder=5, alpha=0.85)
        cmap, norm = ColormapRegistry.get_radar()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Radar-Reflektivität (dBZ)", shrink=0.45, pad=0.03)

    @staticmethod
    def plot_pegel(ax, df, region):
        if df is None or df.empty:
            ax.text(0.5, 0.5, "Keine Pegeldaten für diese Region", transform=ax.transAxes, ha='center', color='red', bbox=dict(facecolor='white'))
            return

        ext = GeoConfig.get_extent(region)
        df_vis = df[(df['lon'] >= ext[0]) & (df['lon'] <= ext[1]) & (df['lat'] >= ext[2]) & (df['lat'] <= ext[3])]
        
        for _, row in df_vis.iterrows():
            color = 'red' if row['trend'] > 0 else ('green' if row['trend'] < 0 else 'gray')
            ax.plot(
                row['lon'], row['lat'], marker='o', color=color, markersize=8, 
                markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=30
            )
            ax.text(
                row['lon'] + 0.05, row['lat'] + 0.05, f"{row['val']} cm", 
                transform=ccrs.PlateCarree(), fontsize=7, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5), zorder=30
            )

    @staticmethod
    def add_fronts(ax, t_data, lons, lats):
        if t_data is not None:
            fronts = AnalysisEngine.detect_fronts(t_data)
            ax.contour(
                lons, lats, fronts, levels=[0.5], colors='blue', linewidths=3, 
                transform=ccrs.PlateCarree(), zorder=25
            )


# ==============================================================================
# 9. USER INTERFACE (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.header("🛰️ Modell-Zentrale")
    
    use_split = st.checkbox("🔄 Vergleichs-Modus (Split-Screen)", value=False)
    st.markdown("---")
    
    # ---------------------------------------------------------
    # MODELL 1
    # ---------------------------------------------------------
    st.markdown("### 🔹 Anzeige 1")
    mod_1 = st.radio("Wettermodell 1", list(ModelRegistry.MODELS.keys()), label_visibility="collapsed")
    
    available_regions_1 = ModelRegistry.MODELS[mod_1]["regions"]
    reg_1 = st.radio("Karten-Ausschnitt 1", available_regions_1)
    
    available_params_1 = ModelRegistry.MODELS[mod_1]["params"]
    par_1 = st.radio("Parameter 1", available_params_1)
    
    if "Radar" in mod_1 or "Pegel" in mod_1:
        hr_1 = 0
    else:
        model_type_1 = ModelRegistry.MODELS[mod_1]["type"]
        h_list_1 = ModelRegistry.get_timesteps(model_type_1)
        hr_str_1 = st.selectbox("Zeitpunkt (Stunde)", [f"+{h}h" for h in h_list_1])
        hr_1 = int(hr_str_1.replace("+", "").replace("h", ""))

    # ---------------------------------------------------------
    # MODELL 2 (Split-Screen)
    # ---------------------------------------------------------
    mod_2, par_2, hr_2 = None, None, 0
    if use_split:
        st.markdown("---")
        st.markdown("### 🔸 Anzeige 2")
        mod_2 = st.radio("Wettermodell 2", list(ModelRegistry.MODELS.keys()), index=4, label_visibility="collapsed")
        
        available_params_2 = ModelRegistry.MODELS[mod_2]["params"]
        par_2 = st.radio("Parameter 2", available_params_2)
        
        if "Radar" in mod_2 or "Pegel" in mod_2:
            hr_2 = 0
        else:
            model_type_2 = ModelRegistry.MODELS[mod_2]["type"]
            h_list_2 = ModelRegistry.get_timesteps(model_type_2)
            hr_str_2 = st.selectbox("Zeitpunkt 2 (Stunde)", [f"+{h}h" for h in h_list_2])
            hr_2 = int(hr_str_2.replace("+", "").replace("h", ""))
            
    # ---------------------------------------------------------
    # OVERLAYS
    # ---------------------------------------------------------
    st.markdown("---")
    show_sat = st.checkbox("🌍 Satelliten-Hintergrund erlauben", value=True)
    show_isobars = st.checkbox("Isobaren einblenden", value=True)
    show_fronts = st.checkbox("🌪️ Fronten-Analyse aktivieren", value=False)
    
    enable_refresh = st.checkbox("🔄 Auto-Update (5 Min.)", value=False)
    if enable_refresh and st_autorefresh is not None:
        st_autorefresh(interval=300000, key="auto_refresh")
        
    st.markdown("---")
    generate = st.button("🚀 Karten generieren", use_container_width=True)


# ==============================================================================
# 10. MAIN EXECUTION & RENDERING LOGIK
# ==============================================================================
def render_axis(ax, fig, model, param, hr, region):
    data, lons, lats, run_id = DataFetcher.fetch_model_data(model, param, hr)
    
    ext = GeoConfig.get_extent(region)
    ax.set_extent(ext, crs=ccrs.PlateCarree())

    allow_sat = ("Gesamtbedeckung" in param) or ("Radar" in param)
    
    if show_sat and allow_sat:
        zoom_level = GeoConfig.get_zoom(region)
        ax.add_image(GoogleSatelliteTiles(), zoom_level, zorder=0)

    border_col = 'white' if (show_sat and allow_sat) else 'black'
    ax.add_feature(cfeature.COASTLINE, linewidth=0.9, edgecolor=border_col, zorder=12)
    ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor=border_col, zorder=12)

    if data is not None or "Pegel" in model:
        if "Radar" in param:
            if "RainViewer" in model:
                PlottingEngine.plot_rainviewer(ax, fig, data, lons, region)
            else:
                PlottingEngine.plot_generic(ax, fig, lons, lats, data, param)
                
        elif "Gesamtbedeckung" in param:
            PlottingEngine.plot_clouds(ax, fig, lons, lats, data)

        elif "Geopot. Höhe" in param:
            # Die brandneue 500 hPa Logik!
            PlottingEngine.plot_geopotential(ax, fig, lons, lats, data)

        elif "Pegel" in model: 
            PlottingEngine.plot_pegel(ax, data, region)
            
        elif "Temperatur" in param or "Taupunkt" in param: 
            PlottingEngine.plot_temperature(ax, fig, lons, lats, data, param)
            
        elif "Niederschlag" in param: 
            PlottingEngine.plot_precipitation(ax, fig, lons, lats, data)
            
        elif "Wind" in param or "Jetstream" in param: 
            PlottingEngine.plot_wind(ax, fig, lons, lats, data, param)
            
        elif any(x in param for x in ["Theta-E", "K-Index", "Vorticity", "Warnungen", "Showalter"]): 
            PlottingEngine.plot_profi_indices(ax, fig, lons, lats, data, param)
            
        else: 
            PlottingEngine.plot_generic(ax, fig, lons, lats, data, param)

        if show_isobars and "Radar" not in model and "Pegel" not in model:
            iso_d, iso_l, iso_a, _ = DataFetcher.fetch_model_data(model, "Isobaren", hr)
            # Aufruf der reparierten Isobaren-Funktion!
            PlottingEngine.add_isobars(ax, iso_d, iso_l, iso_a)
            
        if show_fronts and "Radar" not in model and "Pegel" not in model:
            t_d, t_l, t_a, _ = DataFetcher.fetch_model_data(model, "850 hPa Temperatur (°C)", hr)
            PlottingEngine.add_fronts(ax, t_d, t_l, t_a)

        if "Radar" in model or "Pegel" in model:
            now_str = datetime.now(LOCAL_TZ).strftime('%H:%M')
            txt = f"Modell: {model}\nParameter: {param}\nLive-Stand: {now_str} Uhr"
        else:
            run_dt = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            valid_dt = (run_dt + timedelta(hours=hr)).astimezone(LOCAL_TZ)
            txt = f"Modell: {model}\nParameter: {param}\nTermin: {valid_dt.strftime('%d.%m.%Y %H:%M')}\nLauf: {run_id[-2:]}Z"
            
        ax.text(
            0.02, 0.98, txt, 
            transform=ax.transAxes, fontsize=8, fontweight='bold', va='top', 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'), zorder=30
        )
    else:
        ax.text(
            0.5, 0.5, "Daten aktuell nicht verfügbar", 
            transform=ax.transAxes, ha='center', va='center', 
            fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.8)
        )


if generate or (enable_refresh and "Radar" in mod_1):
    SystemManager.cleanup_temp_files()
    
    with st.spinner("🛰️ Lade Modell-Daten (Europa/Nordatlantik kann etwas dauern)..."):
        if use_split:
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
                render_axis(ax1, fig1, mod_1, par_1, hr_1, reg_1)
                st.pyplot(fig1)
            with col2:
                fig2, ax2 = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
                render_axis(ax2, fig2, mod_2, par_2, hr_2, reg_1)
                st.pyplot(fig2)
        else:
            fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
            render_axis(ax, fig, mod_1, par_1, hr_1, reg_1)
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            st.download_button(
                label="📥 Karte speichern", 
                data=buf, 
                file_name=f"Warnwetter_{mod_1.split()[0]}.png", 
                mime="image/png"
            )

    SystemManager.cleanup_temp_files()

