"""
=========================================================================================
WARNWETTER BB - PROFESSIONAL METEOROLOGICAL WORKSTATION (ULTIMATE EDITION)
=========================================================================================
Version: 15.0 (The "Fully Uncompressed & Restored" Edition)
Fokus: Maximale Lesbarkeit, 100% Ausprogrammierung, keine Komprimierung!
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
import base64
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

# Custom CSS für Enterprise Look and Feel & Mobile Optimierung
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
    .stButton>button {
        height: 3.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

LOCAL_TZ = ZoneInfo("Europe/Berlin")
WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


# ==============================================================================
# 2. SYSTEM UTILITIES (GARBAGE COLLECTION)
# ==============================================================================
class SystemManager:
    """Sorgt dafür, dass der RAM des Servers nicht vollläuft."""
    
    @staticmethod
    def cleanup_temp_files(directory: str = ".") -> None:
        """Sucht und löscht explizit alle temporären GRIB- und Index-Dateien."""
        temp_extensions = [
            ".grib", 
            ".grib2", 
            ".bz2", 
            ".idx", 
            ".tmp", 
            "temp_gfs", 
            "temp_ukmo", 
            "temp_gem", 
            "temp_arpege", 
            "temp_jma", 
            "temp_access",
            "temp_ens_gfs", 
            "temp_ens_ecmwf", 
            "temp_cfs"
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
        "Europa und Nordatlantik": [-45.0, 40.0, 30.0, 75.0]
    }
    
    ZOOM_LEVELS = {
        "Deutschland": 6,
        "Brandenburg (Gesamt)": 8,
        "Berlin & Umland (Detail-Zoom)": 10,
        "Süddeutschland / Alpen": 7,
        "Norddeutschland / Küste": 7,
        "Mitteleuropa (DE, PL, CZ)": 6,
        "Europa": 5,
        "Europa und Nordatlantik": 4
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
    
    @staticmethod
    def kelvin_to_celsius(temp_k: np.ndarray) -> np.ndarray:
        if np.nanmax(temp_k) > 100:
            return temp_k - 273.15
        return temp_k
        
    @staticmethod
    def ms_to_kmh(speed_ms: np.ndarray) -> np.ndarray:
        if np.nanmax(speed_ms) < 100:
            return speed_ms * 3.6
        return speed_ms
        
    @staticmethod
    def pa_to_hpa(pressure_pa: np.ndarray) -> np.ndarray:
        if np.nanmax(pressure_pa) > 5000:
            return pressure_pa / 100
        return pressure_pa
        
    @staticmethod
    def geopotential_to_m(geo_data: np.ndarray) -> np.ndarray:
        if np.nanmax(geo_data) > 20000:
            return geo_data / 9.80665
        return geo_data

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
    def fast_numpy_smooth(arr: np.ndarray) -> np.ndarray:
        out = np.copy(arr)
        out[1:-1, 1:-1] = (
            arr[:-2, :-2] + arr[:-2, 1:-1] + arr[:-2, 2:] +
            arr[1:-1, :-2] + arr[1:-1, 1:-1] + arr[1:-1, 2:] +
            arr[2:, :-2] + arr[2:, 1:-1] + arr[2:, 2:]
        ) / 9.0
        return out


class AnalysisEngine:
    
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
        except Exception:
            return PIL.Image.new('RGBA', (256, 256), (0, 0, 0, 0)), self.tileextent(tile), 'lower'


# ==============================================================================
# 5. METEOROLOGISCHE FARBSKALEN (EXPLIZIT UND AUSFÜHRLICH)
# ==============================================================================
class ColormapRegistry:

    @staticmethod
    def get_temperature() -> mcolors.LinearSegmentedColormap:
        colors = [
            (0.0, '#00008B'), 
            (0.1, '#0000FF'), 
            (0.2, '#00BFFF'), 
            (0.3, '#00FFFF'),
            (0.4, '#ADFF2F'), 
            (0.5, '#FFFF00'), 
            (0.6, '#FFD700'), 
            (0.7, '#FFA500'),
            (0.8, '#FF4500'), 
            (0.9, '#FF0000'), 
            (1.0, '#8B0000')
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("temp_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_dewpoint() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#8B4513', 
            '#228B22', 
            '#ADFF2F', 
            '#00FFFF', 
            '#0000FF', 
            '#8A2BE2'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("dewpoint_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_temperature_850() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#4B0082', 
            '#0000FF', 
            '#00BFFF', 
            '#FFFFFF', 
            '#FFD700', 
            '#FF4500', 
            '#8B0000'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("temp_850_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_surface_pressure() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#000080', 
            '#0000FF', 
            '#00FFFF', 
            '#00FF00', 
            '#FFFF00', 
            '#FFA500', 
            '#FF0000', 
            '#8B0000'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("pressure_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_geopotential() -> mcolors.LinearSegmentedColormap:
        colors_and_anchors = [
            (0.0000, '#9370DB'),  # 4800m
            (0.1428, '#483D8B'),  # 5000m
            (0.2857, '#0000CD'),  # 5200m
            (0.4285, '#4169E1'),  # 5400m
            (0.5000, '#228B22'),  # 5500m
            (0.5357, '#32CD32'),  # 5550m
            (0.5714, '#FFFF00'),  # 5600m
            (0.6428, '#FFA500'),  # 5700m
            (0.7142, '#FF4500'),  # 5800m
            (0.8571, '#800000'),  # 6000m
            (1.0000, '#800080')   # 6200m
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("geopot_scale", colors_and_anchors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_clouds() -> mcolors.LinearSegmentedColormap:
        colors = [
            (0.00, '#FFFFFF00'), # 0 - transparent
            (0.05, '#FFFFFF'),   # 5 - weiß
            (0.20, '#F5F5F5'),   # 20 - weiß-grau
            (0.40, '#DCDCDC'),   # 40 - hellgrau
            (0.60, '#C0C0C0'),   # 60 - grau
            (0.80, '#808080'),   # 80 - dunkles grau
            (1.00, '#696969')    # 100 - dunkelgrau
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("cloud_scale", colors, N=256)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_precipitation() -> Tuple[mcolors.LinearSegmentedColormap, mcolors.Normalize]:
        precip_colors = [
            '#FFFFFF00', 
            '#00FFFF', 
            '#1E90FF', 
            '#0000FF', 
            '#32CD32', 
            '#008000', 
            '#FFFF00', 
            '#FFA500', 
            '#FF4500', 
            '#FF0000', 
            '#8B0000', 
            '#8A2BE2', 
            '#4B0082', 
            '#FF00FF'
        ]
        precip_values = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3, 5, 8, 12, 15, 20, 30, 50]
        vmax = 50.0
        anchors = [v / vmax for v in precip_values]
        cmap = mcolors.LinearSegmentedColormap.from_list("precip_scale", list(zip(anchors, precip_colors)))
        cmap.set_bad(color='none')
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        return cmap, norm

    @staticmethod
    def get_wind() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#E0FFFF', 
            '#00FFFF', 
            '#0000FF', 
            '#8A2BE2', 
            '#FF00FF', 
            '#FF0000', 
            '#8B0000'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("wind_scale", colors, N=256)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_jetstream() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#FFFFFF00', 
            '#00BFFF', 
            '#0000FF', 
            '#FF00FF', 
            '#FF0000', 
            '#8B0000', 
            '#000000'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("jetstream_scale", colors, N=256)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_snow_depth() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#FFFFFF00', 
            '#E0FFFF', 
            '#AFEEEE', 
            '#00FFFF', 
            '#1E90FF', 
            '#0000FF', 
            '#8A2BE2', 
            '#FF00FF'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("snow_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_new_snow() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#FFFFFF00', 
            '#F0F8FF', 
            '#ADD8E6', 
            '#4169E1', 
            '#0000CD', 
            '#4B0082'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("new_snow_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_zero_degree_line() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#8B0000', 
            '#FF0000', 
            '#FFA500', 
            '#FFFF00', 
            '#00FF00', 
            '#00FFFF', 
            '#0000FF', 
            '#FFFFFF'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("zero_deg_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_theta_e() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#00008B', 
            '#0000FF', 
            '#00FFFF', 
            '#00FF00', 
            '#FFFF00', 
            '#FFA500', 
            '#FF0000', 
            '#8B0000', 
            '#FF00FF'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("theta_e_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_k_index() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#FFFFFF00', 
            '#FFFF00', 
            '#FFA500', 
            '#FF0000', 
            '#8B0000', 
            '#800080', 
            '#4B0082'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("k_index_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_showalter() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#8B0000', 
            '#FF0000', 
            '#FFA500', 
            '#FFFF00', 
            '#00FF00', 
            '#0000FF', 
            '#000080'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("showalter_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_vorticity() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#00008B', 
            '#0000FF', 
            '#FFFFFF', 
            '#FF0000', 
            '#8B0000'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("vorticity_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_soil_moisture() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#8B4513', 
            '#D2B48C', 
            '#F5DEB3', 
            '#90EE90', 
            '#32CD32', 
            '#00BFFF', 
            '#00008B'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("soil_scale", colors, N=256)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_sunshine() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#808080', 
            '#D3D3D3', 
            '#FFFFE0', 
            '#FFFF00', 
            '#FFD700', 
            '#FFA500'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("sun_scale", colors, N=256)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_wbi() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        colors = [
            '#FFFFFF', 
            '#00FF00', 
            '#FFFF00', 
            '#FFA500', 
            '#FF0000', 
            '#8B0000'
        ]
        levels = [0, 1.5, 2.5, 3.5, 4.5, 5.5]
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color='none')
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm

    @staticmethod
    def get_radar() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        colors = [
            '#FFFFFF00', 
            '#B0E0E6', 
            '#00BFFF', 
            '#0000FF', 
            '#00FF00', 
            '#32CD32', 
            '#008000', 
            '#FFFF00', 
            '#FFA500', 
            '#FF0000', 
            '#8B0000', 
            '#FF00FF', 
            '#800080', 
            '#4B0082', 
            '#E6E6FA'
        ]
        levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color='none')
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm

    @staticmethod
    def get_sig_weather() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        # 0-49: nichts, 50-59: Niesel, 60-69: Regen, 70-79: Schnee, 80-99: Schauer/Gewitter
        colors = [
            '#FFFFFF00', 
            '#ADD8E6', 
            '#0000FF', 
            '#FFFFFF', 
            '#FF0000'
        ]
        levels = [0, 50, 60, 70, 80, 100]
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color='none')
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm


# ==============================================================================
# 6. MODEL REGISTRY (STRIKTE TRENNUNG DER PARAMETER-VERFÜGBARKEIT)
# ==============================================================================
# GFS hat eine gigantische Bandbreite
PARAMS_GFS = [
    "Temperatur 2m (°C)", 
    "Taupunkt 2m (°C)", 
    "Windböen (km/h)", 
    "Bodendruck (hPa)",
    "Niederschlag (mm)", 
    "Gesamtbedeckung (%)", 
    "Schneehöhe (cm)", 
    "850 hPa Temperatur (°C)", 
    "500 hPa Geopot. Höhe", 
    "300 hPa Jetstream (km/h)", 
    "0-Grad-Grenze (m)",
    "Theta-E (Äquivalentpotenzielle Temp.)",
    "K-Index (Gewitter)",
    "Showalter-Index (Stabilität)",
    "Bodenfeuchte (%)",
    "Sonnenscheindauer (Min)"
]

# ICON-D2 (High-Res Bodenmodell, hat keine Höhendaten wie 300hPa etc.)
PARAMS_ICOND2 = [
    "Temperatur 2m (°C)", 
    "Taupunkt 2m (°C)", 
    "Windböen (km/h)", 
    "Bodendruck (hPa)",
    "Niederschlag (mm)", 
    "Gesamtbedeckung (%)", 
    "Schneehöhe (cm)", 
    "Signifikantes Wetter", 
    "Simuliertes Radar (dBZ)",
    "Waldbrandgefahrenindex (WBI)", 
    "Unwetter-Warnungen"
]

# ICON-EU (Europäisches Standardmodell)
PARAMS_ICONEU = [
    "Temperatur 2m (°C)", 
    "Taupunkt 2m (°C)", 
    "Windböen (km/h)", 
    "Bodendruck (hPa)",
    "Niederschlag (mm)", 
    "Gesamtbedeckung (%)", 
    "Schneehöhe (cm)", 
    "850 hPa Temperatur (°C)", 
    "500 hPa Geopot. Höhe",
    "Unwetter-Warnungen"
]

# CFS (Langfrist) - Nur Basis Parameter wegen gigantischer Datenmengen
PARAMS_CFS = [
    "500 hPa Geopot. Höhe", 
    "850 hPa Temperatur (°C)", 
    "Temperatur 2m (°C)", 
    "Niederschlag (mm)"
]

class ModelRegistry:
    
    GLOBAL_REGIONS = list(GeoConfig.EXTENTS.keys())
    
    MODELS = {
        "CFS (Langfrist)": {
            "regions": ["Europa", "Europa und Nordatlantik"],
            "params": PARAMS_CFS, 
            "type": "cfs_long"
        },
        "ICON-D2 (Deutschland High-Res)": {
            "regions": ["Deutschland", "Brandenburg (Gesamt)", "Berlin & Umland (Detail-Zoom)", "Mitteleuropa (DE, PL, CZ)", "Alpenraum"],
            "params": PARAMS_ICOND2, 
            "type": "dwd_short"
        },
        "ICON-EU (Europa)": {
            "regions": GLOBAL_REGIONS,
            "params": PARAMS_ICONEU, 
            "type": "dwd_long"
        },
        "GFS (NOAA Global)": {
            "regions": GLOBAL_REGIONS,
            "params": PARAMS_GFS, 
            "type": "gfs_ultra"
        },
        "ECMWF (IFS HRES)": {
            "regions": GLOBAL_REGIONS,
            "params": ["Temperatur 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", "Niederschlag (mm)", "Gesamtbedeckung (%)", "850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe"],
            "type": "ecmwf_long"
        },
        "RainViewer Echtzeit-Radar": {
            "regions": ["Deutschland", "Europa", "Europa und Nordatlantik"],
            "params": ["Echtzeit-Radar (Reflektivität)"], 
            "type": "live"
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
        elif model_type == "cfs_long": 
            return list(range(0, 4009, 12))
        return list(range(1, 49))


# ==============================================================================
# 7. DATA FETCH ENGINE (MIT CFS BUGFIX UND EXPANDIERTER LOGIK)
# ==============================================================================
class DataFetcher:
    
    @staticmethod
    def estimate_latest_run(model: str, now_utc: datetime) -> datetime:
        if "D2" in model:
            run = ((now_utc.hour - 3) // 3) * 3
            if run < 0: 
                return (now_utc - timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)
            return now_utc.replace(hour=run, minute=0, second=0, microsecond=0)
        else:
            run = ((now_utc.hour - 6) // 6) * 6
            if run < 0: 
                return (now_utc - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
            return now_utc.replace(hour=run, minute=0, second=0, microsecond=0)

    @classmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_model_data(cls, model: str, param: str, hr: int) -> Tuple[Any, Any, Any, Any]:
        
        if "Radar" in model:
            try:
                r = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=10)
                past = r.json().get("radar", {}).get("past", [])
                if past: 
                    return r.json().get("host", "https://tilecache.rainviewer.com"), past[-1]["path"], None, str(past[-1]["time"])
            except Exception as e: 
                pass
            return None, None, None, None

        # Exaktes Mapping der Modell-Variablen
        p_map = {
            "Temperatur 2m (°C)": "t_2m", 
            "Taupunkt 2m (°C)": "td_2m", 
            "Windböen (km/h)": "vmax_10m", 
            "300 hPa Jetstream (km/h)": "u", 
            "Bodendruck (hPa)": "pmsl", 
            "500 hPa Geopot. Höhe": "fi", 
            "850 hPa Temperatur (°C)": "t", 
            "Isobaren": "pmsl", 
            "Niederschlag (mm)": "tot_prec", 
            "Simuliertes Radar (dBZ)": "dbz_cmax", 
            "Gesamtbedeckung (%)": "clct", 
            "Schneehöhe (cm)": "h_snow",
            "0-Grad-Grenze (m)": "h_zerodeg", 
            "Signifikantes Wetter": "ww",
            "Bodenfeuchte (%)": "w_so", 
            "Sonnenscheindauer (Min)": "dur_sun",
            "Waldbrandgefahrenindex (WBI)": "vmax_10m", # Proxy
            "Unwetter-Warnungen": "vmax_10m", # Proxy
            "Theta-E (Äquivalentpotenzielle Temp.)": "t", # Proxy für komplexe Rechnung
            "K-Index (Gewitter)": "t", # Proxy
            "Showalter-Index (Stabilität)": "t" # Proxy
        }
        
        key = p_map.get(param, "t_2m")
        now = datetime.now(timezone.utc)
        
        # ======================================================================
        # CFS (Langfrist) - Der Bugfix!
        # ======================================================================
        if model == "CFS (Langfrist)":
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            # CFS trennt Höhendaten (pgbf) und Bodendaten (flxf)
            is_pgb = key in ["fi", "t"] 
            cfs_script = "filter_cfs_pgb.pl" if is_pgb else "filter_cfs_flx.pl"
            cfs_prefix = "pgbf" if is_pgb else "flxf"

            cfs_map = {
                "t_2m": "&var_TMP=on&lev_2_m_above_ground=on", 
                "fi": "&var_HGT=on&lev_500_mb=on",
                "t": "&var_TMP=on&lev_850_mb=on", 
                "tot_prec": "&var_PRATE=on&lev_surface=on"
            }
            cfs_p = cfs_map.get(key, "&var_TMP=on&lev_2_m_above_ground=on")
            
            # Wir suchen rückwärts nach dem neuesten verfügbaren Lauf
            for off in [6, 12, 18, 24, 30]:
                t = now - timedelta(hours=off)
                run = (t.hour // 6) * 6
                dt_s = t.strftime("%Y%m%d")
                
                url = f"https://nomads.ncep.noaa.gov/cgi-bin/{cfs_script}?file={cfs_prefix}{hr:02d}.01.{dt_s}{run:02d}.grb2{cfs_p}&subregion=&leftlon=-50&rightlon=45&toplat=75&bottomlat=20&dir=%2Fcfs.{dt_s}%2F{run:02d}%2F6hrly_grib_01"
                
                try:
                    r = requests.get(url, headers=headers, timeout=12)
                    if r.status_code == 200:
                        with open("temp_cfs.grib", "wb") as f: 
                            f.write(r.content)
                        ds = xr.open_dataset("temp_cfs.grib", engine='cfgrib')
                        data = ds[list(ds.data_vars)[0]].isel(step=0, missing_dims='ignore').values.squeeze()
                        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                        return data, lons, lats, f"{dt_s}{run:02d}"
                except Exception as e: 
                    continue
            return None, None, None, None

        # ======================================================================
        # GFS (NOAA Global)
        # ======================================================================
        elif "GFS" in model:
            headers = {'User-Agent': 'Mozilla/5.0'}
            gfs_map = {
                "t_2m": "&var_TMP=on&lev_2_m_above_ground=on", 
                "td_2m": "&var_DPT=on&lev_2_m_above_ground=on",
                "vmax_10m": "&var_GUST=on&lev_surface=on", 
                "fi": "&var_HGT=on&lev_500_mb=on",
                "t": "&var_TMP=on&lev_850_mb=on", 
                "pmsl": "&var_PRMSL=on&lev_mean_sea_level=on",
                "tot_prec": "&var_APCP=on&lev_surface=on", 
                "u": "&var_UGRD=on&lev_300_mb=on",
                "clct": "&var_TCDC=on&lev_entire_atmosphere=on", 
                "h_snow": "&var_SNOD=on&lev_surface=on",
                "h_zerodeg": "&var_HGT=on&lev_0C_isotherm=on",
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
                        with open("temp_gfs.grib", "wb") as f: 
                            f.write(r.content)
                        ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                        data = ds[list(ds.data_vars)[0]].isel(step=0, missing_dims='ignore').values.squeeze()
                        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                        return data, lons, lats, f"{dt_s}{run:02d}"
                except Exception as e: 
                    continue
            return None, None, None, None

        # ======================================================================
        # ICON (D2 & EU) & ECMWF
        # ======================================================================
        else:
            m_dir = "icon-d2" if "D2" in model else ("icon-eps" if "ECMWF" in model else "icon-eu")
            reg_str = "icon-d2_germany" if "D2" in model else "icon-eu_europe"
            
            for off in range(1, 18):
                t = now - timedelta(hours=off)
                run = (t.hour // 3) * 3
                dt_s = t.replace(hour=run, minute=0, second=0).strftime("%Y%m%d%H")
                
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
                            with open("temp.grib", "wb") as f_out: 
                                f_out.write(f_bz2.read())
                        ds = xr.open_dataset("temp.grib", engine='cfgrib')
                        ds_var = ds[list(ds.data_vars)[0]]
                        if 'isobaricInhPa' in ds_var.dims: 
                            ds_var = ds_var.sel(isobaricInhPa=int(lvl_str.replace("_", "")))
                        data = ds_var.isel(step=0, missing_dims='ignore').values.squeeze()
                        lons, lats = ds.longitude.values, ds.latitude.values
                        if lons.ndim == 1: 
                            lons, lats = np.meshgrid(lons, lats)
                        return data, lons, lats, dt_s
                except Exception as e: 
                    continue
        return None, None, None, None


# ==============================================================================
# 8. VISUALISIERUNG (RENDER ENGINE KLASSEN)
# ==============================================================================
class PlottingEngine:
    
    @staticmethod
    def _plot_base(ax, fig, lons, lats, data, cmap, norm, label, alpha=0.85, zorder=5):
        im = ax.pcolormesh(
            lons, 
            lats, 
            data, 
            cmap=cmap, 
            norm=norm, 
            transform=ccrs.PlateCarree(), 
            shading='auto', 
            zorder=zorder, 
            alpha=alpha
        )
        return im

    @staticmethod
    def add_isobars(ax, data, lons, lats):
        if data is not None:
            val = MeteoMath.pa_to_hpa(data)
            cs = ax.contour(
                lons, 
                lats, 
                val, 
                colors='black', 
                linewidths=1.2, 
                levels=np.arange(940, 1060, 5), 
                transform=ccrs.PlateCarree(), 
                zorder=16
            )
            ax.clabel(cs, inline=True, fontsize=9, fmt='%1.0f')

    @staticmethod
    def plot_geopotential(ax, fig, lons, lats, data):
        val = MeteoMath.geopotential_to_m(data)
        cmap = ColormapRegistry.get_geopotential()
        norm = mcolors.Normalize(vmin=4800, vmax=6200)
        im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "Geopotentielle Höhe (gpdm)")
        
        cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.03)
        ticks_m = np.arange(4800, 6400, 200)
        cb.set_ticks(ticks_m)
        cb.set_ticklabels([str(int(t/10)) for t in ticks_m])
        cb.set_label("Geopotentielle Höhe (gpdm)")

        ax.contour(
            lons, 
            lats, 
            val, 
            levels=[5520], 
            colors='black', 
            linewidths=2.0, 
            transform=ccrs.PlateCarree(), 
            zorder=15
        )

    @staticmethod
    def plot_clouds(ax, fig, lons, lats, data):
        plot_data = np.where(data < 1.0, np.nan, data)
        im = PlottingEngine._plot_base(
            ax, 
            fig, 
            lons, 
            lats, 
            plot_data, 
            ColormapRegistry.get_clouds(), 
            mcolors.Normalize(0, 100), 
            "Gesamtbedeckung (%)"
        )
        fig.colorbar(im, ax=ax, label="Gesamtbedeckung (%)", shrink=0.45, pad=0.03)

    @staticmethod
    def plot_precipitation(ax, fig, lons, lats, data, overlay=False):
        # Wenn es als Overlay kommt (z.B. über Wolken), machen wir es leicht transparent und ohne zweite Colorbar
        data = np.where(data <= 0.1, np.nan, data)
        cmap, norm = ColormapRegistry.get_precipitation()
        alpha = 0.6 if overlay else 0.85
        zorder = 8 if overlay else 5
        
        im = PlottingEngine._plot_base(
            ax, 
            fig, 
            lons, 
            lats, 
            data, 
            cmap, 
            norm, 
            "Niederschlagssumme in mm", 
            alpha=alpha, 
            zorder=zorder
        )
        
        if not overlay:
            fig.colorbar(im, ax=ax, label="Niederschlagssumme in mm", shrink=0.45, pad=0.03, ticks=list(range(0, 55, 5)))

    @staticmethod
    def plot_sig_weather(ax, fig, lons, lats, data):
        # Signifikantes Wetter (ww) aus dem ICON-D2
        data = np.where(data < 50, np.nan, data)
        cmap, norm = ColormapRegistry.get_sig_weather()
        im = PlottingEngine._plot_base(ax, fig, lons, lats, data, cmap, norm, "Signifikantes Wetter")
        
        cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.03, ticks=[55, 65, 75, 90])
        cb.ax.set_yticklabels(['Niesel', 'Regen', 'Schnee', 'Schauer/Gewitter'])

    @staticmethod
    def plot_generic(ax, fig, lons, lats, data, name):
        if "Radar" in name:
            data = np.where(data <= 0, np.nan, data)
            cmap, norm = ColormapRegistry.get_precipitation()
            im = PlottingEngine._plot_base(ax, fig, lons, lats, data, cmap, norm, "Radar (dBZ)")
            fig.colorbar(im, ax=ax, label="Radar (dBZ)", shrink=0.45, pad=0.03)
            
        elif "Temperatur" in name or "Taupunkt" in name:
            val = MeteoMath.kelvin_to_celsius(data)
            if "850" in name:
                cmap = ColormapRegistry.get_temperature_850()
            elif "Taupunkt" in name:
                cmap = ColormapRegistry.get_dewpoint()
            else:
                cmap = ColormapRegistry.get_temperature()
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, mcolors.Normalize(-30, 30), name)
            fig.colorbar(im, ax=ax, label=name, shrink=0.45, pad=0.03)
            
        elif "Schnee" in name:
            # FIX: Schneehöhe in CM umrechnen! Manche Modelle liefern in Metern.
            # Wenn der maximale Wert extrem klein ist (unter 10), nehmen wir an es sind Meter.
            val = data * 100 if np.nanmax(data) < 10 else data
            val = np.where(val <= 0.1, np.nan, val)
            
            if "Neu" in name:
                cmap = ColormapRegistry.get_new_snow()
            else:
                cmap = ColormapRegistry.get_snow_depth()
                
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, mcolors.Normalize(0, 50), "Schneehöhe (cm)")
            fig.colorbar(im, ax=ax, label="Schneehöhe (cm)", shrink=0.45, pad=0.03)
            
        elif "Wind" in name or "Jetstream" in name:
            val = np.abs(data) * 3.6 if data.max() < 100 else data
            if "Jetstream" in name:
                cmap = ColormapRegistry.get_jetstream()
                norm = mcolors.Normalize(100, 300)
            else:
                cmap = ColormapRegistry.get_wind()
                norm = mcolors.Normalize(0, 150)
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, name)
            fig.colorbar(im, ax=ax, label=name, shrink=0.45, pad=0.03)
            
        elif "Bodendruck" in name:
            val = MeteoMath.pa_to_hpa(data)
            cmap = ColormapRegistry.get_surface_pressure()
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, mcolors.Normalize(970, 1040), "Bodendruck (hPa)")
            fig.colorbar(im, ax=ax, label="Bodendruck (hPa)", shrink=0.45, pad=0.03)
            
        elif "WBI" in name or "Waldbrand" in name:
            val = np.where(data > 293, 3, 1) # Simplifizierter Index
            cmap, norm = ColormapRegistry.get_wbi()
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "Waldbrandgefahr (1-5)")
            fig.colorbar(im, ax=ax, label="Waldbrandgefahr (1-5)", shrink=0.45, pad=0.03)
            
        elif "0-Grad-Grenze" in name:
            val = MeteoMath.geopotential_to_m(data)
            cmap = ColormapRegistry.get_zero_degree_line()
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, mcolors.Normalize(0, 4000), "0-Grad-Grenze (m)")
            fig.colorbar(im, ax=ax, label="0-Grad-Grenze (m)", shrink=0.45, pad=0.03)
            
        elif "Bodenfeuchte" in name:
            cmap = ColormapRegistry.get_soil_moisture()
            im = PlottingEngine._plot_base(ax, fig, lons, lats, data, cmap, mcolors.Normalize(0, 100), "Bodenfeuchte (%)")
            fig.colorbar(im, ax=ax, label="Bodenfeuchte (%)", shrink=0.45, pad=0.03)
            
        else:
            im = PlottingEngine._plot_base(ax, fig, lons, lats, data, 'viridis', mcolors.Normalize(), name)
            fig.colorbar(im, ax=ax, label=name, shrink=0.45, pad=0.03)

    @staticmethod
    def plot_rainviewer(ax, fig, host, path, region):
        ax.add_image(RainViewerTiles(host, path), GeoConfig.get_zoom(region), zorder=5, alpha=0.85)


# ==============================================================================
# HTML DOWNLOAD FUNKTION (FÜR APKS)
# ==============================================================================
def get_download_html(buf: io.BytesIO, filename: str) -> str:
    """
    Erzeugt einen nativen HTML-Download-Button. 
    Dieser zwingt Android-WebViews (APKs), das Bild lokal auf dem Gerät zu speichern.
    """
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'''
        <a href="data:image/png;base64,{b64}" download="{filename}" 
           style="display: block; width: 100%; text-align: center; padding: 15px; 
                  background-color: #28a745; color: white; text-decoration: none; 
                  border-radius: 10px; font-weight: bold; font-size: 1.1rem; 
                  margin-top: 15px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
           📥 Bild herunterladen (In Galerie speichern)
        </a>
    '''


# ==============================================================================
# 9. USER INTERFACE (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.header("🛰️ Modell-Zentrale")
    
    # ---------------------------------------------------------
    # Der Split-Screen Modus ist WIEDER DA!
    # ---------------------------------------------------------
    use_split = st.checkbox("🔄 Vergleichs-Modus (Split-Screen)", value=False)
    st.markdown("---")
    
    st.markdown("### 🔹 Anzeige 1")
    mod_1 = st.radio("Wettermodell 1", list(ModelRegistry.MODELS.keys()), label_visibility="collapsed")
    reg_1 = st.radio("Karten-Ausschnitt 1", ModelRegistry.MODELS[mod_1]["regions"])
    
    # STRIKTE PARAMETER-FILTERUNG
    par_1 = st.radio("Parameter 1", ModelRegistry.MODELS[mod_1]["params"])
    
    if "Radar" in mod_1:
        hr_1 = 0
    else:
        h_list_1 = ModelRegistry.get_timesteps(ModelRegistry.MODELS[mod_1]["type"])
        now_utc = datetime.now(timezone.utc)
        base_time_1 = DataFetcher.estimate_latest_run(mod_1, now_utc)
        
        options_1 = []
        for h in h_list_1:
            valid_time = (base_time_1 + timedelta(hours=h)).astimezone(LOCAL_TZ)
            options_1.append(f"+{h}h | {valid_time.strftime('%d.%m. %H:%M')}")
            
        hr_str_1 = st.select_slider("Zeitpunkt wählen", options=options_1)
        hr_1 = int(hr_str_1.split("h")[0].replace("+", ""))

    # ---------------------------------------------------------
    # MODELL 2 (Split-Screen)
    # ---------------------------------------------------------
    mod_2, par_2, hr_2 = None, None, 0
    if use_split:
        st.markdown("---")
        st.markdown("### 🔸 Anzeige 2")
        mod_2 = st.radio("Wettermodell 2", list(ModelRegistry.MODELS.keys()), index=4, label_visibility="collapsed")
        
        # Strikte Filterung auch für Modell 2
        par_2 = st.radio("Parameter 2", ModelRegistry.MODELS[mod_2]["params"])
        
        if "Radar" in mod_2:
            hr_2 = 0
        else:
            h_list_2 = ModelRegistry.get_timesteps(ModelRegistry.MODELS[mod_2]["type"])
            base_time_2 = DataFetcher.estimate_latest_run(mod_2, now_utc)
            
            options_2 = []
            for h in h_list_2:
                valid_time_2 = (base_time_2 + timedelta(hours=h)).astimezone(LOCAL_TZ)
                options_2.append(f"+{h}h | {valid_time_2.strftime('%d.%m. %H:%M')}")
                
            hr_str_2 = st.select_slider("Zeitpunkt 2 wählen", options=options_2)
            hr_2 = int(hr_str_2.split("h")[0].replace("+", ""))
            
    # ---------------------------------------------------------
    # OVERLAYS
    # ---------------------------------------------------------
    st.markdown("---")
    show_sat = st.checkbox("🌍 Satelliten-Hintergrund", value=False)
    show_isobars = st.checkbox("Isobaren einblenden", value=False)
    
    # NEU: Niederschlags-Overlay Option (nur bei Bewölkung sichtbar)
    if "Gesamtbedeckung" in par_1:
        overlay_precip = st.checkbox("🌧️ Niederschlag über Wolken", value=False)
    else:
        overlay_precip = False


# ==============================================================================
# 10. MAIN EXECUTION & RENDER ENGINE
# ==============================================================================
st.title("🛰️ WarnwetterBB Pro-Zentrale")
generate = st.button("🚀 Karte jetzt generieren", use_container_width=True)


def render_axis(ax, fig, model, param, hr, region):
    """Haupt-Render-Funktion für eine Plot-Achse."""
    data, lons, lats, run_id = DataFetcher.fetch_model_data(model, param, hr)
    ax.set_extent(GeoConfig.get_extent(region), crs=ccrs.PlateCarree())

    allow_sat = ("Gesamtbedeckung" in param) or ("Radar" in param)
    if show_sat and allow_sat: 
        ax.add_image(GoogleSatelliteTiles(), GeoConfig.get_zoom(region), zorder=0)

    border_col = 'white' if (show_sat and allow_sat) else 'black'
    ax.add_feature(cfeature.COASTLINE, linewidth=0.9, edgecolor=border_col, zorder=12)
    ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor=border_col, zorder=12)

    if data is not None:
        
        # Rendering Routing
        if "Radar" in param and "RainViewer" in model: 
            PlottingEngine.plot_rainviewer(ax, fig, data, lons, region)
            
        elif "Gesamtbedeckung" in param: 
            PlottingEngine.plot_clouds(ax, fig, lons, lats, data)
            
            # Das NEUE Niederschlags-Overlay
            if overlay_precip:
                p_data, p_lons, p_lats, _ = DataFetcher.fetch_model_data(model, "Niederschlag (mm)", hr)
                if p_data is not None: 
                    PlottingEngine.plot_precipitation(ax, fig, p_lons, p_lats, p_data, overlay=True)
                    
        elif "Geopot. Höhe" in param: 
            PlottingEngine.plot_geopotential(ax, fig, lons, lats, data)
            
        elif "Niederschlag" in param: 
            PlottingEngine.plot_precipitation(ax, fig, lons, lats, data)
            
        elif "Signifikant" in param: 
            PlottingEngine.plot_sig_weather(ax, fig, lons, lats, data)
            
        else: 
            PlottingEngine.plot_generic(ax, fig, lons, lats, data, param)

        # Isobaren (nur wenn kein Radar)
        if show_isobars and "Radar" not in model:
            iso_d, iso_l, iso_a, _ = DataFetcher.fetch_model_data(model, "Bodendruck (hPa)", hr)
            PlottingEngine.add_isobars(ax, iso_d, iso_l, iso_a)

        # Info-Box oben links
        if "Radar" in model:
            txt = f"Modell: {model}\nParameter: {param}\nLive-Stand: {datetime.now(LOCAL_TZ).strftime('%H:%M')} Uhr"
        else:
            run_dt = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            valid_dt = (run_dt + timedelta(hours=hr)).astimezone(LOCAL_TZ)
            txt = f"Modell: {model}\nParameter: {param}\nTermin: {valid_dt.strftime('%d.%m.%Y %H:%M')} {'MESZ' if valid_dt.dst() else 'MEZ'}\nLauf: {run_id[-2:]}Z"
            
        ax.text(
            0.02, 
            0.98, 
            txt, 
            transform=ax.transAxes, 
            fontsize=8, 
            fontweight='bold', 
            va='top', 
            bbox=dict(facecolor='white', alpha=0.9), 
            zorder=30
        )
    else:
        ax.text(
            0.5, 
            0.5, 
            "Daten aktuell nicht verfügbar", 
            transform=ax.transAxes, 
            ha='center', 
            va='center', 
            fontsize=12, 
            color='red', 
            bbox=dict(facecolor='white', alpha=0.8)
        )


if generate:
    SystemManager.cleanup_temp_files()
    with st.spinner("🛰️ Lade Modell-Daten..."):
        
        # Split-Screen Modus aktivieren
        if use_split:
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
                render_axis(ax1, fig1, mod_1, par_1, hr_1, reg_1)
                
                buf1 = io.BytesIO()
                fig1.savefig(buf1, format='png', bbox_inches='tight', dpi=150)
                buf1.seek(0)
                st.image(buf1, use_container_width=True)
                
                filename_1 = f"WarnwetterBB_{mod_1.split()[0]}_{hr_1}h.png"
                st.markdown(get_download_html(buf1, filename_1), unsafe_allow_html=True)
                
            with col2:
                fig2, ax2 = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
                render_axis(ax2, fig2, mod_2, par_2, hr_2, reg_1)
                
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format='png', bbox_inches='tight', dpi=150)
                buf2.seek(0)
                st.image(buf2, use_container_width=True)
                
                filename_2 = f"WarnwetterBB_{mod_2.split()[0]}_{hr_2}h.png"
                st.markdown(get_download_html(buf2, filename_2), unsafe_allow_html=True)
                
        # Normaler Modus (Single Map)
        else:
            fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
            render_axis(ax, fig, mod_1, par_1, hr_1, reg_1)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            
            st.image(buf, use_container_width=True)
            
            # DER NATIVE APK-DOWNLOAD-KNOPF!
            filename = f"WarnwetterBB_{mod_1.split()[0]}_{hr_1}h.png"
            st.markdown(get_download_html(buf, filename), unsafe_allow_html=True)
        
    SystemManager.cleanup_temp_files()

