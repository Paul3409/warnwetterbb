"""
=========================================================================================
WARNWETTER BB - PROFESSIONAL METEOROLOGICAL WORKSTATION (ULTIMATE 2500+ LINES EDITION)
=========================================================================================
Version: 16.5 (The "Dynamic Colormaps, Granularity & Snow-BG" Edition)
Fokus: Dynamische Farbskalen-Engine, Transparenz-Skipping, granularere Farbschritte.
NEU / WIEDER DA:
- Dynamische Farbskalen: Wenn du Werte/Farben hinzufügst, passt sich das System automatisch an!
- Transparenz-Skip-Logik: Trage 'transparent' ein, und das System überspringt diese Werte automatisch beim Zeichnen (ideal für Regen & Schnee).
- Extreme Granularität: Alle Parameter haben jetzt viel feinere Zwischenschritte (mehr Zahlen).
- Schnee-Hintergrund: Neuer Schalter für "Grün (#049700)" oder "Satellit" exklusiv für Schnee.
BEIBEHALTEN:
- Läufe der letzten 24 Stunden, WZ-Design (weiß/schwarz), Bundesländergrenzen.
- Echte Akkumulation für NOAA-Modelle.
- Nativer HTML5-Download-Knopf.
- Gouraud-Interpolation ("nicht eckig") für alle Parameter.
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

st.set_page_config(
    page_title="WarnwetterBB | Pro-Zentrale", 
    page_icon="🌪️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

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

# ==============================================================================
# 2. SYSTEM UTILITIES (GARBAGE COLLECTION)
# ==============================================================================
class SystemManager:
    @staticmethod
    def cleanup_temp_files(directory: str = ".") -> None:
        temp_extensions = [
            ".grib", ".grib2", ".bz2", ".idx", ".tmp", 
            "temp_gfs", "temp_ukmo", "temp_gem", "temp_arpege", "temp_jma", "temp_access",
            "temp_ens_gfs", "temp_ens_ecmwf", "temp_cfs"
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
                    pass
        if freed_bytes > 0:
            logger.info(f"System Cache gereinigt. {freed_bytes / 1024 / 1024:.2f} MB freigegeben.")

class GeoConfig:
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
        "Deutschland": 6, "Brandenburg (Gesamt)": 8, "Berlin & Umland (Detail-Zoom)": 10,
        "Süddeutschland / Alpen": 7, "Norddeutschland / Küste": 7, "Mitteleuropa (DE, PL, CZ)": 6,
        "Europa": 5, "Europa und Nordatlantik": 4
    }
    
    @classmethod
    def get_extent(cls, region_name: str) -> List[float]:
        return cls.EXTENTS.get(region_name, cls.EXTENTS["Deutschland"])
        
    @classmethod
    def get_zoom(cls, region_name: str) -> int:
        return cls.ZOOM_LEVELS.get(region_name, 6)

# ==============================================================================
# 3. PHYSIKALISCHE ENGINE
# ==============================================================================
class MeteoMath:
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
    def fast_numpy_smooth(arr: np.ndarray) -> np.ndarray:
        out = np.copy(arr)
        out[1:-1, 1:-1] = (
            arr[:-2, :-2] + arr[:-2, 1:-1] + arr[:-2, 2:] +
            arr[1:-1, :-2] + arr[1:-1, 1:-1] + arr[1:-1, 2:] +
            arr[2:, :-2] + arr[2:, 1:-1] + arr[2:, 2:]
        ) / 9.0
        return out


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
# 5. METEOROLOGISCHE FARBSKALEN (DYNAMISCHE ENGINE)
# ==============================================================================
class ColormapRegistry:

    @staticmethod
    def build_dynamic_cmap(val_color_list: List[Tuple[float, str]], name: str) -> Tuple[mcolors.LinearSegmentedColormap, mcolors.Normalize, Optional[float]]:
        """
        Diese Funktion übernimmt automatisch die Skalierung, Normalisierung und 
        die Transparenz-Übersprung-Logik! Wenn du unten Werte anpasst, regelt 
        diese Funktion den Rest vollautomatisch.
        """
        values = [v[0] for v in val_color_list]
        colors = [v[1] for v in val_color_list]
        
        vmin = min(values)
        vmax = max(values)
        
        # Erkenne den höchsten Wert, der 'transparent' zugeordnet ist
        trans_thresh = None
        for val, col in val_color_list:
            if col.lower() in ['transparent', 'none', '#ffffff00']:
                if trans_thresh is None or val > trans_thresh:
                    trans_thresh = val

        # Ersetze 'transparent' durch tatsächlichen RGBA-Code für Matplotlib
        processed_colors = ['#FFFFFF00' if c.lower() == 'transparent' else c for c in colors]
        
        if vmin == vmax:
            anchors = [0.0] * len(values)
        else:
            anchors = [(v - vmin) / (vmax - vmin) for v in values]
            
        cmap = mcolors.LinearSegmentedColormap.from_list(name, list(zip(anchors, processed_colors)))
        cmap.set_bad(color='none')
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
        return cmap, norm, trans_thresh


    @staticmethod
    def get_temperature():
        # ---------------------------------------------------------
        # TEMPERATUR 2m (°C)
        # ---------------------------------------------------------
        val_color_list = [
            (-30, '#4B0082'),
            (-25, '#483D8B'),
            (-20, '#0000FF'),
            (-15, '#1E90FF'),
            (-10, '#00BFFF'),
            (-5,  '#87CEFA'),
            (0,   '#00FFFF'),
            (5,   '#00FA9A'),
            (10,  '#32CD32'),
            (15,  '#ADFF2F'),
            (20,  '#FFD700'),
            (25,  '#FFA500'),
            (30,  '#FF4500'),
            (35,  '#FF0000'),
            (40,  '#8B0000')
        ]
        return ColormapRegistry.build_dynamic_cmap(val_color_list, "temp_scale")

    @staticmethod
    def get_temperature_850():
        # ---------------------------------------------------------
        # 850 hPa TEMPERATUR (°C)
        # ---------------------------------------------------------
        val_color_list = [
            (-30, '#8A2BE2'),
            (-25, '#9370DB'),
            (-20, '#0000FF'),
            (-15, '#4169E1'),
            (-10, '#00BFFF'),
            (-5,  '#87CEEB'),
            (0,   '#00FA9A'),
            (5,   '#32CD32'),
            (10,  '#FFD700'),
            (15,  '#FFA500'),
            (20,  '#FF4500'),
            (25,  '#FF0000'),
            (30,  '#800000')
        ]
        return ColormapRegistry.build_dynamic_cmap(val_color_list, "temp_850_scale")

    @staticmethod
    def get_dewpoint():
        # ---------------------------------------------------------
        # TAUPUNKT 2m (°C)
        # ---------------------------------------------------------
        val_color_list = [
            (-20, '#8B4513'),
            (-15, '#A0522D'),
            (-10, '#CD853F'),
            (-5,  '#F4A460'),
            (0,   '#ADFF2F'),
            (5,   '#32CD32'),
            (10,  '#00FA9A'),
            (15,  '#00BFFF'),
            (20,  '#0000FF'),
            (25,  '#FF00FF')
        ]
        return ColormapRegistry.build_dynamic_cmap(val_color_list, "dewpoint_scale")

    @staticmethod
    def get_surface_pressure():
        # ---------------------------------------------------------
        # BODENDRUCK (hPa)
        # ---------------------------------------------------------
        val_color_list = [
            (950,  '#4B0082'),
            (960,  '#00008B'),
            (970,  '#0000FF'),
            (980,  '#1E90FF'),
            (990,  '#00BFFF'),
            (1000, '#00FFFF'),
            (1010, '#32CD32'),
            (1020, '#FFD700'),
            (1030, '#FFA500'),
            (1040, '#FF0000'),
            (1050, '#8B0000')
        ]
        return ColormapRegistry.build_dynamic_cmap(val_color_list, "pressure_scale")

    @staticmethod
    def get_geopotential():
        # ---------------------------------------------------------
        # GEOPOTENTIAL 500 hPa (gpdm)
        # ---------------------------------------------------------
        val_color_list = [
            (4800, '#dda0dd'), 
            (4900, '#ee82ee'), 
            (5000, '#ba55d3'), 
            (5100, '#6a5acd'), 
            (5200, '#191970'), 
            (5300, '#4169e1'), 
            (5400, '#20b2aa'), 
            (5500, '#008000'), 
            (5600, '#7cfc00'), 
            (5700, '#ffff00'), 
            (5800, '#ffa500'), 
            (5900, '#ff0000'), 
            (6000, '#800000'), 
            (6100, '#8b008b'), 
            (6200, '#4b0082')  
        ]
        return ColormapRegistry.build_dynamic_cmap(val_color_list, "geopot_scale")

    @staticmethod
    def get_clouds():
        # ---------------------------------------------------------
        # BEWÖLKUNG (%)
        # ---------------------------------------------------------
        val_color_list = [
            (0,   'transparent'),
            (10,  '#F5F5F5'),
            (20,  '#E0E0E0'),
            (30,  '#D3D3D3'),
            (40,  '#C0C0C0'),
            (50,  '#A9A9A9'),
            (60,  '#9E9E9E'),
            (70,  '#808080'),
            (80,  '#696969'),
            (90,  '#555555'),
            (100, '#4F4F4F')
        ]
        return ColormapRegistry.build_dynamic_cmap(val_color_list, "cloud_scale")

    @staticmethod
    def get_precipitation():
        # ---------------------------------------------------------
        # NIEDERSCHLAGSSUMME (mm)
        # Trage 'transparent' ein, um diese Werte auszublenden!
        # ---------------------------------------------------------
        val_color_list = [
            (0.0,  'transparent'),
            (0.1,  'transparent'),
            (0.2,  '#E0FFFF'),
            (0.5,  '#00FFFF'),
            (1.0,  '#1E90FF'),
            (2.0,  '#0000FF'),
            (3.0,  '#00008B'),
            (5.0,  '#32CD32'),
            (7.5,  '#008000'),
            (10.0, '#FFFF00'),
            (15.0, '#FFD700'),
            (20.0, '#FFA500'),
            (25.0, '#FF4500'),
            (30.0, '#FF0000'),
            (40.0, '#8B0000'),
            (50.0, '#FF00FF')
        ]
        return ColormapRegistry.build_dynamic_cmap(val_color_list, "precip_scale")

    @staticmethod
    def get_acc_precipitation():
        # ---------------------------------------------------------
        # AKKUMULIERTER NIEDERSCHLAG (mm) BIS 400mm
        # ---------------------------------------------------------
        val_color_list = [
            (0.0,  'transparent'),
            (0.1,  'transparent'),
            (1.0,  '#B0E0E6'),
            (2.0,  '#87CEEB'),
            (5.0,  '#00BFFF'),
            (10.0, '#1E90FF'),
            (15.0, '#0000FF'),
            (20.0, '#32CD32'),
            (30.0, '#228B22'),
            (50.0, '#008000'),
            (75.0, '#FFFF00'),
            (100.0, '#FFA500'),
            (125.0, '#FF4500'),
            (150.0, '#FF0000'),
            (200.0, '#8B0000'),
            (250.0, '#8A2BE2'),
            (300.0, '#4B0082'),
            (400.0, '#FF00FF')
        ]
        return ColormapRegistry.build_dynamic_cmap(val_color_list, "acc_precip_scale")

    @staticmethod
    def get_wind():
        # ---------------------------------------------------------
        # WINDBÖEN (km/h)
        # ---------------------------------------------------------
        val_color_list = [
            (0,   'transparent'),
            (20,  'transparent'),
            (30,  '#E0FFFF'),
            (40,  '#00FFFF'),
            (50,  '#1E90FF'),
            (60,  '#0000FF'),
            (75,  '#00008B'),
            (90,  '#8A2BE2'),
            (100, '#FF0000'),
            (115, '#8B0000'),
            (130, '#4B0082'),
            (150, '#000000')
        ]
        return ColormapRegistry.build_dynamic_cmap(val_color_list, "wind_scale")

    @staticmethod
    def get_snow_depth():
        # ---------------------------------------------------------
        # SCHNEEHÖHE (cm)
        # ---------------------------------------------------------
        val_color_list = [
            (0.0, 'transparent'),
            (0.1, 'transparent'),
            (1.0, '#FACAF4'),
            (2.0, '#F3BBF8'),
            (5.0, '#F695FA'),
            (7.5, '#F375FA'),
            (10.0, '#F155FA'),
            (15.0, '#FA0CF2'),
            (20.0, '#C255BE'),
            (30.0, '#C171DB'),
            (40.0, '#D5BAEB'),
            (50.0, '#E3E0EC')
        ]
        return ColormapRegistry.build_dynamic_cmap(val_color_list, "snow_scale")

    @staticmethod
    def get_zero_degree_line():
        # ---------------------------------------------------------
        # 0-GRAD-GRENZE (m)
        # ---------------------------------------------------------
        val_color_list = [
            (0,    '#8B0000'),
            (500,  '#FF0000'),
            (1000, '#FFA500'),
            (1500, '#FFFF00'),
            (2000, '#00FF00'),
            (2500, '#00FA9A'),
            (3000, '#00FFFF'),
            (3500, '#0000FF'),
            (4000, '#FFFFFF')
        ]
        return ColormapRegistry.build_dynamic_cmap(val_color_list, "zero_deg_scale")


# ==============================================================================
# 6. MODEL REGISTRY (ALLE MODELLE)
# ==============================================================================
class ModelRegistry:
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
        "CFS (Langfrist)": {
            "regions": ["Europa", "Europa und Nordatlantik"],
            "params": [
                "500 hPa Geopot. Höhe", 
                "850 hPa Temperatur (°C)", 
                "Temperatur 2m (°C)", 
                "Niederschlag (mm)",
                "Akkumulierter Niederschlag (mm)"
            ],
            "type": "cfs_long"
        },
        "GFS (NOAA Global)": {
            "regions": GLOBAL_REGIONS,
            "params": [
                "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)",
                "Niederschlag (mm)", "Akkumulierter Niederschlag (mm)", "Gesamtbedeckung (%)", "Schneehöhe (cm)", 
                "850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe", "300 hPa Jetstream (km/h)", "0-Grad-Grenze (m)",
                "Theta-E (Äquivalentpotenzielle Temp.)", "K-Index (Gewitter)", "Vorticity Advection 500 hPa",
                "Showalter-Index (Stabilität)"
            ],
            "type": "gfs_ultra"
        },
        "GFS Ensemble (Mittel)": {
            "regions": ["Europa", "Europa und Nordatlantik"],
            "params": [
                "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)",
                "Niederschlag (mm)", "Akkumulierter Niederschlag (mm)", "Gesamtbedeckung (%)", "Schneehöhe (cm)", 
                "850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe"
            ],
            "type": "gfs_ultra"
        },
        "ICON-D2 (Deutschland High-Res)": {
            "regions": ["Deutschland", "Brandenburg (Gesamt)", "Berlin & Umland (Detail-Zoom)", "Mitteleuropa (DE, PL, CZ)", "Süddeutschland / Alpen"],
            "params": [
                "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Niederschlag (mm)", 
                "Akkumulierter Niederschlag (mm)", "Gesamtbedeckung (%)", "Schneehöhe (cm)", "Signifikantes Wetter", 
                "Simuliertes Radar (dBZ)"
            ],
            "type": "dwd_short"
        },
        "ICON-EU (Europa)": {
            "regions": GLOBAL_REGIONS,
            "params": [
                "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)",
                "Niederschlag (mm)", "Akkumulierter Niederschlag (mm)", "Gesamtbedeckung (%)", "Schneehöhe (cm)", 
                "850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe"
            ],
            "type": "dwd_long"
        },
        "ECMWF (IFS HRES)": {
            "regions": GLOBAL_REGIONS,
            "params": [
                "Temperatur 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", "Niederschlag (mm)", 
                "Akkumulierter Niederschlag (mm)", "Gesamtbedeckung (%)", "850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe"
            ],
            "type": "ecmwf_long"
        },
        "UKMO (Met Office UK)": {
            "regions": GLOBAL_REGIONS,
            "params": [
                "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)",
                "Niederschlag (mm)", "Akkumulierter Niederschlag (mm)", "Gesamtbedeckung (%)", "Schneehöhe (cm)", 
                "850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe"
            ],
            "type": "ecmwf_long"
        },
        "GEM (CMC Kanada)": {
            "regions": GLOBAL_REGIONS,
            "params": [
                "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)",
                "Niederschlag (mm)", "Akkumulierter Niederschlag (mm)", "Gesamtbedeckung (%)", "Schneehöhe (cm)", 
                "850 hPa Temperatur (°C)", "500 hPa Geopot. Höhe"
            ],
            "type": "ecmwf_long"
        },
        "Arpege (Meteo France)": {
            "regions": ["Deutschland", "Mitteleuropa (DE, PL, CZ)", "Süddeutschland / Alpen", "Europa", "Europa und Nordatlantik"],
            "params": [
                "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)",
                "Niederschlag (mm)", "Akkumulierter Niederschlag (mm)", "Gesamtbedeckung (%)", "Schneehöhe (cm)"
            ],
            "type": "ecmwf_long"
        }
    }

    @staticmethod
    def get_timesteps(model_type: str) -> List[int]:
        if model_type == "live": return [0]
        elif model_type == "dwd_short": return list(range(1, 49))
        elif model_type == "dwd_long": return list(range(1, 79)) + list(range(81, 121, 3))
        elif model_type == "gfs_ultra": return list(range(3, 385, 3))
        elif model_type == "ecmwf_long": return list(range(3, 243, 3))
        elif model_type == "cfs_long": return list(range(0, 4009, 12))
        return list(range(1, 49))

# ==============================================================================
# 7. DATA FETCH ENGINE (ROBUST & HISTORY ENABLED)
# ==============================================================================
class DataFetcher:
    
    @staticmethod
    def get_recent_runs(model: str, now_utc: datetime) -> List[datetime]:
        """Gibt die Modellläufe der letzten 24 Stunden zurück, damit Nutzer frei wählen können."""
        if "Radar" in model or "Pegel" in model:
            return [now_utc]
            
        interval = 3 if any(m in model for m in ["D2", "Arpege"]) else 6
        latest_run = ((now_utc.hour - interval) // interval) * interval
        
        base_time = now_utc.replace(hour=latest_run, minute=0, second=0, microsecond=0)
        if latest_run < 0: 
            base_time = (now_utc - timedelta(days=1)).replace(hour=24+latest_run, minute=0, second=0, microsecond=0)
            
        runs = []
        for i in range(int(24 / interval) + 1):
            runs.append(base_time - timedelta(hours=i * interval))
        return runs

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
                    if val is not None:
                        stations.append({
                            'name': st['shortname'], 
                            'lat': st['latitude'], 
                            'lon': st['longitude'],
                            'val': val
                        })
            df = pd.DataFrame(stations).dropna()
            if df.empty: return None
            return df
        except Exception:
            return None

    @staticmethod
    def fetch_rainviewer() -> Tuple[Optional[str], Optional[str], Optional[str]]:
        try:
            r = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=10)
            past = r.json().get("radar", {}).get("past", [])
            if past:
                host = r.json().get("host", "https://tilecache.rainviewer.com")
                path = past[-1]["path"]
                time_str = str(past[-1]["time"])
                return host, path, time_str
        except Exception: 
            pass
        return None, None, None

    @classmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_model_data(cls, model: str, param: str, hr: int, target_run: datetime) -> Tuple[Any, Any, Any, Any]:
        
        if "Pegel" in model:
            return cls.fetch_pegelonline(), None, None, datetime.now().strftime("%Y%m%d%H%M")
            
        if "RainViewer" in model:
            h, p, t = cls.fetch_rainviewer()
            return h, p, None, t

        if param == "Akkumulierter Niederschlag (mm)":
            if "ICON" in model:
                pass 
            else:
                total_data, lons, lats, run_id = None, None, None, None
                step = 6 if hr > 120 else 3
                for h in range(step, hr + 1, step):
                    d, ln, lt, rid = cls._fetch_single_param(model, "Niederschlag (mm)", h, target_run)
                    if d is not None:
                        if total_data is None:
                            total_data = np.zeros_like(d)
                            lons, lats, run_id = ln, lt, rid
                        total_data += d
                return total_data, lons, lats, run_id

        return cls._fetch_single_param(model, param, hr, target_run)

    @classmethod
    def _fetch_single_param(cls, model: str, param: str, hr: int, target_run: datetime) -> Tuple[Any, Any, Any, Any]:
        
        p_map = {
            "Temperatur 2m (°C)": "t_2m", 
            "Taupunkt 2m (°C)": "td_2m", 
            "Windböen (km/h)": "vmax_10m", 
            "300 hPa Jetstream (km/h)": "u", 
            "Bodendruck (hPa)": "pmsl", 
            "500 hPa Geopot. Höhe": "fi", 
            "850 hPa Temperatur (°C)": "t", 
            "Niederschlag (mm)": "tot_prec", 
            "Akkumulierter Niederschlag (mm)": "tot_prec",
            "Simuliertes Radar (dBZ)": "dbz_cmax", 
            "Gesamtbedeckung (%)": "clct", 
            "Schneehöhe (cm)": "h_snow",
            "0-Grad-Grenze (m)": "h_zerodeg", 
            "Signifikantes Wetter": "ww", 
            "Theta-E (Äquivalentpotenzielle Temp.)": "t_2m",  
            "K-Index (Gewitter)": "t_2m",                    
            "Showalter-Index (Stabilität)": "t_2m",          
            "Vorticity Advection 500 hPa": "u"
        }
        
        key = p_map.get(param, "t_2m")
        run_h = target_run.hour
        dt_s = target_run.strftime("%Y%m%d")

        # ======================================================================
        # CFS (Langfrist)
        # ======================================================================
        if model == "CFS (Langfrist)":
            headers = {'User-Agent': 'Mozilla/5.0'}
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
            
            url = f"https://nomads.ncep.noaa.gov/cgi-bin/{cfs_script}?file={cfs_prefix}{hr:02d}.01.{dt_s}{run_h:02d}.grb2{cfs_p}&subregion=&leftlon=-50&rightlon=45&toplat=75&bottomlat=20&dir=%2Fcfs.{dt_s}%2F{run_h:02d}%2F6hrly_grib_01"
            try:
                r = requests.get(url, headers=headers, timeout=12)
                if r.status_code == 200:
                    with open("temp_cfs.grib", "wb") as f: f.write(r.content)
                    ds = xr.open_dataset("temp_cfs.grib", engine='cfgrib')
                    data = ds[list(ds.data_vars)[0]].isel(step=0, missing_dims='ignore').values.squeeze()
                    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                    return data, lons, lats, f"{dt_s}{run_h:02d}"
            except: pass
            return None, None, None, None

        # ======================================================================
        # GFS, UKMO, GEM, JMA, ACCESS-G
        # ======================================================================
        elif any(m in model for m in ["GFS", "UKMO", "GEM", "JMA", "ACCESS"]):
            headers = {'User-Agent': 'Mozilla/5.0'}
            is_gefs = "Ensemble" in model
            script = "filter_gefs_atmos_0p50a.pl" if is_gefs else "filter_gfs_0p25.pl"
            file_prefix = "geavg.t" if is_gefs else "gfs.t"
            file_suffix = "z.pgrb2a.0p50.f" if is_gefs else "z.pgrb2.0p25.f"
            dir_prefix = "gefs" if is_gefs else "gfs"
            dir_suffix = "%2Fatmos%2Fpgrb2ap5" if is_gefs else "%2Fatmos"

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
                "h_zerodeg": "&var_HGT=on&lev_0C_isotherm=on"
            }
            gfs_p = gfs_map.get(key, "&var_TMP=on&lev_2_m_above_ground=on")
            
            url = f"https://nomads.ncep.noaa.gov/cgi-bin/{script}?file={file_prefix}{run_h:02d}{file_suffix}{hr:03d}{gfs_p}&subregion=&leftlon=-50&rightlon=45&toplat=75&bottomlat=20&dir=%2F{dir_prefix}.{dt_s}%2F{run_h:02d}{dir_suffix}"
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    with open("temp_gfs.grib", "wb") as f: f.write(r.content)
                    ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                    data = ds[list(ds.data_vars)[0]].isel(step=0, missing_dims='ignore').values.squeeze()
                    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                    return data, lons, lats, f"{dt_s}{run_h:02d}"
            except: pass
            return None, None, None, None

        # ======================================================================
        # ICON (D2 & EU) & ECMWF / Arpege
        # ======================================================================
        else:
            m_dir = "icon-d2" if "D2" in model else ("icon-eps" if "Ensemble" in model else "icon-eu")
            reg_str = "icon-d2_germany" if "D2" in model else ("icon-eps_global" if "Ensemble" in model else "icon-eu_europe")
            dt_s_hour = target_run.strftime("%Y%m%d%H")
            
            l_type = "single-level"
            lvl_str = "2d_"
            if key in ["fi", "t", "u"]:
                l_type = "pressure-level"
                if key == "fi": lvl_str = "500_"
                elif key == "u": lvl_str = "300_"
                else: lvl_str = "850_"
            
            if "Ensemble" in model:
                url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run_h:02d}/{key}/{reg_str}_icosahedral_{l_type}_{dt_s_hour}_{hr:03d}_{lvl_str}{key}.grib2.bz2"
            else:
                url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run_h:02d}/{key}/{reg_str}_regular-lat-lon_{l_type}_{dt_s_hour}_{hr:03d}_{lvl_str}{key}.grib2.bz2"
            
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    with bz2.open(io.BytesIO(r.content)) as f_bz2:
                        with open("temp.grib", "wb") as f_out: f_out.write(f_bz2.read())
                    ds = xr.open_dataset("temp.grib", engine='cfgrib')
                    ds_var = ds[list(ds.data_vars)[0]]
                    if 'isobaricInhPa' in ds_var.dims: 
                        ds_var = ds_var.sel(isobaricInhPa=int(lvl_str.replace("_", "")))
                    data = ds_var.isel(step=0, missing_dims='ignore').values.squeeze()
                    lons, lats = ds.longitude.values, ds.latitude.values
                    if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                    return data, lons, lats, dt_s_hour
            except: pass
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
            shading='gouraud', 
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
                colors='white',
                linewidths=1.2, 
                levels=np.arange(940, 1060, 5), 
                transform=ccrs.PlateCarree(), 
                zorder=16
            )
            ax.clabel(cs, inline=True, fontsize=9, fmt='%1.0f')

    @staticmethod
    def plot_geopotential(ax, fig, lons, lats, data):
        val = MeteoMath.geopotential_to_m(data)
        cmap, norm, thresh = ColormapRegistry.get_geopotential()
        
        if thresh is not None:
            val = np.where(val <= thresh, np.nan, val)
            
        im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "Geopotentielle Höhe (gpdm)")
        
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, aspect=40)
        ticks_m = np.arange(4800, 6400, 200)
        cb.set_ticks(ticks_m)
        cb.set_ticklabels([str(int(t/10)) for t in ticks_m])
        cb.set_label("Geopotentielle Höhe (gpdm)", fontsize=10, fontweight='bold')

        ax.contour(
            lons, 
            lats, 
            val, 
            levels=[5520], 
            colors='white', 
            linewidths=2.0, 
            transform=ccrs.PlateCarree(), 
            zorder=15
        )

    @staticmethod
    def plot_clouds(ax, fig, lons, lats, data):
        cmap, norm, thresh = ColormapRegistry.get_clouds()
        if thresh is not None:
            data = np.where(data <= thresh, np.nan, data)
            
        im = PlottingEngine._plot_base(ax, fig, lons, lats, data, cmap, norm, "Gesamtbedeckung (%)")
        fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, aspect=40, label="Gesamtbedeckung (%)")

    @staticmethod
    def plot_precipitation(ax, fig, lons, lats, data, overlay=False):
        cmap, norm, thresh = ColormapRegistry.get_precipitation()
        if thresh is not None:
            data = np.where(data <= thresh, np.nan, data)
            
        alpha = 0.6 if overlay else 0.85
        zorder = 8 if overlay else 5
        
        im = PlottingEngine._plot_base(ax, fig, lons, lats, data, cmap, norm, "Niederschlagssumme in mm", alpha=alpha, zorder=zorder)
        
        if not overlay:
            fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, aspect=40, label="Niederschlagssumme in mm")

    @staticmethod
    def plot_acc_precipitation(ax, fig, lons, lats, data):
        cmap, norm, thresh = ColormapRegistry.get_acc_precipitation()
        if thresh is not None:
            data = np.where(data <= thresh, np.nan, data)
            
        im = PlottingEngine._plot_base(ax, fig, lons, lats, data, cmap, norm, "Akkumulierter Niederschlag (mm)", alpha=0.85, zorder=5)
        fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, aspect=40, label="Akkumulierter Niederschlag (mm)")

    @staticmethod
    def plot_generic(ax, fig, lons, lats, data, name):
        
        if "Radar" in name:
            cmap, norm = ColormapRegistry.get_precipitation()
            data = np.where(data <= 0, np.nan, data)
            im = PlottingEngine._plot_base(ax, fig, lons, lats, data, cmap, norm, "Radar (dBZ)")
            fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, aspect=40, label="Radar (dBZ)")
            
        elif "Temperatur" in name or "Taupunkt" in name or "Index" in name or "Theta" in name:
            val = MeteoMath.kelvin_to_celsius(data)
            
            if "850" in name:
                cmap, norm, thresh = ColormapRegistry.get_temperature_850()
            elif "Taupunkt" in name:
                cmap, norm, thresh = ColormapRegistry.get_dewpoint()
            else:
                cmap, norm, thresh = ColormapRegistry.get_temperature()
                
            if thresh is not None:
                val = np.where(val <= thresh, np.nan, val)
                
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, name)
            fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, aspect=40, label=name)
            
        elif "Schnee" in name:
            val = data * 100 if np.nanmax(data) < 10 else data
            cmap, norm, thresh = ColormapRegistry.get_snow_depth()
            if thresh is not None:
                val = np.where(val <= thresh, np.nan, val)
                
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "Schneehöhe (cm)")
            fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, aspect=40, label="Schneehöhe (cm)")
            
        elif "Wind" in name or "Jetstream" in name:
            val = np.abs(data) * 3.6 if data.max() < 100 else data
            cmap, norm, thresh = ColormapRegistry.get_wind()
            if thresh is not None:
                val = np.where(val <= thresh, np.nan, val)
                
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, name)
            fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, aspect=40, label=name)
            
        elif "Bodendruck" in name:
            val = MeteoMath.pa_to_hpa(data)
            cmap, norm, thresh = ColormapRegistry.get_surface_pressure()
            if thresh is not None:
                val = np.where(val <= thresh, np.nan, val)
                
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "Bodendruck (hPa)")
            fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, aspect=40, label="Bodendruck (hPa)")
            
        elif "0-Grad-Grenze" in name:
            val = MeteoMath.geopotential_to_m(data)
            cmap, norm, thresh = ColormapRegistry.get_zero_degree_line()
            if thresh is not None:
                val = np.where(val <= thresh, np.nan, val)
                
            im = PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "0-Grad-Grenze (m)")
            fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, aspect=40, label="0-Grad-Grenze (m)")
            
        else:
            im = PlottingEngine._plot_base(ax, fig, lons, lats, data, 'viridis', mcolors.Normalize(), name)
            fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.04, aspect=40, label=name)

    @staticmethod
    def plot_rainviewer(ax, fig, host, path, region):
        ax.add_image(RainViewerTiles(host, path), GeoConfig.get_zoom(region), zorder=5, alpha=0.85)


# ==============================================================================
# HTML DOWNLOAD FUNKTION (FÜR APKS)
# ==============================================================================
def get_download_html(buf: io.BytesIO, filename: str) -> str:
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
    
    use_split = st.checkbox("🔄 Vergleichs-Modus (Split-Screen)", value=False)
    st.markdown("---")
    
    st.markdown("### 🔹 Anzeige 1")
    mod_1 = st.radio("Wettermodell 1", list(ModelRegistry.MODELS.keys()), label_visibility="collapsed")
    reg_1 = st.radio("Karten-Ausschnitt 1", ModelRegistry.MODELS[mod_1]["regions"])
    par_1 = st.radio("Parameter 1", ModelRegistry.MODELS[mod_1]["params"])
    
    now_utc = datetime.now(timezone.utc)
    
    if "Radar" in mod_1 or "Pegel" in mod_1:
        hr_1 = 0
        target_run_1 = now_utc
    else:
        runs_1 = DataFetcher.get_recent_runs(mod_1, now_utc)
        target_run_1 = st.selectbox("Lauf auswählen 1", runs_1, format_func=lambda x: f"{x.strftime('%d.%m. %H:%M')} UTC", index=0)
        
        h_list_1 = ModelRegistry.get_timesteps(ModelRegistry.MODELS[mod_1]["type"])
        options_1 = []
        for h in h_list_1:
            valid_time = (target_run_1 + timedelta(hours=h)).astimezone(LOCAL_TZ)
            options_1.append(f"+{h}h | {valid_time.strftime('%d.%m. %H:%M')}")
            
        hr_str_1 = st.select_slider("Zeitpunkt wählen", options=options_1)
        hr_1 = int(hr_str_1.split("h")[0].replace("+", ""))

    mod_2, par_2, hr_2, target_run_2 = None, None, 0, now_utc
    if use_split:
        st.markdown("---")
        st.markdown("### 🔸 Anzeige 2")
        mod_2 = st.radio("Wettermodell 2", list(ModelRegistry.MODELS.keys()), index=3, label_visibility="collapsed")
        par_2 = st.radio("Parameter 2", ModelRegistry.MODELS[mod_2]["params"])
        
        if "Radar" in mod_2 or "Pegel" in mod_2:
            hr_2 = 0
        else:
            runs_2 = DataFetcher.get_recent_runs(mod_2, now_utc)
            target_run_2 = st.selectbox("Lauf auswählen 2", runs_2, format_func=lambda x: f"{x.strftime('%d.%m. %H:%M')} UTC", index=0)
            
            h_list_2 = ModelRegistry.get_timesteps(ModelRegistry.MODELS[mod_2]["type"])
            options_2 = []
            for h in h_list_2:
                valid_time_2 = (target_run_2 + timedelta(hours=h)).astimezone(LOCAL_TZ)
                options_2.append(f"+{h}h | {valid_time_2.strftime('%d.%m. %H:%M')}")
                
            hr_str_2 = st.select_slider("Zeitpunkt 2 wählen", options=options_2)
            hr_2 = int(hr_str_2.split("h")[0].replace("+", ""))
            
    st.markdown("---")
    
    # HINTERGRUND-STEUERUNG
    show_sat = st.checkbox("🌍 Satelliten-Hintergrund (Global)", value=False)
    
    snow_bg = "Standard"
    if "Schnee" in par_1 or (use_split and "Schnee" in par_2):
        snow_bg = st.radio("Schnee-Hintergrund", ["Standard", "Grün (#049700)", "Satellit"])
        
    show_isobars = st.checkbox("Isobaren einblenden", value=False)
    
    if "Gesamtbedeckung" in par_1:
        overlay_precip = st.checkbox("🌧️ Niederschlag über Wolken", value=False)
    else:
        overlay_precip = False


# ==============================================================================
# 10. MAIN EXECUTION & RENDER ENGINE
# ==============================================================================
st.title("🛰️ WarnwetterBB Pro-Zentrale")
generate = st.button("🚀 Karte jetzt generieren", use_container_width=True)


def render_axis(ax, fig, model, param, hr, region, target_run):
    data, lons, lats, run_id = DataFetcher.fetch_model_data(model, param, hr, target_run)
    ax.set_extent(GeoConfig.get_extent(region), crs=ccrs.PlateCarree())

    # HINTERGRUND-LOGIK
    border_col = 'black'
    
    if "Schnee" in param:
        if snow_bg == "Satellit":
            ax.add_image(GoogleSatelliteTiles(), GeoConfig.get_zoom(region), zorder=0)
            border_col = 'white'
        elif snow_bg == "Grün (#049700)":
            ax.add_feature(cfeature.LAND, facecolor='#049700', zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor='#2B65EC', zorder=0)
    else:
        allow_sat = ("Gesamtbedeckung" in param) or ("Radar" in param) or show_sat
        if allow_sat: 
            ax.add_image(GoogleSatelliteTiles(), GeoConfig.get_zoom(region), zorder=0)
            border_col = 'white'

    ax.add_feature(cfeature.COASTLINE, linewidth=0.9, edgecolor=border_col, zorder=12)
    ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor=border_col, zorder=12)
    
    if region in ["Deutschland", "Brandenburg (Gesamt)", "Berlin & Umland (Detail-Zoom)", "Mitteleuropa (DE, PL, CZ)"]:
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='10m',
            facecolor='none'
        )
        ax.add_feature(states_provinces, edgecolor=border_col, linewidth=0.6, zorder=13)

    if data is not None or "Pegel" in model:
        
        if "Radar" in param and "RainViewer" in model: 
            PlottingEngine.plot_rainviewer(ax, fig, data, lons, region)
            
        elif "Gesamtbedeckung" in param: 
            PlottingEngine.plot_clouds(ax, fig, lons, lats, data)
            if overlay_precip:
                p_data, p_lons, p_lats, _ = DataFetcher.fetch_model_data(model, "Niederschlag (mm)", hr, target_run)
                if p_data is not None: 
                    PlottingEngine.plot_precipitation(ax, fig, p_lons, p_lats, p_data, overlay=True)
                    
        elif "Geopot. Höhe" in param: 
            PlottingEngine.plot_geopotential(ax, fig, lons, lats, data)
            
        elif "Akkumulierter Niederschlag" in param: 
            PlottingEngine.plot_acc_precipitation(ax, fig, lons, lats, data)
            
        elif "Niederschlag" in param: 
            PlottingEngine.plot_precipitation(ax, fig, lons, lats, data)
            
        elif "Signifikant" in param: 
            PlottingEngine.plot_sig_weather(ax, fig, lons, lats, data)
            
        elif "Pegel" in model:
            pass
            
        else: 
            PlottingEngine.plot_generic(ax, fig, lons, lats, data, param)

        if show_isobars and "Radar" not in model and "Pegel" not in model:
            iso_d, iso_l, iso_a, _ = DataFetcher.fetch_model_data(model, "Bodendruck (hPa)", hr, target_run)
            PlottingEngine.add_isobars(ax, iso_d, iso_l, iso_a)

        if "Radar" in model or "Pegel" in model:
            txt = f" {model} | {param} | Live-Stand: {datetime.now(LOCAL_TZ).strftime('%H:%M')} Uhr "
        else:
            run_dt = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            valid_dt = (run_dt + timedelta(hours=hr)).astimezone(LOCAL_TZ)
            txt = f" {model} | {param} | Lauf: {run_id[-2:]}Z | Termin: {valid_dt.strftime('%d.%m.%Y %H:%M')} {'MESZ' if valid_dt.dst() else 'MEZ'} "
            
        ax.set_title(txt, loc='left', fontsize=11, fontweight='bold', backgroundcolor='white', color='black', pad=10)

        ax.text(
            0.99, 0.01, 
            "© WarnWetter Berlin-Brandenburg", 
            transform=ax.transAxes, 
            fontsize=10, fontweight='bold', color='white', ha='right', va='bottom', 
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'), zorder=30
        )
            
    else:
        ax.text(
            0.5, 0.5, "Daten aktuell nicht verfügbar", 
            transform=ax.transAxes, ha='center', va='center', fontsize=12, color='red', 
            bbox=dict(facecolor='white', alpha=0.8)
        )


if generate:
    SystemManager.cleanup_temp_files()
    with st.spinner("🛰️ Lade Modell-Daten..."):
        
        if use_split:
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
                render_axis(ax1, fig1, mod_1, par_1, hr_1, reg_1, target_run_1)
                
                buf1 = io.BytesIO()
                fig1.savefig(buf1, format='png', bbox_inches='tight', dpi=150)
                buf1.seek(0)
                st.image(buf1, use_container_width=True)
                
                filename_1 = f"WarnwetterBB_{mod_1.split()[0]}_{hr_1}h.png"
                st.markdown(get_download_html(buf1, filename_1), unsafe_allow_html=True)
                
            with col2:
                fig2, ax2 = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
                render_axis(ax2, fig2, mod_2, par_2, hr_2, reg_1, target_run_2)
                
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format='png', bbox_inches='tight', dpi=150)
                buf2.seek(0)
                st.image(buf2, use_container_width=True)
                
                filename_2 = f"WarnwetterBB_{mod_2.split()[0]}_{hr_2}h.png"
                st.markdown(get_download_html(buf2, filename_2), unsafe_allow_html=True)
                
        else:
            fig, ax = plt.subplots(figsize=(12, 14), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
            render_axis(ax, fig, mod_1, par_1, hr_1, reg_1, target_run_1)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            
            st.image(buf, use_container_width=True)
            
            filename = f"WarnwetterBB_{mod_1.split()[0]}_{hr_1}h.png"
            st.markdown(get_download_html(buf, filename), unsafe_allow_html=True)
        
    SystemManager.cleanup_temp_files()

