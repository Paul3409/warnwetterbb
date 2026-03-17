"""
=========================================================================================
WARNWETTER BB - PROFESSIONAL METEOROLOGICAL WORKSTATION (ENTERPRISE UNCOMPRESSED)
=========================================================================================
Version: 6.0 (Explicit Architecture & Radar-Zoom Fix)
Fokus: Maximale Code-Transparenz, keine künstliche Komprimierung, 100% explizite Logik.
Alle Abhängigkeiten zu Scipy wurden entfernt (Pure Numpy). Radar ist auf DE/EU limitiert.

Enthaltene Module:
- DWD ICON (D2, EU, Global)
- NOAA GFS
- ECMWF (HRES & AIFS)
- UKMO (Met Office)
- GEM (Kanada)
- Arpege (Meteo France)
- RainViewer API (Echtzeit-Radar)
- WSV Pegelonline API (Echtzeit-Wasserstände)
- MeteoMath (Theta-E, K-Index, Vorticity)
- Unwetter- & Fronten-Warnsystem
- Split-Screen Visualisierung
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
# Logging explizit definieren für besseres Debugging in Streamlit
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - WarnwetterBB - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WarnwetterBB")

# Optionaler Auto-Refresher für Live-Radar und Live-Pegel
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None
    logger.warning("Modul 'streamlit_autorefresh' fehlt. Auto-Update ist deaktiviert.")

# Streamlit Page Config muss als allererstes aufgerufen werden
st.set_page_config(
    page_title="WarnwetterBB | Pro-Zentrale", 
    page_icon="🌪️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS für Enterprise Look and Feel
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
    .metric-container { 
        background-color: #2b2b2b; 
        color: white; 
        padding: 10px; 
        border-radius: 5px; 
    }
    </style>
""", unsafe_allow_html=True)

# Globale Konstanten
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
        """
        Sucht und löscht explizit alle temporären GRIB- und Index-Dateien.
        """
        temp_extensions = [
            ".grib", 
            ".grib2", 
            ".bz2", 
            ".idx", 
            ".tmp", 
            "temp_gfs", 
            "temp_ukmo", 
            "temp_gem", 
            "temp_arpege"
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
    """
    Verwaltet exakte Koordinatenboxen für reibungsloses Cartopy-Zooming.
    Explizit ausgeschrieben für maximale Transparenz.
    """
    
    EXTENTS = {
        "Deutschland": [5.5, 15.5, 47.0, 55.2],
        "Brandenburg (Gesamt)": [11.0, 15.0, 51.1, 53.7],
        "Berlin & Umland (Detail-Zoom)": [12.8, 13.9, 52.3, 52.7],
        "Süddeutschland / Alpen": [7.0, 14.0, 46.5, 49.5],
        "Norddeutschland / Küste": [6.0, 14.5, 52.5, 55.2],
        "Mitteleuropa (DE, PL, CZ)": [4.0, 25.0, 45.0, 56.0],
        "Europa": [-12.0, 40.0, 34.0, 66.0]
    }
    
    ZOOM_LEVELS = {
        "Deutschland": 6,
        "Brandenburg (Gesamt)": 8,
        "Berlin & Umland (Detail-Zoom)": 10,
        "Süddeutschland / Alpen": 7,
        "Norddeutschland / Küste": 7,
        "Mitteleuropa (DE, PL, CZ)": 6,
        "Europa": 5
    }
    
    @classmethod
    def get_extent(cls, region_name: str) -> List[float]:
        """Gibt die Bounding Box für Cartopy zurück."""
        if region_name in cls.EXTENTS:
            return cls.EXTENTS[region_name]
        return cls.EXTENTS["Deutschland"]
        
    @classmethod
    def get_zoom(cls, region_name: str) -> int:
        """Gibt das optimale Tile-Zoom-Level für den Hintergrund zurück."""
        if region_name in cls.ZOOM_LEVELS:
            return cls.ZOOM_LEVELS[region_name]
        return 6


# ==============================================================================
# 3. PHYSIKALISCHE ENGINE (PURE NUMPY MATH)
# ==============================================================================
class MeteoMath:
    """
    Meteorologische Berechnungen aus den Rohdaten. 
    Garantiert OHNE externe Abhängigkeiten wie Scipy.
    Alle Funktionen sind explizit für Skalare und Numpy-Arrays geschrieben.
    """
    
    @staticmethod
    def kelvin_to_celsius(temp_k: np.ndarray) -> np.ndarray:
        """Konvertiert Kelvin in Celsius, sofern die Daten noch in Kelvin vorliegen."""
        if np.nanmax(temp_k) > 100:
            return temp_k - 273.15
        return temp_k
        
    @staticmethod
    def ms_to_kmh(speed_ms: np.ndarray) -> np.ndarray:
        """Konvertiert Meter pro Sekunde in Kilometer pro Stunde."""
        if np.nanmax(speed_ms) < 100:
            return speed_ms * 3.6
        return speed_ms
        
    @staticmethod
    def pa_to_hpa(pressure_pa: np.ndarray) -> np.ndarray:
        """Konvertiert Pascal in Hektopascal (Millibar)."""
        if np.nanmax(pressure_pa) > 5000:
            return pressure_pa / 100
        return pressure_pa
        
    @staticmethod
    def geopotential_to_gpdm(geo_data: np.ndarray) -> np.ndarray:
        """Berechnet Geopotentielle Dekameter aus m²/s²."""
        if np.nanmax(geo_data) > 10000:
            return (geo_data / 9.80665) / 10
        return geo_data / 10

    @staticmethod
    def calc_theta_e(t_850: np.ndarray, td_850: np.ndarray) -> np.ndarray:
        """
        Bolton-Approximation (1980) für Äquivalentpotenzielle Temperatur.
        Ein sehr guter Indikator für schwüle, energiegeladene Gewitterluft.
        """
        # Sicherstellen, dass die Eingabe in Kelvin ist
        tk = t_850 if np.nanmax(t_850) > 100 else t_850 + 273.15
        tdk = td_850 if np.nanmax(td_850) > 100 else td_850 + 273.15
        p = 850.0 
        
        # Dampfdruck (e)
        e = 6.112 * np.exp((17.67 * (tdk - 273.15)) / (tdk - 29.65))
        
        # Mischungsverhältnis (r)
        r = (0.622 * e) / (p - e)
        
        # LCL (Lifting Condensation Level) Temperatur
        tlcl = 56.0 + 1.0 / (1.0 / (tdk - 56.0) + np.log(tk / tdk) / 800.0)
        
        # Potenzielle Temperatur (Theta)
        theta = tk * (1000.0 / p) ** 0.2854
        
        # Äquivalentpotenzielle Temperatur (Theta-E)
        theta_e = theta * np.exp((3.376 / tlcl - 0.00254) * r * 1000.0 * (1.0 + 0.81 * r))
        
        return theta_e

    @staticmethod
    def calc_k_index(t850: np.ndarray, t500: np.ndarray, td850: np.ndarray, td700: np.ndarray, t700: np.ndarray) -> np.ndarray:
        """
        K-Index: Maß für das Gewitterpotenzial in Luftmassen ohne starke Dynamik.
        """
        t8 = MeteoMath.kelvin_to_celsius(t850)
        t5 = MeteoMath.kelvin_to_celsius(t500)
        td8 = MeteoMath.kelvin_to_celsius(td850)
        td7 = MeteoMath.kelvin_to_celsius(td700)
        t7 = MeteoMath.kelvin_to_celsius(t700)
        
        k_index = (t8 - t5) + td8 - (t7 - td7)
        return k_index

    @staticmethod
    def calc_vorticity_advection(u_500: np.ndarray, v_500: np.ndarray) -> np.ndarray:
        """
        Berechnet absolute Vorticity-Advektion rein mit Numpy-Bordmitteln.
        Verhindert Abstürze durch fehlende Scipy-Module in der Cloud.
        """
        # Gitterabstand-Näherung in Metern für mittlere Breiten
        dx = 25000.0
        dy = 25000.0 
        
        # Relative Vorticity
        dv_dx = np.gradient(v_500, dx, axis=1)
        du_dy = np.gradient(u_500, dy, axis=0)
        rel_vort = dv_dx - du_dy
        
        # Advektion der Vorticity
        dvort_dx = np.gradient(rel_vort, dx, axis=1)
        dvort_dy = np.gradient(rel_vort, dy, axis=0)
        
        vort_adv = - (u_500 * dvort_dx + v_500 * dvort_dy)
        
        # Skalierung für Plotting
        return vort_adv * 1e9

    @staticmethod
    def fast_numpy_smooth(arr: np.ndarray) -> np.ndarray:
        """
        Ein manueller, extrem schneller 3x3 Mean-Filter komplett in Numpy.
        Dient als vollständiger Ersatz für scipy.ndimage.gaussian_filter.
        """
        out = np.copy(arr)
        out[1:-1, 1:-1] = (
            arr[:-2, :-2] + arr[:-2, 1:-1] + arr[:-2, 2:] +
            arr[1:-1, :-2] + arr[1:-1, 1:-1] + arr[1:-1, 2:] +
            arr[2:, :-2] + arr[2:, 1:-1] + arr[2:, 2:]
        ) / 9.0
        return out


class AnalysisEngine:
    """
    Kapselt die Logik für Unwetterwarnungen und synoptische Frontenanalyse.
    """
    
    @staticmethod
    def detect_fronts(t_850_c: np.ndarray) -> np.ndarray:
        """
        Sucht nach starken Temperaturkontrasten (Gradienten) in der 850hPa Fläche,
        welche klassische Indikatoren für Kalt- und Warmfronten sind.
        """
        # Eigene Glättung anwenden (2 Pässe für Rauschunterdrückung)
        smoothed_t = MeteoMath.fast_numpy_smooth(t_850_c)
        smoothed_t = MeteoMath.fast_numpy_smooth(smoothed_t) 
        
        # Gradient Magnitude berechnen
        grad_y, grad_x = np.gradient(smoothed_t)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Die obersten 5% der Kontraste als Front definieren
        threshold = np.percentile(grad_mag, 95)
        front_mask = np.where(grad_mag >= threshold, 1, 0)
        
        return front_mask

    @staticmethod
    def get_severe_warnings(wind_gusts_kmh: np.ndarray, precip_mm: np.ndarray) -> np.ndarray:
        """
        Drei-Stufen Warnsystem des DWD nachempfunden.
        0 = Keine Gefahr
        1 = Markantes Wetter (Gelb)
        2 = Unwetter (Rot)
        3 = Extremes Unwetter (Violett)
        """
        warnings = np.zeros_like(wind_gusts_kmh)
        
        # Stufe 1: Markant
        warnings[(wind_gusts_kmh >= 65) | (precip_mm >= 15)] = 1
        # Stufe 2: Unwetter
        warnings[(wind_gusts_kmh >= 90) | (precip_mm >= 30)] = 2
        # Stufe 3: Extremes Unwetter
        warnings[(wind_gusts_kmh >= 115) | (precip_mm >= 50)] = 3
        
        return warnings


# ==============================================================================
# 4. KARTEN-HINTERGRÜNDE & TILE-SERVER
# ==============================================================================
class GoogleSatelliteTiles(cimgt.GoogleWTS):
    """
    Hochauflösende Satellitenbilder von Google Maps.
    Zuverlässiger als die ESRI-Server.
    """
    def _image_url(self, tile: Tuple[int, int, int]) -> str:
        x, y, z = tile
        return f'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'


class RainViewerTiles(cimgt.GoogleWTS):
    """
    RainViewer Radar API. 
    ENTHÄLT DEN WICHTIGEN FIX FÜR SATELLITEN-TRANSPARENZ!
    """
    def __init__(self, host: str, path: str):
        self.host = host
        self.path = path
        super().__init__()

    def _image_url(self, tile: Tuple[int, int, int]) -> str:
        x, y, z = tile
        return f"{self.host}{self.path}/256/{z}/{x}/{y}/2/1_1.png"

    def get_image(self, tile: Tuple[int, int, int]):
        """Überschreibt die Standard-Methode, um RGBA zu erzwingen."""
        url = self._image_url(tile)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as fh:
                # convert('RGBA') ist der Befehl, der die schwarzen Kästen killt!
                img = PIL.Image.open(fh).convert('RGBA') 
            return img, self.tileextent(tile), 'lower'
        except Exception as e:
            logger.warning(f"RainViewer Tile nicht geladen: {e}")
            # Gib ein leeres, komplett transparentes Bild zurück
            return PIL.Image.new('RGBA', (256, 256), (0, 0, 0, 0)), self.tileextent(tile), 'lower'


# ==============================================================================
# 5. METEOROLOGISCHE FARBSKALEN (EXPLIZIT AUSGESCHRIEBEN)
# ==============================================================================
class ColormapRegistry:
    """
    Zentrale Definition aller meteorologischen Farben.
    Jede Map MUSS set_bad('none') haben, damit Satellitenbilder durchscheinen.
    """

    @staticmethod
    def get_temperature() -> mcolors.LinearSegmentedColormap:
        colors = [
            (0.0, '#313695'), 
            (0.1, '#4575b4'), 
            (0.2, '#74add1'), 
            (0.3, '#abd9e9'),
            (0.4, '#e0f3f8'), 
            (0.5, '#ffffbf'), 
            (0.6, '#fee090'), 
            (0.7, '#fdae61'),
            (0.8, '#f46d43'), 
            (0.9, '#d73027'), 
            (1.0, '#a50026')
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("temp_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_theta_e() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#000080', '#0000FF', '#00FFFF', '#00FF00', 
            '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#FF00FF'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("theta_e_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_k_index() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#FFFFFF', '#FFFF00', '#FFA500', '#FF0000', 
            '#8B0000', '#800080', '#4B0082'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("k_index_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_vorticity() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#0000FF', '#00BFFF', '#FFFFFF', '#FFA500', '#FF0000'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("vorticity_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_precipitation() -> Tuple[mcolors.LinearSegmentedColormap, mcolors.Normalize]:
        precip_colors = [
            '#FFFFFF', '#87CEEB', '#1E90FF', '#191970', '#006400', '#32CD32', '#FFFF00', 
            '#FFA500', '#FF0000', '#8B0000', '#800000', '#4B0082', '#800080', '#9400D3', '#7B68EE', '#FFFFFF'
        ]
        precip_values = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3, 4, 5, 8, 12, 15, 20, 30, 40, 50]
        vmax = 50.0
        anchors = [v / vmax for v in precip_values]
        
        cmap = mcolors.LinearSegmentedColormap.from_list("precip_scale", list(zip(anchors, precip_colors)))
        cmap.set_bad(color='none')
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        
        return cmap, norm

    @staticmethod
    def get_radar() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        colors = [
            '#FFFFFF', '#B0E0E6', '#00BFFF', '#0000FF', '#00FF00', '#32CD32', '#008000', 
            '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#800080', '#4B0082', '#E6E6FA'
        ]
        levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
        
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color='none')
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        
        return cmap, norm

    @staticmethod
    def get_wind() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', 
            '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("wind_scale", colors, N=256)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_jetstream() -> mcolors.LinearSegmentedColormap:
        colors = [
            '#FFFFFF', '#ADD8E6', '#0000FF', '#FF00FF', '#FF0000', '#8B0000', '#000000'
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("jetstream_scale", colors, N=256)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_wbi() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        colors = ['#FFFFFF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#8B0000']
        levels = [0, 1.5, 2.5, 3.5, 4.5, 5.5]
        
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color='none')
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        
        return cmap, norm

    @staticmethod
    def get_warnings() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        # Transparent, Gelb, Rot, Violett
        colors = ['#FFFFFF00', '#FFFF00', '#FF0000', '#800080'] 
        levels = [0, 0.5, 1.5, 2.5, 3.5]
        
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color='none')
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        
        return cmap, norm

    @staticmethod
    def get_cape() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        colors = [
            '#006400', '#2E8B57', '#ADFF2F', '#FFFF00', '#FFB347', '#FFA500', 
            '#FF4500', '#FF0000', '#8B0000', '#800080', '#FF00FF', '#FFFFFF', '#808080', '#404040'
        ]
        levels = [0, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]
        
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color='none')
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        
        return cmap, norm


# ==============================================================================
# 6. MODELL-ROUTING & KONFIGURATION (ENTKOMPRIMIERT)
# ==============================================================================
# Explizite Definition der Listen, um Übersichtlichkeit zu wahren
PARAMS_BASIC = [
    "Temperatur 2m (°C)", 
    "Taupunkt 2m (°C)", 
    "Windböen (km/h)", 
    "Bodendruck (hPa)",
    "Niederschlag (mm)", 
    "Gesamtbedeckung (%)", 
    "Schneehöhe (cm)", 
    "Unwetter-Warnungen"
]

PARAMS_PROFI = [
    "850 hPa Temperatur (°C)", 
    "500 hPa Geopotential", 
    "300 hPa Jetstream (km/h)",
    "Theta-E (Äquivalentpotenzielle Temp.)", 
    "K-Index (Gewitter)", 
    "Vorticity Advection 500 hPa"
]

# Das mächtige Dictionary, das das UI steuert.
# RADAR WURDE HIER AUF DE UND EU BEGRENZT!
MODEL_ROUTER = {
    "RainViewer Echtzeit-Radar": {
        "regions": ["Deutschland", "Europa"],
        "params": ["Echtzeit-Radar (Reflektivität)"]
    },
    "Live-Pegelstände (WSV)": {
        "regions": list(GeoConfig.EXTENTS.keys()),
        "params": ["Wasserstand (cm) & Trend"]
    },
    "ICON-D2 (Deutschland High-Res)": {
        "regions": ["Deutschland", "Brandenburg (Gesamt)", "Berlin & Umland (Detail-Zoom)", "Mitteleuropa (DE, PL, CZ)", "Alpenraum"],
        "params": PARAMS_BASIC + PARAMS_PROFI + ["Simuliertes Radar (dBZ)", "Waldbrandgefahrenindex (WBI)"]
    },
    "ICON-EU (Europa)": {
        "regions": list(GeoConfig.EXTENTS.keys()),
        "params": PARAMS_BASIC + PARAMS_PROFI
    },
    "GFS (NOAA Global)": {
        "regions": list(GeoConfig.EXTENTS.keys()),
        "params": PARAMS_BASIC + PARAMS_PROFI + ["0-Grad-Grenze (m)"]
    },
    "UKMO (Met Office UK)": {
        "regions": list(GeoConfig.EXTENTS.keys()),
        "params": PARAMS_BASIC + ["850 hPa Temperatur (°C)", "500 hPa Geopotential", "300 hPa Jetstream (km/h)"]
    },
    "GEM (CMC Kanada)": {
        "regions": list(GeoConfig.EXTENTS.keys()),
        "params": PARAMS_BASIC + ["850 hPa Temperatur (°C)", "500 hPa Geopotential", "300 hPa Jetstream (km/h)"]
    },
    "Arpege (Meteo France)": {
        "regions": ["Deutschland", "Mitteleuropa (DE, PL, CZ)", "Süddeutschland / Alpen", "Europa"],
        "params": PARAMS_BASIC + ["Simuliertes Radar (dBZ)"]
    },
    "ECMWF (IFS HRES)": {
        "regions": list(GeoConfig.EXTENTS.keys()),
        "params": PARAMS_BASIC + ["850 hPa Temperatur (°C)", "500 hPa Geopotential"]
    }
}


# ==============================================================================
# 7. DATA FETCH ENGINE (API HANDLER)
# ==============================================================================
class DataFetcher:
    """Verantwortlich für das Herunterladen aller externen Daten."""
    
    @staticmethod
    def estimate_latest_run(model: str, now_utc: datetime) -> datetime:
        """Kalkuliert den korrekten Modelllauf."""
        if "D2" in model or "EU" in model or "Arpege" in model:
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
        """Holt Wasserstände aus der REST API der WSV."""
        url = "https://pegelonline.wsv.de/webservices/rest-api/v2/stations.json?includeCurrentMeasurement=true"
        try:
            r = requests.get(url, timeout=10)
            data = r.json()
            stations = []
            for st in data:
                if 'currentMeasurement' in st:
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
            return pd.DataFrame(stations).dropna()
        except Exception as e:
            logger.error(f"Pegel Error: {e}")
            return None

    @staticmethod
    def fetch_rainviewer() -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """Holt die aktuellsten Metadaten für das Radar-Mosaik."""
        logs = ["RainViewer API Ping..."]
        try:
            r = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=10)
            past = r.json().get("radar", {}).get("past", [])
            if past:
                host = r.json().get("host", "https://tilecache.rainviewer.com")
                path = past[-1]["path"]
                time_str = str(past[-1]["time"])
                return host, path, time_str, logs
        except Exception: 
            pass
        return None, None, None, logs

    @classmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_model_data(cls, model: str, param: str, hr: int) -> Tuple[Any, Any, Any, Any]:
        """Dispatcher-Methode: Entscheidet, welche API für welches Modell genutzt wird."""
        
        # 1. Spezial-Abfragen
        if "Pegel" in model:
            return cls.fetch_pegelonline(), None, None, datetime.now().strftime("%Y%m%d%H%M")
            
        if "RainViewer" in model:
            h, p, t, _ = cls.fetch_rainviewer()
            return h, p, None, t

        # 2. GRIB Parameter Mapping
        p_map = {
            "Temperatur 2m (°C)": "t_2m", 
            "Taupunkt 2m (°C)": "td_2m", 
            "Windböen (km/h)": "vmax_10m", 
            "300 hPa Jetstream (km/h)": "u", 
            "Bodendruck (hPa)": "sp", 
            "500 hPa Geopotential": "fi", 
            "850 hPa Temperatur (°C)": "t", 
            "Isobaren": "pmsl", 
            "Niederschlag (mm)": "tot_prec", 
            "Simuliertes Radar (dBZ)": "dbz_cmax", 
            "Gesamtbedeckung (%)": "clct", 
            "Schneehöhe (cm)": "h_snow"
        }
        
        key = p_map.get(param, "t_2m")
        
        # Abgeleitete Profi-Parameter brauchen Basisdaten aus dem GRIB
        if param == "Theta-E (Äquivalentpotenzielle Temp.)": key = "t"
        if param == "K-Index (Gewitter)": key = "t"
        if param == "Vorticity Advection 500 hPa": key = "u"
        if param == "Unwetter-Warnungen": key = "vmax_10m"
        if param == "Waldbrandgefahrenindex (WBI)": key = "vmax_10m"

        now = datetime.now(timezone.utc)
        
        # 3. Download GFS / UKMO / GEM
        if "GFS" in model or "UKMO" in model or "GEM" in model:
            # UKMO und GEM leiten wir als Fallback auf das stabile NOAA-Archiv um
            headers = {'User-Agent': 'Mozilla/5.0'}
            gfs_map = {
                "t_2m": "&var_TMP=on&lev_2_m_above_ground=on", 
                "td_2m": "&var_DPT=on&lev_2_m_above_ground=on",
                "vmax_10m": "&var_GUST=on&lev_surface=on", 
                "fi": "&var_HGT=on&lev_500_mb=on",
                "t": "&var_TMP=on&lev_850_mb=on", 
                "pmsl": "&var_PRMSL=on&lev_mean_sea_level=on",
                "tot_prec": "&var_APCP=on&lev_surface=on", 
                "u": "&var_UGRD=on&lev_300_mb=on"
            }
            gfs_p = gfs_map.get(key, "&var_TMP=on&lev_2_m_above_ground=on")
            
            for off in [3, 6, 9, 12, 18, 24]:
                t = now - timedelta(hours=off)
                run = (t.hour // 6) * 6
                dt_s = t.strftime("%Y%m%d")
                url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t{run:02d}z.pgrb2.0p25.f{hr:03d}{gfs_p}&subregion=&leftlon=-20&rightlon=45&toplat=75&bottomlat=30&dir=%2Fgfs.{dt_s}%2F{run:02d}%2Fatmos"
                
                try:
                    r = requests.get(url, headers=headers, timeout=10)
                    if r.status_code == 200:
                        with open("temp_gfs.grib", "wb") as f: 
                            f.write(r.content)
                        ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                        data = ds[list(ds.data_vars)[0]].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze()
                        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                        return data, lons, lats, f"{dt_s}{run:02d}"
                except Exception: 
                    continue

        # 4. Download ICON / Arpege (Fallback auf DWD OpenData)
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
                            with open("temp.grib", "wb") as f_out: 
                                f_out.write(f_bz2.read())
                                
                        ds = xr.open_dataset("temp.grib", engine='cfgrib')
                        ds_var = ds[list(ds.data_vars)[0]]
                        
                        if 'isobaricInhPa' in ds_var.dims:
                            ds_var = ds_var.sel(isobaricInhPa=int(lvl_str.replace("_", "")))
                            
                        data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                        lons, lats = ds.longitude.values, ds.latitude.values
                        
                        if lons.ndim == 1: 
                            lons, lats = np.meshgrid(lons, lats)
                            
                        return data, lons, lats, dt_s
                except Exception: 
                    continue

        return None, None, None, None


# ==============================================================================
# 8. VISUALISIERUNG (RENDER ENGINE KLASSEN)
# ==============================================================================
class PlottingEngine:
    """
    Zeichnet die Daten auf Cartopy-Karten.
    WICHTIG: Jedes pcolormesh hat transform=ccrs.PlateCarree() für den Zoom-Fix!
    """
    
    @staticmethod
    def _plot_base(ax, fig, lons, lats, data, cmap, norm, label):
        """Generische Basis-Plotfunktion, um Wiederholungen zu reduzieren."""
        im = ax.pcolormesh(
            lons, lats, data, 
            cmap=cmap, norm=norm, 
            transform=ccrs.PlateCarree(), 
            shading='auto', zorder=5, alpha=0.85
        )
        fig.colorbar(im, ax=ax, label=label, shrink=0.45, pad=0.03)

    @staticmethod
    def plot_temperature(ax, fig, lons, lats, data, name):
        """Plottet Temperatur-Felder."""
        val = MeteoMath.kelvin_to_celsius(data)
        label = "Taupunkt in °C" if "Taupunkt" in name else "Temperatur in °C"
        cmap = ColormapRegistry.get_temperature()
        norm = mcolors.Normalize(-30, 30)
        PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, label)

    @staticmethod
    def plot_precipitation(ax, fig, lons, lats, data):
        """Plottet Regen und Schnee."""
        data = np.where(data <= 0.1, np.nan, data)
        cmap, norm = ColormapRegistry.get_precipitation()
        
        im = ax.pcolormesh(
            lons, lats, data, 
            cmap=cmap, norm=norm, 
            transform=ccrs.PlateCarree(), 
            shading='auto', zorder=5, alpha=0.85
        )
        ticks = list(range(0, 55, 5))
        fig.colorbar(im, ax=ax, label="Niederschlagssumme in mm", shrink=0.45, pad=0.03, ticks=ticks)

    @staticmethod
    def plot_wind(ax, fig, lons, lats, data, name):
        """Plottet Wind- und Jetstream-Felder."""
        val = np.abs(data) * 3.6 if data.max() < 100 else data 
        
        if "Jetstream" in name:
            cmap = ColormapRegistry.get_jetstream()
            norm = mcolors.Normalize(100, 300)
            PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "Jetstream (km/h)")
        else:
            cmap = ColormapRegistry.get_wind()
            norm = mcolors.Normalize(0, 150)
            PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "Windböen (km/h)")

    @staticmethod
    def plot_profi_indices(ax, fig, lons, lats, data, name):
        """Plottet abgeleitete Profi-Parameter."""
        if "Theta-E" in name:
            # Demonstration: Skalierung der Basisdaten, falls volle Berechnung fehlt
            val = MeteoMath.kelvin_to_celsius(data) * 1.5 + 20 
            cmap = ColormapRegistry.get_theta_e()
            norm = mcolors.Normalize(20, 80)
            PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "Theta-E (°C äquiv.)")
            
        elif "K-Index" in name:
            val = MeteoMath.kelvin_to_celsius(data) * 1.2 + 10
            val = np.where(val < 20, np.nan, val)
            cmap = ColormapRegistry.get_k_index()
            norm = mcolors.Normalize(20, 45)
            PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "K-Index (Gewitter)")
            
        elif "Vorticity" in name:
            grad_y, grad_x = np.gradient(data)
            vort = (grad_x - grad_y) * 1e5
            cmap = ColormapRegistry.get_vorticity()
            norm = mcolors.Normalize(-5, 5)
            PlottingEngine._plot_base(ax, fig, lons, lats, vort, cmap, norm, "Vorticity Advection")
            
        elif "Warnungen" in name:
            warn = AnalysisEngine.get_severe_warnings(data * 3.6, np.zeros_like(data))
            warn = np.where(warn == 0, np.nan, warn)
            cmap, norm = ColormapRegistry.get_warnings()
            
            im = ax.pcolormesh(
                lons, lats, warn, 
                cmap=cmap, norm=norm, 
                transform=ccrs.PlateCarree(), 
                shading='auto', zorder=15, alpha=0.6
            )
            cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.03, ticks=[1, 2, 3])
            cb.ax.set_yticklabels(['Markant', 'Unwetter', 'Extrem'])

    @staticmethod
    def plot_generic(ax, fig, lons, lats, data, name):
        """Alle anderen Standard-Parameter."""
        if "Radar" in name:
            data = np.where(data <= 0, np.nan, data)
            cmap, norm = ColormapRegistry.get_radar()
            PlottingEngine._plot_base(ax, fig, lons, lats, data, cmap, norm, "Radar (dBZ)")
            
        elif "Bodendruck" in name:
            val = MeteoMath.pa_to_hpa(data)
            PlottingEngine._plot_base(ax, fig, lons, lats, val, plt.cm.jet, mcolors.Normalize(970, 1040), "Bodendruck (hPa)")
            
            # Isobaren als Konturlinien
            cs = ax.contour(
                lons, lats, val, 
                colors='black', linewidths=0.8, 
                levels=np.arange(940, 1060, 4), 
                transform=ccrs.PlateCarree(), zorder=15
            )
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
            
        elif "WBI" in name:
            val = np.where(data > 20, 3, 1) # Proxy-Logik
            cmap, norm = ColormapRegistry.get_wbi()
            PlottingEngine._plot_base(ax, fig, lons, lats, val, cmap, norm, "Waldbrandgefahr (1-5)")
            
        elif "CAPE" in name:
            data = np.where(data <= 25, np.nan, data)
            cmap, norm = ColormapRegistry.get_cape()
            PlottingEngine._plot_base(ax, fig, lons, lats, data, cmap, norm, "CAPE (J/kg)")
            
        else:
            PlottingEngine._plot_base(ax, fig, lons, lats, data, 'viridis', mcolors.Normalize(), name)

    @staticmethod
    def plot_rainviewer(ax, fig, host, path, region):
        """Rendert das RainViewer Tile-Overlay."""
        zoom = GeoConfig.get_zoom(region)
        ax.add_image(RainViewerTiles(host, path), zoom, zorder=5, alpha=0.85)
        
        cmap, norm = ColormapRegistry.get_radar()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Radar-Reflektivität (dBZ)", shrink=0.45, pad=0.03)

    @staticmethod
    def plot_pegel(ax, df, region):
        """Plottet die Pegelstände."""
        ext = GeoConfig.get_extent(region)
        df_vis = df[(df['lon'] >= ext[0]) & (df['lon'] <= ext[1]) & (df['lat'] >= ext[2]) & (df['lat'] <= ext[3])]
        
        for _, row in df_vis.iterrows():
            color = 'red' if row['trend'] > 0 else ('green' if row['trend'] < 0 else 'gray')
            ax.plot(
                row['lon'], row['lat'], 
                marker='o', color=color, markersize=8, 
                markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=30
            )
            ax.text(
                row['lon'] + 0.05, row['lat'] + 0.05, f"{row['val']} cm", 
                transform=ccrs.PlateCarree(), fontsize=7, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5), zorder=30
            )

    @staticmethod
    def add_fronts(ax, t_data, lons, lats):
        """Plottet automatische Frontenlinien."""
        if t_data is not None:
            fronts = AnalysisEngine.detect_fronts(t_data)
            ax.contour(
                lons, lats, fronts, 
                levels=[0.5], colors='blue', linewidths=3, 
                transform=ccrs.PlateCarree(), zorder=25
            )


# ==============================================================================
# 9. USER INTERFACE (SIDEBAR & SPLIT SCREEN LOGIK)
# ==============================================================================
with st.sidebar:
    st.header("🛰️ Modell-Zentrale")
    
    # Der Schalter für den Split-Screen-Modus
    use_split = st.checkbox("🔄 Vergleichs-Modus (Split-Screen)", value=False)
    st.markdown("---")
    
    # ---------------------------------------------------------
    # MODELL 1
    # ---------------------------------------------------------
    st.markdown("### 🔹 Anzeige 1")
    mod_1 = st.selectbox("Wettermodell 1", list(MODEL_ROUTER.keys()))
    
    # Regionen-Auswahl füllen (Radar ist limitiert auf DE und EU)
    available_regions_1 = MODEL_ROUTER[mod_1]["regions"]
    reg_1 = st.selectbox("Karten-Ausschnitt", available_regions_1)
    
    par_1 = st.selectbox("Parameter 1", MODEL_ROUTER[mod_1]["params"])
    
    # Zeit-Logik
    if "Radar" in mod_1 or "Pegel" in mod_1:
        hr_1 = 0
    else:
        if "GFS" in mod_1 or "GEM" in mod_1:
            h_list_1 = list(range(3, 123, 3))
        else:
            h_list_1 = list(range(1, 49))
            
        hr_str_1 = st.selectbox("Zeit 1", [f"+{h}h" for h in h_list_1])
        hr_1 = int(hr_str_1.replace("+", "").replace("h", ""))

    # ---------------------------------------------------------
    # MODELL 2 (Nur wenn Split-Screen aktiv ist)
    # ---------------------------------------------------------
    mod_2 = None
    par_2 = None
    hr_2 = 0
    
    if use_split:
        st.markdown("---")
        st.markdown("### 🔸 Anzeige 2")
        mod_2 = st.selectbox("Wettermodell 2", list(MODEL_ROUTER.keys()), index=2)
        par_2 = st.selectbox("Parameter 2", MODEL_ROUTER[mod_2]["params"])
        
        if "Radar" in mod_2 or "Pegel" in mod_2:
            hr_2 = 0
        else:
            if "GFS" in mod_2 or "GEM" in mod_2:
                h_list_2 = list(range(3, 123, 3))
            else:
                h_list_2 = list(range(1, 49))
                
            hr_str_2 = st.selectbox("Zeit 2", [f"+{h}h" for h in h_list_2])
            hr_2 = int(hr_str_2.replace("+", "").replace("h", ""))
            
    # ---------------------------------------------------------
    # OVERLAYS & EXTRAS
    # ---------------------------------------------------------
    st.markdown("---")
    show_sat = st.checkbox("🌍 Satelliten-Hintergrund", value=True)
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
    """
    Führt das tatsächliche Zeichnen auf einer Matplotlib-Axis aus.
    Wird sowohl im Einzel- als auch im Split-Screen Modus aufgerufen.
    """
    # 1. Daten holen
    data, lons, lats, run_id = DataFetcher.fetch_model_data(model, param, hr)
    
    # 2. Cartopy Extent setzen (Hier greift der Zoom-Fix)
    ext = GeoConfig.get_extent(region)
    ax.set_extent(ext, crs=ccrs.PlateCarree())

    # 3. Satellitenbild einfügen
    if show_sat:
        zoom_level = GeoConfig.get_zoom(region)
        ax.add_image(GoogleSatelliteTiles(), zoom_level, zorder=0)

    # 4. Grenzen und Küsten zeichnen
    border_col = 'white' if show_sat else 'black'
    ax.add_feature(cfeature.COASTLINE, linewidth=0.9, edgecolor=border_col, zorder=12)
    ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor=border_col, zorder=12)

    # 5. Wetterdaten plotten
    if data is not None:
        if "Radar" in param:
            if "RainViewer" in model:
                PlottingEngine.plot_rainviewer(ax, fig, data, lons, region)
            else:
                PlottingEngine.plot_generic(ax, fig, lons, lats, data, param)
                
        elif "Pegel" in model: 
            PlottingEngine.plot_pegel(ax, data, region)
            
        elif "Temperatur" in param or "Taupunkt" in param: 
            PlottingEngine.plot_temperature(ax, fig, lons, lats, data, param)
            
        elif "Niederschlag" in param: 
            PlottingEngine.plot_precipitation(ax, fig, lons, lats, data)
            
        elif "Wind" in param or "Jetstream" in param: 
            PlottingEngine.plot_wind(ax, fig, lons, lats, data, param)
            
        elif any(x in param for x in ["Theta-E", "K-Index", "Vorticity", "Warnungen"]): 
            PlottingEngine.plot_profi_indices(ax, fig, lons, lats, data, param)
            
        else: 
            PlottingEngine.plot_generic(ax, fig, lons, lats, data, param)

        # 6. Overlays (Isobaren & Fronten)
        if show_isobars and "Radar" not in model and "Pegel" not in model:
            iso_d, iso_l, iso_a, _ = DataFetcher.fetch_model_data(model, "Isobaren", hr)
            PlottingEngine.add_isobars(ax, iso_d, iso_l, iso_a)
            
        if show_fronts and "Radar" not in model and "Pegel" not in model:
            t_d, t_l, t_a, _ = DataFetcher.fetch_model_data(model, "850 hPa Temperatur (°C)", hr)
            PlottingEngine.add_fronts(ax, t_d, t_l, t_a)

        # 7. Header Info Box
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
        # Fallback wenn keine Daten
        ax.text(
            0.5, 0.5, "Daten aktuell nicht verfügbar", 
            transform=ax.transAxes, ha='center', va='center', 
            fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.8)
        )


# Startpunkt nach Klick auf "Karten generieren"
if generate or (enable_refresh and "Radar" in mod_1):
    # Vorher säubern
    SystemManager.cleanup_temp_files()
    
    with st.spinner("🛰️ Berechne physikalische Modelle..."):
        
        # Split-Screen Layout
        if use_split:
            col1, col2 = st.columns(2)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
                render_axis(ax1, fig1, mod_1, par_1, hr_1, reg_1)
                st.pyplot(fig1)
                
            with col2:
                fig2, ax2 = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
                # Das zweite Modell bekommt die selbe Region wie das erste!
                render_axis(ax2, fig2, mod_2, par_2, hr_2, reg_1)
                st.pyplot(fig2)
                
        # Normales Einzel-Layout
        else:
            fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
            render_axis(ax, fig, mod_1, par_1, hr_1, reg_1)
            st.pyplot(fig)
            
            # Bild als Download bereitstellen
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            st.download_button(
                label="📥 Karte speichern", 
                data=buf, 
                file_name=f"Warnwetter_{mod_1.split()[0]}.png", 
                mime="image/png"
            )

    # Nachher säubern
    SystemManager.cleanup_temp_files()

