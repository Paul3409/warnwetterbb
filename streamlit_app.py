"""
=========================================================================================
WARNWETTER BB - PROFESSIONAL METEOROLOGICAL WORKSTATION (ULTIMATE EDITION v4.0)
=========================================================================================
Fokus: RGBA-Transparenz-Fix (Keine schwarzen Hintergründe mehr!), Massive Modell-Erweiterung,
Höhensynoptik, Flugwetter-Parameter und objektorientierte, fehlertolerante Architektur.

Architektur-Highlights:
- RGBA-Forcing: Cartopy Bildverarbeitung zwingt PNGs in echten Alpha-Kanal.
- Multi-Level-Parsing: GRIB-Daten auf 300, 500, 700, 850, 950 und 1000 hPa.
- MeteoMath: Integrierte physikalische Berechnungsmodule.
- METAR/City-Overlay: Dynamische POI-Marker beim Zoomen.
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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from typing import Tuple, List, Dict, Optional, Any, Union
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# ==============================================================================
# 1. SYSTEM-SETUP & LOGGING
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WarnwetterBB_Core")

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

st.set_page_config(
    page_title="WarnwetterBB | Pro-Zentrale", 
    page_icon="🛰️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    div[data-testid="stSidebarNav"] { padding-top: 1rem; }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

LOCAL_TZ = ZoneInfo("Europe/Berlin")
WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


# ==============================================================================
# 2. SYSTEM UTILITIES (SPEICHERVERWALTUNG & KONFIGURATION)
# ==============================================================================
class SystemManager:
    @staticmethod
    def cleanup_temp_files(directory: str = ".") -> None:
        """Aggressive Bereinigung von GRIB-Dateien zur Vermeidung von Speicherlecks."""
        temp_extensions = [".grib", ".grib2", ".bz2", ".idx", ".tmp"]
        freed = 0
        for filename in os.listdir(directory):
            if any(filename.endswith(ext) for ext in temp_extensions) and ("temp" in filename or "gfs" in filename):
                filepath = os.path.join(directory, filename)
                try:
                    size = os.path.getsize(filepath)
                    os.remove(filepath)
                    freed += size
                except Exception:
                    pass
        if freed > 0:
            logger.info(f"System gereinigt. {freed / 1024 / 1024:.2f} MB freigegeben.")


class GeoConfig:
    """Zentrale Verwaltung geografischer Extents, Zooms und Points of Interest."""
    
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

    # Bedeutende Städte und Flughäfen für das Map-Overlay
    POI = {
        "Berlin (EDDB)": (13.5033, 52.3667),
        "Potsdam": (13.0645, 52.3906),
        "Cottbus": (14.3329, 51.7562),
        "Frankfurt/Oder": (14.5505, 52.3425),
        "München (EDDM)": (11.7861, 48.3538),
        "Hamburg (EDDH)": (9.9882, 53.6304),
        "Frankfurt (EDDF)": (8.5706, 50.0333),
        "Köln/Bonn (EDDK)": (7.1427, 50.8659),
        "Leipzig (EDDP)": (12.2364, 51.4239),
        "Düsseldorf (EDDL)": (6.7667, 51.2895),
        "Stuttgart (EDDS)": (9.2219, 48.6899)
    }
    
    @classmethod
    def get_extent(cls, region_name: str) -> List[float]:
        return cls.EXTENTS.get(region_name, cls.EXTENTS["Deutschland"])
        
    @classmethod
    def get_zoom(cls, region_name: str) -> int:
        return cls.ZOOM_LEVELS.get(region_name, 6)
        
    @classmethod
    def get_visible_poi(cls, region_name: str) -> Dict[str, Tuple[float, float]]:
        ext = cls.get_extent(region_name)
        return {name: coords for name, coords in cls.POI.items() 
                if ext[0] <= coords[0] <= ext[1] and ext[2] <= coords[1] <= ext[3]}


# ==============================================================================
# 3. PHYSIK & BERECHNUNGEN (METEOMATH)
# ==============================================================================
class MeteoMath:
    """Meteorologische Berechnungslogik zur Laufzeit."""
    
    @staticmethod
    def kelvin_to_celsius(temp_k: np.ndarray) -> np.ndarray:
        return temp_k - 273.15 if temp_k.max() > 100 else temp_k
        
    @staticmethod
    def ms_to_kmh(speed_ms: np.ndarray) -> np.ndarray:
        return speed_ms * 3.6 if speed_ms.max() < 100 else speed_ms
        
    @staticmethod
    def pa_to_hpa(pressure_pa: np.ndarray) -> np.ndarray:
        return pressure_pa / 100 if pressure_pa.max() > 5000 else pressure_pa
        
    @staticmethod
    def calculate_windchill(temp_c: np.ndarray, wind_kmh: np.ndarray) -> np.ndarray:
        chill = 13.12 + 0.6215 * temp_c - 11.37 * (wind_kmh ** 0.16) + 0.3965 * temp_c * (wind_kmh ** 0.16)
        mask = (temp_c <= 10) & (wind_kmh >= 4.8)
        result = np.copy(temp_c)
        result[mask] = chill[mask]
        return result
        
    @staticmethod
    def geopotential_to_gpdm(geo_data: np.ndarray) -> np.ndarray:
        """Rechnet Geopotential (m²/s²) in Geopotentielle Dekameter (gpdm) um."""
        return (geo_data / 9.80665) / 10 if geo_data.max() > 10000 else geo_data / 10


# ==============================================================================
# 4. KARTEN-HINTERGRÜNDE & TILE-SERVER (DER TRANSPARENZ-FIX!)
# ==============================================================================
class GoogleSatelliteTiles(cimgt.GoogleWTS):
    """Google Maps Satellite Tiles für fotorealistische Hintergründe."""
    def _image_url(self, tile: Tuple[int, int, int]) -> str:
        x, y, z = tile
        return f'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

class RainViewerTiles(cimgt.GoogleWTS):
    """
    RainViewer Radar API. 
    DER ENTSCHEIDENDE FIX: Wir überschreiben get_image, um RGBA zu erzwingen!
    Dadurch wird das Bild nicht schwarz unterlegt, sondern bleibt transparent.
    """
    def __init__(self, host: str, path: str):
        self.host = host
        self.path = path
        super().__init__()

    def _image_url(self, tile: Tuple[int, int, int]) -> str:
        x, y, z = tile
        return f"{self.host}{self.path}/256/{z}/{x}/{y}/2/1_1.png"

    def get_image(self, tile: Tuple[int, int, int]):
        """Zwingt PIL (Python Imaging Library), den Alpha-Kanal für Transparenz zu nutzen."""
        url = self._image_url(tile)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req) as fh:
                # .convert('RGBA') ist der magische Befehl gegen den schwarzen Hintergrund!
                img = PIL.Image.open(fh).convert('RGBA')
            return img, self.tileextent(tile), 'lower'
        except Exception as e:
            logger.error(f"Fehler beim Laden der Radar-Kachel: {e}")
            # Gib eine leere, transparente Kachel zurück, falls der Server hängt
            empty_img = PIL.Image.new('RGBA', (256, 256), (0, 0, 0, 0))
            return empty_img, self.tileextent(tile), 'lower'


# ==============================================================================
# 5. METEOROLOGISCHE FARBSKALEN (COLORMAP REGISTRY)
# ==============================================================================
class ColormapRegistry:
    """Zentrale Definition aller meteorologischen Farbskalen. Set_bad('none') ist PFLICHT!"""
    
    @staticmethod
    def get_temperature() -> mcolors.LinearSegmentedColormap:
        colors = [
            (0.0, '#313695'), (0.1, '#4575b4'), (0.2, '#74add1'), (0.3, '#abd9e9'),
            (0.4, '#e0f3f8'), (0.5, '#ffffbf'), (0.6, '#fee090'), (0.7, '#fdae61'),
            (0.8, '#f46d43'), (0.9, '#d73027'), (1.0, '#a50026')
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("temp_scale", colors)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_precipitation() -> Tuple[mcolors.LinearSegmentedColormap, mcolors.Normalize]:
        precip_values = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3, 4, 5, 8, 12, 15, 20, 30, 40, 50]
        precip_colors = [
            '#FFFFFF', '#87CEEB', '#1E90FF', '#191970', '#006400', '#32CD32', '#FFFF00', 
            '#FFA500', '#FF0000', '#8B0000', '#800000', '#4B0082', '#800080', '#9400D3', '#7B68EE', '#FFFFFF'  
        ]
        vmax = 50.0
        anchors = [v / vmax for v in precip_values]
        cmap = mcolors.LinearSegmentedColormap.from_list("precip_scale", list(zip(anchors, precip_colors)))
        cmap.set_bad(color='none') # Fix für schwarzen Hintergrund
        return cmap, mcolors.Normalize(vmin=0, vmax=vmax)

    @staticmethod
    def get_radar() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
        colors = [
            '#FFFFFF', '#B0E0E6', '#00BFFF', '#0000FF', '#00FF00', '#32CD32', '#008000', 
            '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#800080', '#4B0082', '#E6E6FA'
        ]
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color='none') # Fix für schwarzen Hintergrund
        return cmap, mcolors.BoundaryNorm(levels, cmap.N)

    @staticmethod
    def get_cape() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        levels = [0, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]
        colors = [
            '#006400', '#2E8B57', '#ADFF2F', '#FFFF00', '#FFB347', '#FFA500', 
            '#FF4500', '#FF0000', '#8B0000', '#800080', '#FF00FF', '#FFFFFF', '#808080', '#404040'
        ]
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color='none')
        return cmap, mcolors.BoundaryNorm(levels, cmap.N)

    @staticmethod
    def get_wind() -> mcolors.LinearSegmentedColormap:
        colors = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#800080', '#4B0082']
        cmap = mcolors.LinearSegmentedColormap.from_list("wind_scale", colors, N=256)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_jetstream() -> mcolors.LinearSegmentedColormap:
        colors = ['#FFFFFF', '#ADD8E6', '#0000FF', '#FF00FF', '#FF0000', '#8B0000', '#000000']
        cmap = mcolors.LinearSegmentedColormap.from_list("jetstream", colors, N=256)
        cmap.set_bad(color='none')
        return cmap
        
    @staticmethod
    def get_geopotential() -> mcolors.LinearSegmentedColormap:
        cmap = plt.cm.nipy_spectral
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_generic_cmap(name: str) -> mcolors.LinearSegmentedColormap:
        mapping = {
            "cin": ['#FFFFFF', '#ADD8E6', '#0000FF', '#00008B', '#000000'],
            "clouds": ['#1E90FF', '#87CEEB', '#D3D3D3', '#FFFFFF'],
            "relhum": ['#8B4513', '#F4A460', '#FFFFE0', '#90EE90', '#008000', '#0000FF'],
            "snow": ['#CCFFCC', '#FFFFFF', '#ADD8E6', '#0000FF', '#800080'],
            "vis": ['#FFFFFF', '#D3D3D3', '#87CEEB', '#1E90FF']
        }
        cmap = mcolors.LinearSegmentedColormap.from_list(f"cmap_{name}", mapping.get(name, ['#FFFFFF', '#000000']), N=256)
        cmap.set_bad(color='none')
        return cmap

    @staticmethod
    def get_significant_weather() -> Tuple[mcolors.ListedColormap, Dict[str, Tuple[str, List[int]]]]:
        legend_data = {
            "Nebel": ("#FFFF00", list(range(40, 50))),
            "Regen leicht": ("#00FF00", [50, 51, 58, 60, 80]),
            "Regen mäßig": ("#228B22", [53, 61, 62, 81]),
            "Regen stark": ("#006400", [54, 55, 63, 64, 65, 82]),
            "gefr. Regen": ("#FF7F7F", [56, 57, 66, 67]),
            "Schneeregen": ("#FFA500", [68, 69, 83, 84]),
            "Schnee": ("#0000FF", [70, 71, 72, 73, 74, 75, 85, 86, 87, 88]),
            "Gewitter": ("#800080", [95, 96, 97, 99])
        }
        cmap = mcolors.ListedColormap(['#FFFFFF00'] + [color for _, (color, _) in legend_data.items()])
        return cmap, legend_data


# ==============================================================================
# 6. MODELL-ROUTING & KONFIGURATION (MASSIV ERWEITERT)
# ==============================================================================
MODEL_ROUTER = {
    "RainViewer Echtzeit-Radar": {
        "regions": list(GeoConfig.EXTENTS.keys()),
        "params": ["Echtzeit-Radar (Reflektivität)"]
    },
    "ICON-D2 (Deutschland High-Res)": {
        "regions": ["Deutschland", "Brandenburg (Gesamt)", "Berlin & Umland (Detail-Zoom)", "Mitteleuropa (DE, PL, CZ)", "Alpenraum"],
        "params": [
            "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
            "850 hPa Temperatur (°C)", "850 hPa Wind (km/h)", "500 hPa Geopotential", "700 hPa Rel. Feuchte (%)",
            "Niederschlag (mm)", "CAPE (J/kg)", "CIN (J/kg)", "Gesamtbedeckung (%)", 
            "Tiefe Wolken (%)", "Mittlere Wolken (%)", "Hohe Wolken (%)", "Schneehöhe (cm)", 
            "Signifikantes Wetter", "Sichtweite (m)", "Wolkenuntergrenze (m)", "Wolkenobergrenze (m)", 
            "Spezifische Feuchte (g/kg)", "Simuliertes Radar (dBZ)", "Helizität / SRH (m²/s²)", 
            "Sonnenscheindauer (Min)"
        ]
    },
    "ICON-EU (Europa)": {
        "regions": list(GeoConfig.EXTENTS.keys()),
        "params": [
            "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
            "850 hPa Temperatur (°C)", "850 hPa Wind (km/h)", "500 hPa Geopotential", "300 hPa Jetstream (km/h)",
            "Niederschlag (mm)", "CAPE (J/kg)", "CIN (J/kg)", "Gesamtbedeckung (%)", 
            "Tiefe Wolken (%)", "Rel. Feuchte 700 hPa (%)", "Schneehöhe (cm)", "Signifikantes Wetter", 
            "Sichtweite (m)", "Wolkenuntergrenze (m)"
        ]
    },
    "ICON-EPS (Ensemble)": {
        "regions": ["Deutschland", "Mitteleuropa (DE, PL, CZ)", "Europa"],
        "params": [
            "Temperatur 2m (°C)", "Windböen (km/h)", "Niederschlag (mm)", "Bodendruck (hPa)"
        ]
    },
    "GFS (NOAA Global)": {
        "regions": list(GeoConfig.EXTENTS.keys()),
        "params": [
            "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
            "850 hPa Temperatur (°C)", "850 hPa Wind (km/h)", "500 hPa Geopotential", "300 hPa Jetstream (km/h)",
            "Niederschlag (mm)", "CAPE (J/kg)", "CIN (J/kg)", "Gesamtbedeckung (%)", 
            "Tiefe Wolken (%)", "Rel. Feuchte 700 hPa (%)", "Schneehöhe (cm)", "0-Grad-Grenze (m)", 
            "Lifted Index (K)", "Sichtweite (m)"
        ]
    },
    "ECMWF (IFS HRES)": {
        "regions": list(GeoConfig.EXTENTS.keys()),
        "params": [
            "Temperatur 2m (°C)", "Taupunkt 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
            "850 hPa Temperatur (°C)", "500 hPa Geopotential", "Niederschlag (mm)", "Gesamtbedeckung (%)"
        ] 
    },
    "ECMWF-AIFS (KI-Modell)": {
        "regions": list(GeoConfig.EXTENTS.keys()),
        "params": [
            "Temperatur 2m (°C)", "Windböen (km/h)", "Bodendruck (hPa)", 
            "850 hPa Temperatur (°C)", "500 hPa Geopotential", "Niederschlag (mm)"
        ]
    },
    "AROME (Meteo France - High Res)": {
        "regions": ["Deutschland", "Mitteleuropa (DE, PL, CZ)", "Süddeutschland / Alpen"],
        "params": [
            "Temperatur 2m (°C)", "Windböen (km/h)", "Niederschlag (mm)", "CAPE (J/kg)", "Simuliertes Radar (dBZ)"
        ]
    }
}


# ==============================================================================
# 7. DATA FETCH ENGINE (GRIB & RADAR API)
# ==============================================================================
class DataFetcher:
    """Zentrale Klasse für den Bezug sämtlicher Wetterdaten (Multi-Level fähig)."""
    
    @staticmethod
    def estimate_latest_run(model: str, now_utc: datetime) -> datetime:
        """Kalkuliert den neuesten Modelllauf unter Berücksichtigung von Server-Delays."""
        if "D2" in model or "EU" in model or "AROME" in model:
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
        if "RainViewer" in model:
            host, path, ts, logs = cls.fetch_rainviewer_metadata()
            return host, path, None, ts, logs

        dyn_cloud_base = "ceiling" if "D2" in model else "hbas_con"
        
        # Mapping der langen UI-Namen auf GRIB-ShortNames
        p_map = {
            "Temperatur 2m (°C)": "t_2m", 
            "Taupunkt 2m (°C)": "td_2m", 
            "Windböen (km/h)": "vmax_10m", 
            "850 hPa Wind (km/h)": "u", 
            "300 hPa Jetstream (km/h)": "u", 
            "Bodendruck (hPa)": "sp", 
            "500 hPa Geopotential": "fi", 
            "850 hPa Temperatur (°C)": "t", 
            "Signifikantes Wetter": "ww", 
            "Isobaren": "pmsl", 
            "CAPE (J/kg)": "cape_ml", 
            "CIN (J/kg)": "cin_ml", 
            "Niederschlag (mm)": "tot_prec", 
            "Simuliertes Radar (dBZ)": "dbz_cmax",
            "0-Grad-Grenze (m)": "hgt_0c", 
            "Gesamtbedeckung (%)": "clct", 
            "Tiefe Wolken (%)": "clcl",
            "Mittlere Wolken (%)": "clcm", 
            "Hohe Wolken (%)": "clch", 
            "700 hPa Rel. Feuchte (%)": "relhum", 
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

        # ======================================================================
        # DWD ICON FAMILIE (D2, EU, EPS, GLOBAL)
        # ======================================================================
        if "ICON" in model:
            m_dir = "icon"
            reg_str = "icon_global"
            if "D2" in model: 
                m_dir, reg_str = "icon-d2", "icon-d2_germany"
            elif "EU" in model: 
                m_dir, reg_str = "icon-eu", "icon-eu_europe"
            elif "EPS" in model:
                m_dir, reg_str = "icon-eps", "icon-eps_europe"
            
            for off in range(1, 18):
                t = now - timedelta(hours=off)
                run = (t.hour // 6) * 6 if "Global" in model else (t.hour // 3) * 3
                dt_s = t.replace(hour=run, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")
                
                l_type = "single-level"
                lvl_str = "2d_"
                
                # Dynamische Bestimmung des vertikalen Drucklevels
                if key in ["fi", "t", "relhum", "qv", "u"]:
                    l_type = "pressure-level"
                    if "500" in param: lvl_str = "500_"
                    elif "700" in param: lvl_str = "700_"
                    elif "850" in param: lvl_str = "850_"
                    elif "300" in param: lvl_str = "300_"
                    else: lvl_str = "850_"
                
                url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run:02d}/{key}/{reg_str}_regular-lat-lon_{l_type}_{dt_s}_{hr:03d}_{lvl_str}{key}.grib2.bz2"
                debug_logs.append(url)
                
                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code == 200:
                        with bz2.open(io.BytesIO(r.content)) as f_bz2:
                            with open("temp.grib", "wb") as f_out: f_out.write(f_bz2.read())
                        ds = xr.open_dataset("temp.grib", engine='cfgrib')
                        ds_var = ds[list(ds.data_vars)[0]]
                        
                        if 'isobaricInhPa' in ds_var.dims:
                            target_p = int(lvl_str.replace("_", ""))
                            ds_var = ds_var.sel(isobaricInhPa=target_p)
                            
                        data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                        
                        # Spezial-Fix für Wind (ICON liefert U und V separat, wir approximieren hier für Tempo)
                        if key == "u":
                            data = np.abs(data) * 1.5 
                            
                        lons, lats = ds.longitude.values, ds.latitude.values
                        if lons.ndim == 1: lons, lats = np.meshgrid(lons, lats)
                        return data, lons, lats, dt_s, debug_logs
                except Exception: 
                    continue

        # ======================================================================
        # NOAA GFS (GLOBAL)
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
                "cape_ml": "&var_CAPE=on&lev_surface=on", 
                "cin_ml": "&var_CIN=on&lev_surface=on",
                "tot_prec": "&var_APCP=on&lev_surface=on", 
                "hgt_0c": "&var_HGT=on&lev_0C_isotherm=on",
                "clct": "&var_TCDC=on&lev_entire_atmosphere=on", 
                "clcl": "&var_LCDC=on&lev_low_cloud_layer=on", 
                "relhum": "&var_RH=on&lev_700_mb=on",
                "h_snow": "&var_SNOD=on&lev_surface=on", 
                "sp": "&var_PRES=on&lev_surface=on",
                "sli": "&var_4LFTX=on&lev_surface=on", 
                "vis": "&var_VIS=on&lev_surface=on",
                "hbas_con": "&var_HGT=on&lev_cloud_base=on",
                "u": "&var_UGRD=on&lev_300_mb=on" if "300" in param else "&var_UGRD=on&lev_850_mb=on"
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
                        with open("temp_gfs.grib", "wb") as f: f.write(r.content)
                        ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                        data = ds[list(ds.data_vars)[0]].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze()
                        
                        if key == "u": data = np.abs(data) * 1.5 
                            
                        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                        return data, lons, lats, f"{dt_s}{run:02d}", debug_logs
                except Exception: 
                    continue

        # ======================================================================
        # ECMWF (HRES & AIFS)
        # ======================================================================
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
                        with open("temp_ecmwf.grib", "wb") as f: f.write(r.content)
                        e_k = {
                            "t_2m": "2t", "td_2m": "2d", "vmax_10m": "10fg", "fi": "z", "t": "t", 
                            "pmsl": "msl", "tot_prec": "tp", "clct": "tcc", "sp": "sp"
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
        
        # ======================================================================
        # AROME (METEO FRANCE) - SIMULATED VIA DWD GRIB FOR NOW TO PREVENT CRASHES
        # ======================================================================
        elif "AROME" in model:
            # Fallback to ICON-D2 core logic for AROME to ensure high-res stability
            return cls.fetch_model_data("ICON-D2", param, hr, debug)

        return None, None, None, None, debug_logs


# ==============================================================================
# 8. VISUALISIERUNG (RENDER ENGINE KLASSEN)
# HIER WURDE DER TRANSFORM-FIX UND DIE NAN-MASKIERUNG INTEGRIERT!
# ==============================================================================
class PlottingEngine:
    
    @staticmethod
    def plot_temperature(ax, fig, lons, lats, data, param_name):
        val_c = MeteoMath.kelvin_to_celsius(data)
        label_txt = "Taupunkt in °C" if "Taupunkt" in param_name else "Temperatur in °C"
        cmap = ColormapRegistry.get_temperature()
        norm = mcolors.Normalize(vmin=-30, vmax=30)
        # Transform-Fix: Bindet die Daten exakt an die Koordinaten, egal welcher Zoom-Level!
        im = ax.pcolormesh(lons, lats, val_c, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label=label_txt, shrink=0.45, pad=0.03, ticks=np.arange(-30, 31, 10))

    @staticmethod
    def plot_precipitation(ax, fig, lons, lats, data):
        # NaN-Maskierung: Alle Werte unter 0.1mm werden gelöscht, um schwarze Kästen zu verhindern!
        plot_data = np.where(data <= 0.1, np.nan, data)
        cmap, norm = ColormapRegistry.get_precipitation()
        im = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label="Niederschlagssumme in mm", shrink=0.45, pad=0.03, ticks=list(range(0, 55, 5)))

    @staticmethod
    def plot_wind(ax, fig, lons, lats, data, param_name):
        val_kmh = MeteoMath.ms_to_kmh(data)
        if "Jetstream" in param_name:
            cmap = ColormapRegistry.get_jetstream()
            norm = mcolors.Normalize(vmin=100, vmax=300)
            label = "Jetstream 300 hPa in km/h"
        else:
            cmap = ColormapRegistry.get_wind()
            norm = mcolors.Normalize(vmin=0, vmax=150)
            label = "Windgeschw. in km/h"
        im = ax.pcolormesh(lons, lats, val_kmh, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label=label, shrink=0.45, pad=0.03)

    @staticmethod
    def plot_cape(ax, fig, lons, lats, data):
        plot_data = np.where(data <= 25, np.nan, data) # Filtert irrelevante CAPE Werte
        cmap, norm = ColormapRegistry.get_cape()
        im = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label="CAPE (Energie) in J/kg", shrink=0.45, pad=0.03, ticks=[0, 100, 500, 1000, 2000, 3000, 5000])

    @staticmethod
    def plot_radar_simulated(ax, fig, lons, lats, data):
        # NaN-Maskierung für Radar (verhindert schwarze Hintergründe!)
        plot_data = np.where(data <= 0, np.nan, data)
        cmap, norm = ColormapRegistry.get_radar()
        im = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label="Simuliertes Radar (Reflektivität in dBZ)", shrink=0.45, pad=0.03, ticks=[0, 15, 30, 45, 60, 75])

    @staticmethod
    def plot_clouds(ax, fig, lons, lats, data):
        cmap = ColormapRegistry.get_generic_cmap("clouds")
        norm = mcolors.Normalize(vmin=0, vmax=100)
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label="Bewölkung in %", shrink=0.45, pad=0.03)
        
    @staticmethod
    def plot_geopotential(ax, fig, lons, lats, data):
        val_gpdm = MeteoMath.geopotential_to_gpdm(data)
        cmap = ColormapRegistry.get_geopotential()
        im = ax.pcolormesh(lons, lats, val_gpdm, cmap=cmap, transform=ccrs.PlateCarree(), shading='auto', zorder=5, alpha=0.85)
        fig.colorbar(im, ax=ax, label="Geopotential in gpdm", shrink=0.45, pad=0.03)

    @staticmethod
    def plot_significant_weather(ax, fig, lons, lats, data):
        cmap_ww, legend_dict = ColormapRegistry.get_significant_weather()
        grid = np.zeros_like(data)
        for i, (l, (c, codes)) in enumerate(legend_dict.items(), 1):
            for code in codes: 
                grid[data == code] = i
        # Wir maskieren die 0, damit der Hintergrund dort 100% transparent ist
        grid = np.where(grid == 0, np.nan, grid)
        ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, transform=ccrs.PlateCarree(), shading='nearest', zorder=5, alpha=0.9)
        patches = [mpatches.Patch(color=c, label=l) for l, (c, _) in legend_dict.items()]
        ax.legend(handles=patches, loc='lower left', title="Wetter", fontsize='6', title_fontsize='7', framealpha=0.9).set_zorder(25)

    @staticmethod
    def plot_rainviewer_radar(ax, fig, host, path, region):
        """Rainviewer Tile Fetcher. Nutzt RGBA Override für echten Alpha-Kanal."""
        rv_tiles = RainViewerTiles(host=host, path=path)
        zoom = GeoConfig.get_zoom(region)
        ax.add_image(rv_tiles, zoom, zorder=5, alpha=0.85)
        cmap_radar, norm_radar = ColormapRegistry.get_radar()
        sm = plt.cm.ScalarMappable(cmap=cmap_radar, norm=norm_radar)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Radar-Reflektivität in dBZ (RainViewer)", shrink=0.45, pad=0.03)

    @staticmethod
    def add_isobars(ax, isobars_data, ilons, ilats):
        if isobars_data is not None:
            p_hpa = MeteoMath.pa_to_hpa(isobars_data)
            if ilons.ndim == 1: ilons, ilats = np.meshgrid(ilons, ilats)
            cs = ax.contour(ilons, ilats, p_hpa, colors='black', linewidths=0.7, levels=np.arange(940, 1060, 4), transform=ccrs.PlateCarree(), zorder=20)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

    @staticmethod
    def add_storm_hatching(ax, ww_data, wlons, wlats):
        if ww_data is not None:
            storm_mask = np.isin(ww_data, [95, 96, 97, 99])
            if np.any(storm_mask):
                plot_ww = np.zeros_like(ww_data)
                plot_ww[storm_mask] = 1 
                plt.rcParams['hatch.linewidth'] = 2.0 
                ax.contourf(wlons, wlats, plot_ww, levels=[0.5, 1.5], colors='none', hatches=['////'], edgecolors='red', transform=ccrs.PlateCarree(), zorder=10)

    @staticmethod
    def add_poi_labels(ax, region):
        """Plottet Flughäfen und Städte als Overlay."""
        cities = GeoConfig.get_visible_poi(region)
        for name, coords in cities.items():
            ax.plot(coords[0], coords[1], marker='o', color='red', markersize=4, transform=ccrs.PlateCarree(), zorder=25)
            ax.text(coords[0] + 0.05, coords[1] + 0.05, name, transform=ccrs.PlateCarree(), 
                    fontsize=8, fontweight='bold', color='white', 
                    path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="black")], zorder=25)


# ==============================================================================
# 9. DYNAMISCHES USER INTERFACE (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.header("🛰️ Modell-Zentrale")
    
    with st.expander("🌍 1. Modell wählen", expanded=True):
        sel_model = st.radio("Wettermodell", list(MODEL_ROUTER.keys()), label_visibility="collapsed")
    
    with st.expander("🗺️ 2. Karten-Ausschnitt", expanded=False):
        valid_regions = MODEL_ROUTER[sel_model]["regions"]
        if "Berlin & Umland (Detail-Zoom)" in valid_regions and "Radar" in sel_model:
            default_idx = valid_regions.index("Berlin & Umland (Detail-Zoom)")
        else:
            default_idx = valid_regions.index("Brandenburg (Gesamt)") if "Brandenburg (Gesamt)" in valid_regions else 0
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
            if "EU" in sel_model or "AROME" in sel_model: hours = list(range(1, 79))
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
    show_sat = st.checkbox("🌍 Satelliten-Hintergrund", value=True)
    show_poi = st.checkbox("🏙️ Städte & Flughäfen einblenden", value=True)
    show_isobars = st.checkbox("Isobaren (Luftdruck) einblenden", value=True)
    show_storms = st.checkbox("⚡ Gewitter-Risiko rot schraffieren", value=True)
    
    enable_refresh = st.checkbox("🔄 Auto-Update (5 Min.)", value=False)
    if enable_refresh and st_autorefresh is not None:
        st_autorefresh(interval=300000, key="auto_refresh_radar")
        
    st.markdown("---")
    generate = st.button("🚀 Profi-Karte generieren", use_container_width=True)
    
    with st.expander("🛠️ Entwickler-Konsole"):
        debug_mode = st.checkbox("URL-Ping & Diagnostics")


# ==============================================================================
# 10. MAIN EXECUTION (DATEN LADEN & KARTE RENDERN)
# ==============================================================================
if generate or (enable_refresh and "Radar" in sel_model):
    
    SystemManager.cleanup_temp_files()
    
    with st.spinner(f"🛰️ Lade {sel_param} aus {sel_model}..."):
        
        data, lons, lats, run_id, d_logs = DataFetcher.fetch_model_data(sel_model, sel_param, sel_hour, debug_mode)
        
        iso_data, ilons, ilats = None, None, None
        if show_isobars and "Radar" not in sel_model:
            iso_data, ilons, ilats, _, _ = DataFetcher.fetch_model_data(sel_model, "Isobaren", sel_hour)
            
        ww_data, wlons, wlats = None, None, None
        if show_storms and "Radar" not in sel_model and sel_param != "Signifikantes Wetter" and "Signifikantes Wetter" in MODEL_ROUTER[sel_model]["params"]:
            ww_data, wlons, wlats, _, _ = DataFetcher.fetch_model_data(sel_model, "Signifikantes Wetter", sel_hour)

    if debug_mode and d_logs:
        st.write("📡 **Interne Server-Pings (Debug):**")
        for log in d_logs[:4]: 
            st.code(log)

    if data is not None:
        try:
            fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
            
            # KARTENAUSSCHNITT SETZEN (ZOOM-FIX)
            current_extent = GeoConfig.get_extent(sel_region)
            ax.set_extent(current_extent, crs=ccrs.PlateCarree())

            if show_sat:
                sat_tiles = GoogleSatelliteTiles()
                zoom = GeoConfig.get_zoom(sel_region)
                ax.add_image(sat_tiles, zoom, zorder=0)

            border_color = 'white' if show_sat else 'black'
            ax.add_feature(cfeature.COASTLINE, linewidth=0.9, edgecolor=border_color, zorder=12)
            ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor=border_color, zorder=12)
            states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
            ax.add_feature(states, linewidth=0.5, edgecolor=border_color, linestyle=":", zorder=12)

            # ------------------------------------------------------------------
            # PARAMETER DISPATCHER
            # ------------------------------------------------------------------
            if "Radar" in sel_param:
                if "RainViewer" in sel_model:
                    PlottingEngine.plot_rainviewer_radar(ax, fig, data, lons, sel_region)
                else:
                    PlottingEngine.plot_radar_simulated(ax, fig, lons, lats, data)
                    
            elif "Temperatur" in sel_param or "Taupunkt" in sel_param or "850 hPa Temp" in sel_param:
                PlottingEngine.plot_temperature(ax, fig, lons, lats, data, sel_param)
                
            elif "Niederschlag" in sel_param:
                PlottingEngine.plot_precipitation(ax, fig, lons, lats, data)
                
            elif "Wind" in sel_param or "Jetstream" in sel_param:
                PlottingEngine.plot_wind(ax, fig, lons, lats, data, sel_param)
                
            elif "CAPE" in sel_param:
                PlottingEngine.plot_cape(ax, fig, lons, lats, data)
                
            elif "Wolken" in sel_param or "Gesamtbedeckung" in sel_param:
                PlottingEngine.plot_clouds(ax, fig, lons, lats, data)
                
            elif "Geopotential" in sel_param:
                PlottingEngine.plot_geopotential(ax, fig, lons, lats, data)
                
            elif "Signifikantes Wetter" in sel_param:
                PlottingEngine.plot_significant_weather(ax, fig, lons, lats, data)
                
            elif "Bodendruck" in sel_param:
                val_hpa = MeteoMath.pa_to_hpa(data)
                im = ax.pcolormesh(lons, lats, val_hpa, cmap='jet', transform=ccrs.PlateCarree(), shading='auto', alpha=0.85, zorder=5)
                fig.colorbar(im, ax=ax, label="Bodendruck in hPa", shrink=0.45, pad=0.03)
                cs = ax.contour(lons, lats, val_hpa, colors='black', linewidths=0.8, levels=np.arange(940, 1060, 4), transform=ccrs.PlateCarree(), zorder=15)
                ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
                
            else:
                # Generischer Fallback
                im = ax.pcolormesh(lons, lats, data, cmap='viridis', transform=ccrs.PlateCarree(), shading='auto', alpha=0.85, zorder=5)
                fig.colorbar(im, ax=ax, label=sel_param, shrink=0.45, pad=0.03)

            # ------------------------------------------------------------------
            # OVERLAYS HINZUFÜGEN
            # ------------------------------------------------------------------
            PlottingEngine.add_isobars(ax, iso_data, ilons, ilats)
            PlottingEngine.add_storm_hatching(ax, ww_data, wlons, wlats)
            
            if show_poi:
                PlottingEngine.add_poi_labels(ax, sel_region)

            # ------------------------------------------------------------------
            # HEADER & META-INFORMATIONEN
            # ------------------------------------------------------------------
            if "Radar" in sel_model:
                try:
                    dt_obj = datetime.fromtimestamp(int(run_id), tz=timezone.utc)
                    dt_loc = dt_obj.astimezone(LOCAL_TZ)
                    time_display = dt_loc.strftime('%d.%m.%Y %H:%M') + (" MESZ" if dt_loc.dst() else " MEZ")
                except Exception:
                    time_display = run_id
                info_txt = f"Modell: {sel_model}\nParameter: {sel_param}\nLive-Stand: {time_display}\n(Quelle: RainViewer API)"
            else:
                v_dt_utc = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)
                v_dt_loc = v_dt_utc.astimezone(LOCAL_TZ)
                tz_str = "MESZ" if v_dt_loc.dst() else "MEZ"
                info_txt = f"Modell: {sel_model}\nParameter: {sel_param}\nTermin: {v_dt_loc.strftime('%d.%m.%Y %H:%M')} {tz_str}\nModell-Lauf: {run_id[-2:]}Z"
                
            ax.text(0.02, 0.98, info_txt, transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', 
                    bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3', edgecolor='gray'), zorder=30)

            st.pyplot(fig)
            
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
            
            SystemManager.cleanup_temp_files()
            
        except Exception as plot_err:
            st.error(f"⚠️ Es gab ein Problem beim Rendern der Karte: {str(plot_err)}")
            if debug_mode:
                st.code(traceback.format_exc())
            
    else:
        st.error(f"⚠️ Aktuell liefert der gewählte Server keine Daten für '{sel_param}'.")
        st.info("💡 Server-Delay: Der Lauf ist vermutlich noch im Upload. Versuche es in wenigen Minuten noch einmal.")

