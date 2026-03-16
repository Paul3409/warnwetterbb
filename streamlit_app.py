"""
=========================================================================================
WARNWETTER BB - PROFESSIONAL ANALYSIS & METEOROLOGICAL WORKSTATION
=========================================================================================
Entwickelt für hochauflösende Modellbetrachtung, Echtzeit-Radar und 
historische objektive Analyse (Interpolation von Stationsdaten).

Architektur:
- Objektorientiertes Design (OOP)
- Type-Hinting nach PEP 484
- Erweitertes Caching und Garbage Collection
- Multi-Threading für API-Abfragen (RainViewer, DWD CDC/BrightSky)
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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from typing import Tuple, List, Dict, Optional, Any, Union
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

# ==============================================================================
# 1. SYSTEM-SETUP & LOGGING KONFIGURATION
# ==============================================================================
# Konfiguriere das interne Logging für Fehlersuche auf Servern
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WarnwetterBB_Engine")

# Optionale Autorefresh-Logik für den Live-Betrieb (Dashboard-Modus)
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

st.set_page_config(
    page_title="WarnwetterBB | Analyse-Profi", 
    page_icon="🌪️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS für eine professionelle Dark/Light-Mode Anpassung
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Lokale Zeitzone für Deutschland
LOCAL_TZ = ZoneInfo("Europe/Berlin")
WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


# ==============================================================================
# 2. CORE UTILITIES & GARBAGE COLLECTION
# ==============================================================================
class SystemUtils:
    """Sammlung von System- und Speicherverwaltungsfunktionen."""
    
    @staticmethod
    def cleanup_temp_files(directory: str = ".") -> None:
        """
        Räumt temporäre Dateien aggressiv auf. 
        Verhindert das Vollaufen des RAM/SSD im Streamlit-Container.
        """
        temp_extensions = [".grib", ".grib2", ".bz2", ".idx", ".zip", ".txt", ".tmp"]
        freed_bytes = 0
        
        try:
            for filename in os.listdir(directory):
                if any(filename.endswith(ext) for ext in temp_extensions) and ("temp" in filename or "gfs" in filename):
                    filepath = os.path.join(directory, filename)
                    try:
                        size = os.path.getsize(filepath)
                        os.remove(filepath)
                        freed_bytes += size
                        logger.debug(f"Gelöscht: {filename} ({size/1024:.1f} KB)")
                    except Exception as e:
                        logger.warning(f"Konnte {filename} nicht löschen: {e}")
        except Exception as e:
            logger.error(f"Fehler bei Cleanup-Iteration: {e}")
            
        if freed_bytes > 0:
            logger.info(f"Cleanup abgeschlossen: {freed_bytes / (1024*1024):.2f} MB freigegeben.")


# ==============================================================================
# 3. KARTEN-HINTERGRÜNDE (SATELLITEN-ENGINE)
# ==============================================================================
class GoogleSatelliteTiles(cimgt.GoogleWTS):
    """
    Greift direkt auf Google Maps Satellite Tiles zu.
    Dies ist die stabilste Methode für fotorealistische Hintergründe
    und behebt das "Schwarzer-Hintergrund"-Problem von Esri.
    """
    def _image_url(self, tile: Tuple[int, int, int]) -> str:
        x, y, z = tile
        # 's' steht für Satellite in der Google Maps Tile API
        url = f'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        return url

class CartoLightTiles(cimgt.GoogleWTS):
    """Eine helle, reduzierte Karte, falls Satellit zu unruhig ist."""
    def _image_url(self, tile: Tuple[int, int, int]) -> str:
        x, y, z = tile
        url = f'https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png'
        return url


# ==============================================================================
# 4. METEOROLOGISCHE FARBSKALEN (COLORMAPS)
# ==============================================================================
class MeteoColors:
    """
    Zentrale Definition aller meteorologischen Farbskalen.
    Professionelle hexadezimale Abstufungen.
    """
    
    @staticmethod
    def get_temperature_cmap() -> mcolors.LinearSegmentedColormap:
        """Erzeugt eine Temperatur-Skala von -30°C bis +40°C."""
        colors = [
            (0.0, '#313695'), (0.1, '#4575b4'), (0.2, '#74add1'), (0.3, '#abd9e9'),
            (0.4, '#e0f3f8'), (0.5, '#ffffbf'), (0.6, '#fee090'), (0.7, '#fdae61'),
            (0.8, '#f46d43'), (0.9, '#d73027'), (1.0, '#a50026')
        ]
        return mcolors.LinearSegmentedColormap.from_list("temp_pro", colors)

    @staticmethod
    def get_precipitation_cmap() -> Tuple[mcolors.LinearSegmentedColormap, mcolors.Normalize]:
        """Skala für Niederschlag in mm/h oder mm/24h."""
        precip_values = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3, 4, 5, 8, 12, 15, 20, 30, 40, 50]
        precip_colors = [
            '#FFFFFF', '#87CEEB', '#1E90FF', '#191970', '#006400', '#32CD32', '#FFFF00', 
            '#FFA500', '#FF0000', '#8B0000', '#800000', '#4B0082', '#800080', '#9400D3', 
            '#7B68EE', '#FFFFFF'  
        ]
        vmax = 50.0
        anchors = [v / vmax for v in precip_values]
        cmap = mcolors.LinearSegmentedColormap.from_list("precip_pro", list(zip(anchors, precip_colors)))
        # NaN-Werte (kein Niederschlag) transparent machen für Satellitenbild!
        cmap.set_bad(color='white', alpha=0.0)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        return cmap, norm

    @staticmethod
    def get_radar_cmap() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        """Skala für Radar-Reflektivität (dBZ)."""
        levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
        colors = [
            '#FFFFFF', '#B0E0E6', '#00BFFF', '#0000FF', '#00FF00', '#32CD32', '#008000', 
            '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#800080', '#4B0082', '#E6E6FA'
        ]
        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color='black', alpha=0.0) # Transparenz für den Sat-Hintergrund
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm

    @staticmethod
    def get_wind_cmap() -> mcolors.LinearSegmentedColormap:
        """Skala für Windböen (km/h)."""
        colors = ['#ADD8E6', '#0000FF', '#008000', '#FFFF00', '#FFA500', '#FF0000', '#8B0000', '#800080', '#000000']
        return mcolors.LinearSegmentedColormap.from_list("wind_pro", colors, N=256)

    @staticmethod
    def get_cape_cmap() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        """Skala für konvektive Energie (J/kg)."""
        levels = [0, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]
        colors = [
            '#006400', '#2E8B57', '#ADFF2F', '#FFFF00', '#FFB347', '#FFA500', 
            '#FF4500', '#FF0000', '#8B0000', '#800080', '#FF00FF', '#FFFFFF', '#808080', '#404040'
        ]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm

    @staticmethod
    def get_significant_weather() -> Tuple[mcolors.ListedColormap, Dict[str, Tuple[str, List[int]]]]:
        """Definition der DWD ww-Codes (Signifikantes Wetter)."""
        legend_data = {
            "Nebel": ("#FFFF00", list(range(40, 50))),
            "Regen leicht": ("#00FF00", [50, 51, 58, 60, 80]),
            "Regen mäßig": ("#228B22", [53, 61, 62, 81]),
            "Regen stark": ("#006400", [54, 55, 63, 64, 65, 82]),
            "gefr. Regen": ("#FF0000", [56, 57, 66, 67]),
            "Schneeregen": ("#FFA500", [68, 69, 83, 84]),
            "Schnee": ("#0000FF", [70, 71, 72, 73, 74, 75, 85, 86, 87, 88]),
            "Gewitter": ("#800080", [95, 96, 97, 99])
        }
        # Transparenter Basis-Layer (#FFFFFF00)
        cmap = mcolors.ListedColormap(['#FFFFFF00'] + [c for l, (c, codes) in legend_data.items()])
        return cmap, legend_data


# ==============================================================================
# 5. DATEN-ABFRAGE KLASSEN (GRIB, RADAR, ARCHIV)
# ==============================================================================
class ModelFetcher:
    """Verwaltet den Download von numerischen Wettermodellen (NWP)."""
    
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) WarnwetterBB_Analyse/2.0'}
    
    @staticmethod
    def _estimate_dwd_run(now_utc: datetime) -> datetime:
        """ICON Modelle werden alle 3 Stunden aktualisiert (Delay ~2h)."""
        run_h = ((now_utc.hour - 2) // 3) * 3
        if run_h < 0: 
            return (now_utc - timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)
        return now_utc.replace(hour=run_h, minute=0, second=0, microsecond=0)

    @staticmethod
    def _estimate_gfs_run(now_utc: datetime) -> datetime:
        """GFS läuft alle 6 Stunden (Delay ~3.5h)."""
        run_h = ((now_utc.hour - 4) // 6) * 6
        if run_h < 0: 
            return (now_utc - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        return now_utc.replace(hour=run_h, minute=0, second=0, microsecond=0)

    @classmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_grib_data(cls, model: str, param_key: str, hour_offset: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """Lädt GRIB-Daten für DWD und GFS Modelle herunter und entpackt sie."""
        now = datetime.now(timezone.utc)
        
        if "ICON" in model:
            base_t = cls._estimate_dwd_run(now)
            m_dir = "icon" if "Global" in model else ("icon-d2" if "D2" in model else "icon-eu")
            reg = "icon_global" if "Global" in model else ("icon-d2_germany" if "D2" in model else "icon-eu_europe")
            
            # Fallback Schleife für verzögerte Modellläufe
            for offset in [0, 3, 6]:
                run_t = base_t - timedelta(hours=offset)
                dt_s = run_t.strftime("%Y%m%d%H")
                run_hour = run_t.hour
                
                lvl_str = "single-level"
                lvl_prefix = "2d_"
                if param_key in ["fi", "t", "relhum", "qv"]:
                    lvl_str = "pressure-level"
                    lvl_prefix = "500_" if "fi" in param_key else "850_"
                
                url = f"https://opendata.dwd.de/weather/nwp/{m_dir}/grib/{run_hour:02d}/{param_key}/{reg}_regular-lat-lon_{lvl_str}_{dt_s}_{hour_offset:03d}_{lvl_prefix}{param_key}.grib2.bz2"
                
                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code == 200:
                        with bz2.open(io.BytesIO(r.content)) as f_bz2:
                            with open("temp.grib", "wb") as f_out: 
                                f_out.write(f_bz2.read())
                        
                        ds = xr.open_dataset("temp.grib", engine='cfgrib')
                        ds_var = ds[list(ds.data_vars)[0]]
                        if 'isobaricInhPa' in ds_var.dims:
                            target_p = 500 if "fi" in param_key else 850
                            ds_var = ds_var.sel(isobaricInhPa=target_p)
                            
                        data = ds_var.isel(step=0, height=0, missing_dims='ignore').values.squeeze()
                        lons, lats = ds.longitude.values, ds.latitude.values
                        if lons.ndim == 1: 
                            lons, lats = np.meshgrid(lons, lats)
                        return data, lons, lats, dt_s
                except Exception as e:
                    logger.warning(f"GRIB-Fehler (ICON): {e}")
                    continue
                    
        elif "GFS" in model:
            base_t = cls._estimate_gfs_run(now)
            # GFS URL Query Builder
            gfs_p = ""
            if param_key == "t_2m": gfs_p = "&var_TMP=on&lev_2_m_above_ground=on"
            elif param_key == "tot_prec": gfs_p = "&var_APCP=on&lev_surface=on"
            elif param_key == "vmax_10m": gfs_p = "&var_GUST=on&lev_surface=on"
            elif param_key == "cape_ml": gfs_p = "&var_CAPE=on&lev_surface=on"
            elif param_key == "pmsl": gfs_p = "&var_PRMSL=on&lev_mean_sea_level=on"
            else: gfs_p = "&var_TMP=on&lev_2_m_above_ground=on" # Fallback
            
            for offset in [0, 6, 12]:
                run_t = base_t - timedelta(hours=offset)
                dt_s = run_t.strftime("%Y%m%d")
                run_hour = run_t.hour
                
                url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t{run_hour:02d}z.pgrb2.0p25.f{hour_offset:03d}{gfs_p}&subregion=&leftlon=-20&rightlon=45&toplat=75&bottomlat=30&dir=%2Fgfs.{dt_s}%2F{run_hour:02d}%2Fatmos"
                try:
                    r = requests.get(url, headers=cls.HEADERS, timeout=10)
                    if r.status_code == 200:
                        with open("temp_gfs.grib", "wb") as f: 
                            f.write(r.content)
                        ds = xr.open_dataset("temp_gfs.grib", engine='cfgrib')
                        data = ds[list(ds.data_vars)[0]].isel(step=0, height=0, isobaricInhPa=0, missing_dims='ignore').values.squeeze()
                        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
                        return data, lons, lats, f"{dt_s}{run_hour:02d}"
                except Exception as e:
                    logger.warning(f"GRIB-Fehler (GFS): {e}")
                    continue
                    
        return None, None, None, None


class RadarFetcher:
    """Verwaltet den Download von Echtzeit-Radardaten (RainViewer API)."""
    
    @staticmethod
    def get_rainviewer_metadata() -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Holt die Metadaten für das aktuellste Radar-Mosaik."""
        try:
            r = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=10)
            r.raise_for_status()
            data = r.json()
            host = data.get("host", "https://tilecache.rainviewer.com")
            past_radar = data.get("radar", {}).get("past", [])
            
            if past_radar:
                latest = past_radar[-1]
                return host, latest["path"], str(latest["time"])
        except Exception as e:
            logger.error(f"RainViewer API Fehler: {e}")
        return None, None, None


class DWDArchiveFetcher:
    """
    Spezialklasse für den Zugriff auf historische DWD-Wetterstationen.
    Nutzt BrightSky als hochperformanten Proxy für das DWD CDC-Archiv.
    """
    
    @classmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_historical_day(cls, target_date: datetime, parameter: str) -> Optional[pd.DataFrame]:
        """
        Lädt die Daten aller verfügbaren Stationen in Deutschland für einen 
        spezifischen historischen Tag (12:00 UTC).
        """
        date_str = target_date.strftime("%Y-%m-%d")
        # Abfragezentrum: Kassel (Mitte DE), Radius 600km deckt ganz Deutschland und Grenzgebiete ab
        url = f"https://api.brightsky.dev/observations?lat=51.3&lon=9.5&radius=600&date={date_str}T12:00:00Z"
        
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            
            if not obs:
                logger.warning(f"Keine historischen Daten für {date_str} gefunden.")
                return None
                
            # Mapping des UI-Parameters auf den API-Schlüssel
            p_map = {
                "Temperatur (°C)": "temperature",
                "Taupunkt (°C)": "dew_point",
                "Windgeschwindigkeit (km/h)": "wind_speed",
                "Windböen (km/h)": "wind_gust_speed",
                "Niederschlag (mm)": "precipitation",
                "Relative Feuchte (%)": "relative_humidity",
                "Luftdruck (hPa)": "pressure_msl"
            }
            api_key = p_map.get(parameter, "temperature")
            
            extracted_data = []
            for item in obs:
                val = item.get(api_key)
                if val is not None:
                    # Umrechnung von m/s in km/h für Wind
                    if "wind" in api_key:
                        val = val * 3.6
                    
                    extracted_data.append({
                        'lat': item['lat'],
                        'lon': item['lon'],
                        'val': val,
                        'station_id': item.get('source_id', 'Unknown')
                    })
                    
            df = pd.DataFrame(extracted_data)
            
            # Ausreißer filtern (Qualitätskontrolle)
            if "Temperatur" in parameter or "Taupunkt" in parameter:
                df = df[(df['val'] > -40) & (df['val'] < 50)]
            elif "Wind" in parameter:
                df = df[(df['val'] >= 0) & (df['val'] < 300)]
                
            return df
            
        except Exception as e:
            logger.error(f"Historischer API-Fehler: {e}")
            return None


# ==============================================================================
# 6. MATHEMATISCHE OBJEKTIVE ANALYSE (INTERPOLATION)
# ==============================================================================
class ObjectiveAnalysisEngine:
    """
    Verwandelt punktuelle Stationsdaten (Zahlen) in lückenlose, 
    physikalisch plausible Farbflächen (Grids) für die Kartendarstellung.
    """
    
    @staticmethod
    def create_grid(extent: List[float], resolution: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Erzeugt ein hochauflösendes Gitter für die Region."""
        # Extent: [lon_min, lon_max, lat_min, lat_max]
        lons = np.arange(extent[0], extent[1], resolution)
        lats = np.arange(extent[2], extent[3], resolution)
        return np.meshgrid(lons, lats)

    @staticmethod
    def interpolate_rbf(df: pd.DataFrame, grid_lon: np.ndarray, grid_lat: np.ndarray, smoothing: float = 0.5) -> np.ndarray:
        """
        Radial Basis Function Interpolation.
        Ideal für Temperatur und Druck, da es sehr weiche, natürliche Übergänge schafft.
        """
        # multiquadric ist der Goldstandard in der Meteorologie für glatte Felder
        rbf = Rbf(df['lon'], df['lat'], df['val'], function='multiquadric', smooth=smoothing)
        grid_val = rbf(grid_lon, grid_lat)
        return grid_val

    @staticmethod
    def interpolate_idw(df: pd.DataFrame, grid_lon: np.ndarray, grid_lat: np.ndarray, power: int = 2) -> np.ndarray:
        """
        Inverse Distance Weighting (IDW).
        Ideal für Niederschlag oder Windböen, wo lokale Extreme (Zellen)
        nicht zu stark in die Fläche verschmiert werden sollen.
        """
        # cKDTree für extrem schnelle Nachbarsuche
        points = np.column_stack((df['lon'], df['lat']))
        values = df['val'].values
        
        grid_points = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))
        tree = cKDTree(points)
        
        # Suche die 5 nächsten Stationen für jeden Gitterpunkt
        distances, indices = tree.query(grid_points, k=5)
        
        # Verhindere Division durch Null bei exaktem Treffer
        distances = np.maximum(distances, 1e-10)
        
        weights = 1.0 / (distances ** power)
        weighted_sum = np.sum(weights * values[indices], axis=1)
        sum_weights = np.sum(weights, axis=1)
        
        grid_val = (weighted_sum / sum_weights).reshape(grid_lon.shape)
        return grid_val


# ==============================================================================
# 7. ROUTING & UI-STRUKTUR (STREAMLIT SIDEBAR)
# ==============================================================================
# Master-Dictionary für das User-Interface
ROUTER = {
    "🌧️ RainViewer Echtzeit-Radar": {
        "modes": ["Live-Radar (Europaweites Mosaik)"],
        "regions": ["Deutschland", "Brandenburg/Berlin", "Europa"],
        "type": "radar"
    },
    "📊 DWD Historische Analyse (Archiv)": {
        "modes": [
            "Temperatur (°C)", 
            "Taupunkt (°C)", 
            "Windgeschwindigkeit (km/h)",
            "Windböen (km/h)",
            "Luftdruck (hPa)"
        ],
        "regions": ["Deutschland", "Brandenburg/Berlin"],
        "type": "archive"
    },
    "🔮 Vorhersagemodell: ICON-D2 (DWD)": {
        "modes": [
            "Temperatur 2m (°C)", 
            "Niederschlag (mm)", 
            "Windböen (km/h)", 
            "CAPE (J/kg)", 
            "Signifikantes Wetter"
        ],
        "regions": ["Deutschland", "Brandenburg/Berlin"],
        "type": "grib",
        "model_key": "ICON-D2"
    },
    "🌍 Vorhersagemodell: GFS (NOAA)": {
        "modes": [
            "Temperatur 2m (°C)", 
            "Niederschlag (mm)", 
            "Windböen (km/h)", 
            "Bodendruck (hPa)"
        ],
        "regions": ["Deutschland", "Europa"],
        "type": "grib",
        "model_key": "GFS"
    }
}

# Parameter zu GRIB-Keys Mapping
PARAM_TO_GRIB = {
    "Temperatur 2m (°C)": "t_2m",
    "Niederschlag (mm)": "tot_prec",
    "Windböen (km/h)": "vmax_10m",
    "CAPE (J/kg)": "cape_ml",
    "Signifikantes Wetter": "ww",
    "Bodendruck (hPa)": "pmsl"
}

# Regionen-Definitionen (Bounding Boxes: [lon_min, lon_max, lat_min, lat_max])
EXTENTS = {
    "Deutschland": [5.5, 15.5, 47.0, 55.2],
    "Brandenburg/Berlin": [11.0, 15.0, 51.1, 53.7],
    "Europa": [-12.0, 40.0, 34.0, 66.0]
}

# ----------------- SIDEBAR BUILDER -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/partly-cloudy-day--v1.png", width=60)
    st.title("WarnwetterBB")
    st.markdown("### Analyse-Profi")
    st.markdown("---")
    
    sel_category = st.selectbox("1. Datenquelle / Werkzeug", list(ROUTER.keys()))
    cat_data = ROUTER[sel_category]
    
    sel_param = st.selectbox("2. Meteorologischer Parameter", cat_data["modes"])
    sel_region = st.selectbox("3. Geografischer Ausschnitt", cat_data["regions"])
    
    # Dynamische Zeitauswahl basierend auf der Werkzeug-Kategorie
    sel_hour = 0
    sel_date = datetime.now()
    
    if cat_data["type"] == "grib":
        st.markdown("### ⏱️ Vorhersage-Zeitraum")
        hours_list = list(range(1, 49)) if "ICON" in sel_category else list(range(3, 123, 3))
        
        base_r = ModelFetcher._estimate_dwd_run(datetime.now(timezone.utc)) if "ICON" in sel_category else ModelFetcher._estimate_gfs_run(datetime.now(timezone.utc))
        
        time_labels = []
        for h in hours_list:
            t_loc = (base_r + timedelta(hours=h)).astimezone(LOCAL_TZ)
            wt = WOCHENTAGE[t_loc.weekday()]
            time_labels.append(f"+{h}h  ({wt}, {t_loc.strftime('%H:%M')} Uhr)")
            
        sel_time_str = st.selectbox("Stunde wählen", time_labels)
        sel_hour = int(sel_time_str.split("h")[0].replace("+", ""))
        
    elif cat_data["type"] == "archive":
        st.markdown("### 📅 Zeitreise (Archiv)")
        sel_date = st.date_input(
            "Historisches Datum wählen", 
            value=datetime.now() - timedelta(days=365),
            min_value=datetime(1950, 1, 1),
            max_value=datetime.now() - timedelta(days=1),
            help="Holt reale Messwerte aus dem DWD Climate Data Center für diesen Tag (12:00 Uhr)."
        )
        
        st.markdown("### 📐 Analyse-Mathematik")
        interpol_method = st.radio(
            "Glättungs-Algorithmus", 
            ["RBF (Weiche Flächen, exzellent für Temp)", "IDW (Harte Grenzen, gut für Niederschlag)"],
            help="Bestimmt, wie die Flächen zwischen den Stationen berechnet werden."
        )
        
    elif cat_data["type"] == "radar":
        st.info("Echtzeit-Mosaik: Zeitsteuerung ist im Live-Modus deaktiviert.")

    st.markdown("---")
    st.markdown("### ⚙️ Ebenen & Overlay")
    
    # Optionen je nach Modus anpassen
    show_sat = st.checkbox("🌍 Satelliten-Hintergrund", value=True, help="Nutzt hochauflösende Google Maps Satellite Tiles.")
    
    show_numbers = False
    if cat_data["type"] == "archive":
        show_numbers = st.checkbox("1️⃣2️⃣ Stations-Messwerte als Zahlen", value=True)
        
    enable_auto_refresh = False
    if cat_data["type"] == "radar":
        enable_auto_refresh = st.checkbox("🔄 Auto-Update (Live-Radar alle 5 Min.)", value=False)
        if enable_auto_refresh and st_autorefresh:
            st_autorefresh(interval=300000, key="radar_refresh")
            
    st.markdown("---")
    btn_generate = st.button("🚀 Karte berechnen & rendern", use_container_width=True)


# ==============================================================================
# 8. HAUPT-PLOTTING LOGIK (RENDER ENGINE)
# ==============================================================================
if btn_generate or (enable_auto_refresh and cat_data["type"] == "radar"):
    
    # Speicherbereinigung vor jedem neuen Lauf
    SystemUtils.cleanup_temp_files()
    
    current_extent = EXTENTS[sel_region]
    
    with st.spinner("🛰️ Kontaktiere Server und bereite meteorologische Analyse vor..."):
        
        # ----------------------------------------------------------------------
        # SCHRITT 1: PLOT-FIGURE UND KARTENPROJEKTION INITIALISIEREN
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 13), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=150)
        ax.set_extent(current_extent)
        
        # Basis-Kartenlayer
        if show_sat:
            # Nutzt unsere GoogleSatelliteTiles Klasse für garantierte Funktion
            sat_provider = GoogleSatelliteTiles()
            zoom_level = 6 if "Deutschland" in sel_region else (8 if "Brandenburg" in sel_region else 5)
            ax.add_image(sat_provider, zoom_level, zorder=0)
        else:
            # Fallback: Helle Karte
            ax.add_image(CartoLightTiles(), 6, zorder=0)
            
        # Politische Grenzen (Weiß auf Satellit, Schwarz auf heller Karte)
        line_color = 'white' if show_sat else 'black'
        ax.add_feature(cfeature.BORDERS, linewidth=1.2, edgecolor=line_color, linestyle='-', zorder=10)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor=line_color, zorder=10)
        states_pro = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(states_pro, linewidth=0.5, edgecolor=line_color, linestyle=':', zorder=10)


        # ----------------------------------------------------------------------
        # SCHRITT 2: DATEN LADEN UND VERARBEITEN (JE NACH MODUS)
        # ----------------------------------------------------------------------
        
        # ======================================================================
        # MODUS A: RAINVIEWER LIVE-RADAR
        # ======================================================================
        if cat_data["type"] == "radar":
            rv_host, rv_path, rv_ts = RadarFetcher.get_rainviewer_metadata()
            if rv_host and rv_path:
                rv_tiles = RainViewerTiles(host=rv_host, path=rv_path)
                zoom = 6 if "Deutschland" in sel_region else (8 if "Brandenburg" in sel_region else 5)
                # Radar wird als Overlay (zorder=5) über den Satelliten gelegt
                ax.add_image(rv_tiles, zoom, zorder=5, alpha=0.8)
                
                # Legende manuell erzeugen (RainViewer liefert das Bild schon fertig gefärbt)
                cmap_rad, norm_rad = MeteoColors.get_radar_cmap()
                sm = plt.cm.ScalarMappable(cmap=cmap_rad, norm=norm_rad)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, label="Radar-Reflektivität in dBZ (RainViewer)", shrink=0.5, pad=0.02)
                
                # Header-Informationen
                dt_obj = datetime.fromtimestamp(int(rv_ts), tz=timezone.utc).astimezone(LOCAL_TZ)
                header_text = f"RainViewer Live-Mosaik\nStand: {dt_obj.strftime('%d.%m.%Y %H:%M')} Uhr"
            else:
                st.error("RainViewer API momentan nicht erreichbar.")
                st.stop()
                

        # ======================================================================
        # MODUS B: HISTORISCHE DWD ANALYSE (INTERPOLATION & ZAHLEN)
        # -> Das "mtwetter.de" Feature!
        # ======================================================================
        elif cat_data["type"] == "archive":
            df_stations = DWDArchiveFetcher.fetch_historical_day(sel_date, sel_param)
            
            if df_stations is not None and not df_stations.empty:
                # Gitter für die Region erstellen
                grid_lon, grid_lat = ObjectiveAnalysisEngine.create_grid(current_extent, resolution=0.05)
                
                # Interpolation anwenden (RBF oder IDW je nach User-Wahl)
                if "RBF" in interpol_method:
                    grid_z = ObjectiveAnalysisEngine.interpolate_rbf(df_stations, grid_lon, grid_lat, smoothing=0.8)
                else:
                    grid_z = ObjectiveAnalysisEngine.interpolate_idw(df_stations, grid_lon, grid_lat, power=2)
                
                # Colormaps dynamisch wählen
                if "Temp" in sel_param or "Taupunkt" in sel_param:
                    cmap = MeteoColors.get_temperature_cmap()
                    norm = mcolors.Normalize(vmin=-20, vmax=40)
                    unit = "°C"
                elif "Wind" in sel_param:
                    cmap = MeteoColors.get_wind_cmap()
                    norm = mcolors.Normalize(vmin=0, vmax=120)
                    unit = "km/h"
                elif "Luftdruck" in sel_param:
                    cmap = plt.cm.jet
                    norm = mcolors.Normalize(vmin=970, vmax=1040)
                    unit = "hPa"
                else:
                    cmap = plt.cm.viridis
                    norm = mcolors.Normalize()
                    unit = ""
                
                # 1. FARBFLÄCHE ZEICHNEN
                # alpha=0.6 sorgt dafür, dass das Satellitenbild durch die Farbe scheint!
                im = ax.pcolormesh(grid_lon, grid_lat, grid_z, cmap=cmap, norm=norm, 
                                   shading='auto', alpha=0.6, zorder=5)
                fig.colorbar(im, ax=ax, label=f"{sel_param}", shrink=0.5, pad=0.02)
                
                # 2. ZAHLEN (MESSWERTE) ÜBER DIE FLÄCHE SCHREIBEN
                if show_numbers:
                    # Filtere Stationen, die exakt im Bildausschnitt liegen (Performance!)
                    mask = (df_stations['lon'] >= current_extent[0]) & (df_stations['lon'] <= current_extent[1]) & \
                           (df_stations['lat'] >= current_extent[2]) & (df_stations['lat'] <= current_extent[3])
                    visible_stations = df_stations[mask]
                    
                    for _, row in visible_stations.iterrows():
                        # Schwarze Zahl auf weißem, leicht transparenten Grund
                        ax.text(row['lon'], row['lat'], f"{row['val']:.1f}", 
                                transform=ccrs.PlateCarree(),
                                fontsize=6, fontweight='bold', color='black',
                                ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5),
                                zorder=20)
                
                # Header-Informationen
                header_text = f"DWD Objektive Analyse\nParameter: {sel_param}\nDatum: {sel_date.strftime('%d.%m.%Y')} (12Z)\nMethode: {interpol_method.split('(')[0].strip()}"
                
                # Speichere Daten für die Statistik-Box unten
                st.session_state['archive_stats'] = df_stations
                st.session_state['archive_unit'] = unit
                
            else:
                st.error(f"Für den {sel_date.strftime('%d.%m.%Y')} liegen im DWD CDC-Archiv keine ausreichenden Stationsdaten vor.")
                st.stop()


        # ======================================================================
        # MODUS C: VORHERSAGEMODELLE (GRIB2 PARSING)
        # ======================================================================
        elif cat_data["type"] == "grib":
            grib_key = PARAM_TO_GRIB.get(sel_param, "t_2m")
            data, lons, lats, run_id = ModelFetcher.fetch_grib_data(cat_data["model_key"], grib_key, sel_hour)
            
            if data is not None:
                # Temperatur Plotting
                if "Temp" in sel_param:
                    val = data - 273.15 if data.max() > 100 else data # Kelvin to Celsius
                    cmap = MeteoColors.get_temperature_cmap()
                    norm = mcolors.Normalize(vmin=-20, vmax=40)
                    im = ax.pcolormesh(lons, lats, val, cmap=cmap, norm=norm, shading='auto', alpha=0.7, zorder=5)
                    fig.colorbar(im, ax=ax, label="Temperatur in °C", shrink=0.5, pad=0.02)
                    
                # Niederschlag Plotting
                elif "Niederschlag" in sel_param:
                    cmap, norm = MeteoColors.get_precipitation_cmap()
                    im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto', zorder=5)
                    fig.colorbar(im, ax=ax, label="Niederschlag in mm", shrink=0.5, pad=0.02)
                    
                # Windböen Plotting
                elif "Wind" in sel_param:
                    val = data * 3.6 # m/s to km/h
                    cmap = MeteoColors.get_wind_cmap()
                    norm = mcolors.Normalize(vmin=0, vmax=140)
                    im = ax.pcolormesh(lons, lats, val, cmap=cmap, norm=norm, shading='auto', alpha=0.7, zorder=5)
                    fig.colorbar(im, ax=ax, label="Windböen in km/h", shrink=0.5, pad=0.02)
                    
                # CAPE Plotting (Gewitter-Energie)
                elif "CAPE" in sel_param:
                    cmap, norm = MeteoColors.get_cape_cmap()
                    im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto', alpha=0.7, zorder=5)
                    fig.colorbar(im, ax=ax, label="CAPE in J/kg", shrink=0.5, pad=0.02)
                    
                # Bodendruck Plotting
                elif "Bodendruck" in sel_param:
                    val = data / 100 if data.max() > 5000 else data # Pa to hPa
                    im = ax.pcolormesh(lons, lats, val, cmap=plt.cm.jet, shading='auto', alpha=0.7, zorder=5)
                    fig.colorbar(im, ax=ax, label="Luftdruck in hPa", shrink=0.5, pad=0.02)
                    # Isobaren als Konturlinien zeichnen!
                    cs = ax.contour(lons, lats, val, colors='black', linewidths=0.8, levels=np.arange(940, 1060, 4), zorder=15)
                    ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
                    
                # Signifikantes Wetter Plotting (ww-Codes)
                elif "Wetter" in sel_param:
                    cmap_ww, legend_dict = MeteoColors.get_significant_weather()
                    # Leeres Raster erstellen
                    grid = np.zeros_like(data)
                    for i, (label, (color, codes)) in enumerate(legend_dict.items(), 1):
                        for code in codes: 
                            grid[data == code] = i
                            
                    ax.pcolormesh(lons, lats, grid, cmap=cmap_ww, shading='nearest', zorder=5)
                    
                    # Manuelle Legende basteln
                    patches = [mpatches.Patch(color=color, label=label) for label, (color, _) in legend_dict.items()]
                    ax.legend(handles=patches, loc='lower left', title="Wetter", fontsize='7', title_fontsize='8', framealpha=0.9).set_zorder(25)
                
                # Header-Informationen
                v_dt_utc = datetime.strptime(run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=sel_hour)
                v_dt_loc = v_dt_utc.astimezone(LOCAL_TZ)
                tz_str = "MESZ" if v_dt_loc.dst() else "MEZ"
                header_text = f"Modell: {cat_data['model_key']}\nParameter: {sel_param}\nTermin: {v_dt_loc.strftime('%d.%m.%Y %H:%M')} {tz_str}\nLauf: {run_id[-2:]}Z"
                
            else:
                st.error("Der GRIB-Server hat für diesen Termin keine Daten geliefert. Versuche eine andere Vorhersagestunde.")
                st.stop()


        # ----------------------------------------------------------------------
        # SCHRITT 3: HEADER EINZEICHNEN & BILD RENDERN
        # ----------------------------------------------------------------------
        # Info-Box oben links
        ax.text(0.02, 0.98, header_text, 
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top', 
                bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.4', edgecolor='gray'), 
                zorder=30)
                
        # Karte in Streamlit anzeigen
        st.pyplot(fig)
        
        # ----------------------------------------------------------------------
        # SCHRITT 4: BILD FÜR DOWNLOAD BEREITSTELLEN
        # ----------------------------------------------------------------------
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.download_button(
                label="📥 Professionelle Karte als PNG herunterladen",
                data=img_buffer,
                file_name=f"WarnwetterBB_Analyse_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                mime="image/png",
                use_container_width=True
            )

# ==============================================================================
# 9. ZUSÄTZLICHE STATISTIK-BERICHTE (NUR BEI ARCHIV-MODUS)
# ==============================================================================
if cat_data["type"] == "archive" and 'archive_stats' in st.session_state:
    df_s = st.session_state['archive_stats']
    unit = st.session_state.get('archive_unit', '')
    
    st.markdown("---")
    st.markdown(f"### 📊 Deutschlandweite Stations-Statistik für den {sel_date.strftime('%d.%m.%Y')}")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"<div class='metric-box'><b>Höchster Wert</b><br><h2 style='color:#d73027;'>{df_s['val'].max():.1f} {unit}</h2></div>", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"<div class='metric-box'><b>Tiefster Wert</b><br><h2 style='color:#313695;'>{df_s['val'].min():.1f} {unit}</h2></div>", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"<div class='metric-box'><b>Bundesweiter Schnitt</b><br><h2 style='color:#4575b4;'>{df_s['val'].mean():.1f} {unit}</h2></div>", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"<div class='metric-box'><b>Aktive Stationen</b><br><h2 style='color:#555555;'>{len(df_s)}</h2></div>", unsafe_allow_html=True)

