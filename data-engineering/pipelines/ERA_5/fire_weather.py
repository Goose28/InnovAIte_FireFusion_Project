"""
fire_weather.py
Derived fire-weather features for the FireFusion ERA5 pipeline.

Implements the McArthur Forest Fire Danger Index (Mark 5, Noble et al. 1980):
    FFDI = 2 * exp(-0.450 + 0.987*ln(DF) - 0.0345*RH + 0.0338*T + 0.0234*V)
where
    T  = daily max temperature (degC)
    RH = relative humidity at the time of T (%)
    V  = 10 m wind speed at the time of T (km/h)
    DF = drought factor (0-10), from the Keetch-Byram Drought Index (KBDI)
         plus recent rainfall.

Usage inside era5_weather_processor.py:
    from fire_weather import daily_fire_inputs, add_fire_weather
    daily = daily_fire_inputs(hourly_df)       # hourly ERA5 -> daily inputs
    daily = add_fire_weather(daily)            # adds kbdi, drought_factor, ffdi, ...
"""
import numpy as np
import pandas as pd

KBDI_MAX = 203.2          # mm, soil moisture deficit ceiling (8 inches)
KBDI_INTERCEPT = 5.08     # mm, first rain of each wet spell lost to canopy/litter
DRY_DAY_MM = 0.2          # ERA5 drizzles a lot; below this counts as a dry day
RAIN_EVENT_MM = 2.0       # daily rain needed to count as a "rain event" for DF

FFDI_BINS = [-np.inf, 12, 25, 50, 75, 100, np.inf]
FFDI_LABELS = ["Low-Moderate", "High", "Very High", "Severe", "Extreme", "Catastrophic"]


# ---------------------------------------------------------------- basic physics
def relative_humidity(t_c, td_c):
    """Relative humidity (%) from air temperature and dewpoint (degC), Magnus formula."""
    a, b = 17.625, 243.04
    rh = 100.0 * np.exp(a * td_c / (b + td_c)) / np.exp(a * t_c / (b + t_c))
    return np.clip(rh, 0.0, 100.0)


def wind_speed_kmh(u10, v10):
    """10 m wind speed (km/h) from ERA5 u/v components (m/s)."""
    return np.hypot(u10, v10) * 3.6


def vapour_pressure_deficit(t_c, td_c):
    """Vapour pressure deficit (kPa)."""
    es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    ea = 0.6108 * np.exp(17.27 * td_c / (td_c + 237.3))
    return np.maximum(es - ea, 0.0)


# ------------------------------------------------------- hourly -> daily inputs
def daily_fire_inputs(hourly: pd.DataFrame, cell_col="grid_id",
                      time_col="time", tz="Australia/Melbourne") -> pd.DataFrame:
    """
    Aggregate hourly ERA5 to the daily inputs FFDI needs.
    Expects columns: grid_id, time (UTC), t2m, d2m (K), u10, v10 (m/s), tp (m).
    Days are LOCAL days, and RH/wind are taken at the hour of max temperature,
    because FFDI describes afternoon peak conditions, not daily means.
    """
    h = hourly.copy()
    local = pd.to_datetime(h[time_col], utc=True).dt.tz_convert(tz)
    h["date"] = local.dt.tz_localize(None).dt.normalize()
    h["t_c"] = h["t2m"] - 273.15
    h["td_c"] = h["d2m"] - 273.15
    h["rh_pct"] = relative_humidity(h["t_c"], h["td_c"])
    h["wind_kmh"] = wind_speed_kmh(h["u10"], h["v10"])
    h["rain_mm"] = h["tp"] * 1000.0

    peak_idx = h.groupby([cell_col, "date"])["t_c"].idxmax()
    peak = (h.loc[peak_idx, [cell_col, "date", "t_c", "td_c", "rh_pct", "wind_kmh"]]
              .rename(columns={"t_c": "tmax_c", "td_c": "td_at_tmax_c"}))
    rain = h.groupby([cell_col, "date"], as_index=False)["rain_mm"].sum()
    return peak.merge(rain, on=[cell_col, "date"])


# ------------------------------------------------------------ stateful indices
def kbdi_series(tmax_c, rain_mm, mean_annual_rain_mm, q0=0.0):
    """Keetch-Byram Drought Index (mm, 0-203.2), metric form, for one grid cell."""
    q, wet_spell = q0, 0.0
    denom = 1.0 + 10.88 * np.exp(-0.001736 * mean_annual_rain_mm)
    out = np.empty(len(tmax_c))
    for i, (t, r) in enumerate(zip(tmax_c, rain_mm)):
        if r >= DRY_DAY_MM:
            prev = wet_spell
            wet_spell += r
            net = max(wet_spell - KBDI_INTERCEPT, 0.0) - max(prev - KBDI_INTERCEPT, 0.0)
            q = max(q - net, 0.0)
        else:
            wet_spell = 0.0
        dq = (KBDI_MAX - q) * (0.968 * np.exp(0.0875 * t + 1.5552) - 8.30) / denom * 1e-3
        q = min(q + max(dq, 0.0), KBDI_MAX)
        out[i] = q
    return out


def drought_factor_series(kbdi, rain_mm):
    """
    Drought factor (0-10), Noble et al. (1980):
        DF = 0.191*(I + 104)*(N + 1)^1.5 / (3.52*(N + 1)^1.5 + P - 1)
    I = KBDI (mm), N = days since last rain event, P = total rain of that event (mm).
    """
    n_days, event_total, in_event = 0, 0.0, False
    out = np.empty(len(kbdi))
    for i, (smd, r) in enumerate(zip(kbdi, rain_mm)):
        if r >= RAIN_EVENT_MM:
            event_total = event_total + r if in_event else r
            in_event, n_days = True, 0
        else:
            in_event = False
            n_days += 1
        x = (n_days + 1) ** 1.5
        out[i] = min(0.191 * (smd + 104.0) * x / (3.52 * x + event_total - 1.0), 10.0)
    return out


def ffdi(tmax_c, rh_pct, wind_kmh, drought_factor):
    """McArthur Mark 5 Forest Fire Danger Index."""
    df = np.maximum(drought_factor, 1e-6)
    return 2.0 * np.exp(-0.450 + 0.987 * np.log(df) - 0.0345 * rh_pct
                        + 0.0338 * tmax_c + 0.0234 * wind_kmh)


# ------------------------------------------------------------------ main entry
def add_fire_weather(daily: pd.DataFrame, cell_col="grid_id", date_col="date",
                     mean_annual_rain=None) -> pd.DataFrame:
    """
    Add kbdi, drought_factor, ffdi, ffdi_category (and vpd_kpa if dewpoint present).
    `daily` needs: grid_id, date, tmax_c, rh_pct, wind_kmh, rain_mm.
    Must be run over each cell's FULL history in date order, because KBDI carries
    state from day to day. Discard the first ~6-12 months as spin-up.
    `mean_annual_rain`: optional {grid_id: mm}; otherwise estimated from the data.
    """
    df = daily.sort_values([cell_col, date_col]).copy()
    parts = []
    for cell, g in df.groupby(cell_col, sort=False):
        g = g.copy()
        if mean_annual_rain is not None:
            mar = mean_annual_rain[cell]
        else:
            years = max((g[date_col].max() - g[date_col].min()).days / 365.25, 1.0)
            mar = g["rain_mm"].sum() / years
        g["kbdi"] = kbdi_series(g["tmax_c"].to_numpy(), g["rain_mm"].to_numpy(), mar)
        g["drought_factor"] = drought_factor_series(g["kbdi"].to_numpy(), g["rain_mm"].to_numpy())
        parts.append(g)
    df = pd.concat(parts)

    df["ffdi"] = ffdi(df["tmax_c"], df["rh_pct"], df["wind_kmh"], df["drought_factor"])
    df["ffdi_category"] = pd.cut(df["ffdi"], FFDI_BINS, labels=FFDI_LABELS, right=False)
    if "td_at_tmax_c" in df:
        df["vpd_kpa"] = vapour_pressure_deficit(df["tmax_c"], df["td_at_tmax_c"])
    return df
