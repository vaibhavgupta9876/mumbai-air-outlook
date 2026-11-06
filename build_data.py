"""Build data.json for the Mumbai Air Outlook page. Standalone: needs only the challenge files, two forecast CSVs and Open-Meteo.

    python build_data.py --data <dir with Train.csv, Test.csv, Stations.csv, AdditionalHistory/> \
                         --pair forecast_low.csv forecast_high.csv --weather api --out data.json

--weather api    pulls boundary-layer height and 10 m wind from Open-Meteo (forecast API, 31 past days + 16 ahead, plus the
                 historical-forecast API for the 15 Oct-15 Dec 2025 reference period used for the ventilation terciles).
--weather cache  reads the parquet pulls cached by the challenge repository instead (src.external); same columns.
Inputs: challenge data (CPCB/CAAQMS via OpenAQ, XKDR, Rohan Rao/CPCB-derived; CC-BY-SA 4.0 via Zindi) and Open-Meteo (CC-BY 4.0),
which is listed in the ClimateVerse challenge catalogue. No other source. The forecast band is our own hedged submission pair.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
MAX_VALID_PM25 = 1000.0
DIWALI = {2018: "2018-11-07", 2019: "2019-10-27", 2024: "2024-10-31", 2025: "2025-10-20", 2026: "2026-11-08"}
NAME_TO_LOC = {"Malad West, Mumbai - IITM": "11611", "Borivali East, Mumbai - IITM": "11606",
               "Byculla, Mumbai - BMC": "3409510", "Shivaji Nagar, Mumbai - BMC": "3409511"}
SHORT = {"11611": "Malad West", "11606": "Borivali East", "3409510": "Byculla", "3409511": "Shivaji Nagar"}
REL = np.arange(-72, 144)  # hours from Diwali 00:00: D-3 00:00 .. D+5 23:00
REF = ("2025-10-15", "2025-12-15")  # season reference for the ventilation terciles
API = {"forecast": "https://api.open-meteo.com/v1/forecast",
       "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast"}


# ----------------------------------------------------------------------------------------------------------------- loaders
def ist(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, format="ISO8601").dt.tz_convert(IST)


def load_train(data: Path) -> pd.DataFrame:
    """Hourly grid [hour_start x location_id] from Train.csv (timestamp_ist marks the hour END), sensor faults removed."""
    df = pd.read_csv(data / "Train.csv", dtype={"location_id": "string"})
    df["hour_start"] = ist(df["timestamp_ist"]) - pd.Timedelta(hours=1)
    df = df[df["pm25"].between(0, MAX_VALID_PM25)]
    wide = df.pivot_table(index="hour_start", columns="location_id", values="pm25", aggfunc="mean")
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="h", tz=IST)).astype(float)
    # Isolated single-station spikes (> 200, > 4x the station's own 7-h median, > 3x the other stations that hour) are
    # sensor faults; city-wide events such as Diwali pass because every station rises together.
    med = wide.rolling(7, center=True, min_periods=3).median()
    n_other = np.maximum(wide.notna().sum(axis=1).to_numpy()[:, None] - 1, 1)
    others = pd.DataFrame((wide.sum(axis=1).to_numpy()[:, None] - wide.fillna(0).to_numpy()) / n_other, index=wide.index, columns=wide.columns)
    wide = wide.mask((wide > 200) & (wide > 4 * med) & (wide > 3 * others))
    wide.columns = [str(c) for c in wide.columns]
    return wide


def load_test(data: Path) -> pd.DataFrame:
    t = pd.read_csv(data / "Test.csv", dtype={"ID": "string", "location_id": "string"}, usecols=["ID", "location_id", "target_start_ist"])
    t["hour_start"] = ist(t["target_start_ist"])
    return t


def load_other(data: Path) -> pd.DataFrame:
    o = pd.read_csv(data / "AdditionalHistory" / "Other_Mumbai_Stations_2025_2026.csv.gz", usecols=["timestamp_ist", "mean", "station_name"])
    o["hour_start"] = ist(o["timestamp_ist"]) - pd.Timedelta(hours=1)
    return o.rename(columns={"mean": "pm25"})


def history_2018_2020(data: Path) -> pd.DataFrame:
    h = pd.read_csv(data / "AdditionalHistory" / "History_Mumbai_2018_2020.csv.gz", usecols=["StationName", "Datetime", "PM2.5"])
    h = h[h["PM2.5"].between(0, MAX_VALID_PM25)]
    h["hour_start"] = pd.to_datetime(h["Datetime"]).dt.tz_localize(IST)
    return h.pivot_table(index="hour_start", columns="StationName", values="PM2.5", aggfunc="mean")


def history_2024(data: Path) -> pd.DataFrame:
    x = pd.read_csv(data / "AdditionalHistory" / "History_Mumbai_2024.csv.gz", usecols=["station_name", "collected_at", "value"])
    x = x[x["value"].between(0, MAX_VALID_PM25)].copy()
    x["hour_start"] = pd.to_datetime(x["collected_at"]).dt.tz_localize(IST) - pd.Timedelta(hours=1)  # stamps are hour-end
    return x.pivot_table(index="hour_start", columns="station_name", values="value", aggfunc="mean")


# ----------------------------------------------------------------------------------------------------------------- weather
def fetch_api(kind: str, lat: float, lon: float, **params) -> pd.DataFrame:
    import urllib.parse
    import urllib.request
    q = {"latitude": lat, "longitude": lon, "hourly": "boundary_layer_height,wind_speed_10m", "timezone": IST, **params}
    url = API[kind] + "?" + urllib.parse.urlencode(q)
    err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                h = json.load(r)["hourly"]
            break
        except Exception as e:  # noqa: BLE001 - retry on any transport error, then give up
            err = e
            time.sleep(2 + 2 * attempt)
    else:
        raise RuntimeError(f"Open-Meteo {kind} failed: {err}")
    df = pd.DataFrame({"boundary_layer_height": h["boundary_layer_height"], "wind_speed_10m": h["wind_speed_10m"]},
                      index=pd.DatetimeIndex(pd.to_datetime(h["time"])).tz_localize(IST, ambiguous="NaT", nonexistent="NaT"))
    df.index.name = "hour_start"
    return df


def weather(mode: str, lat: float, lon: float, cache: Path | None) -> tuple[pd.DataFrame, str]:
    """Hourly frame with boundary_layer_height and wind_speed_10m covering the reference period and 31 days back + 16 ahead."""
    if mode == "api":
        hist = fetch_api("historical_forecast", lat, lon, start_date=REF[0], end_date=REF[1])
        live = fetch_api("forecast", lat, lon, past_days=31, forecast_days=16)
        stamp = pd.Timestamp.now(IST).strftime("%Y%m%dT%H")
    else:
        files = sorted(cache.glob(f"forecast_{lat}_{lon}_*.parquet"))
        live = pd.read_parquet(files[-1]).drop(columns=["fetched_at"]).set_index("hour_start")
        stamp = files[-1].stem.split("_")[-1]
        hist = pd.concat([pd.read_parquet(f) for f in cache.glob(f"historical_forecast_{lat}_{lon}_*.parquet")])
        hist = hist.drop(columns=["fetched_at"]).drop_duplicates("hour_start").set_index("hour_start").sort_index()
        hist = hist.loc[hist.index < live.index.min()]
    w = pd.concat([hist[["boundary_layer_height", "wind_speed_10m"]], live[["boundary_layer_height", "wind_speed_10m"]]]).sort_index()
    return w[~w.index.duplicated()], stamp


# ----------------------------------------------------------------------------------------------------------------- pieces
def iso(idx) -> list[str]:
    return [t.strftime("%Y-%m-%dT%H:%M") for t in idx]


def nums(s) -> list:
    a = np.asarray(s, dtype=float)
    return [None if np.isnan(x) else round(float(x), 1) for x in a]


def window(d: str) -> pd.DatetimeIndex:
    d0 = pd.Timestamp(d, tz=IST)
    return pd.date_range(d0 - pd.Timedelta(hours=72), d0 + pd.Timedelta(hours=143), freq="h")


def diwali_year(year: int, wide: pd.DataFrame, source: str, targets: dict[str, str] | None = None) -> dict | None:
    """City line = median station per hour (mean below 3 stations). Stations need >= 50% coverage of the window and a window
    median above 10 ug/m3 (Sewri reported 1-8 in Oct 2025: a failing sensor, not clean air). Baseline = D-3..D-2; peak = the
    highest hour from D-1 00:00 to D+2 23:00; 'clear' = the first hour after the peak at which the centred 6-h mean is back
    within 25% of the baseline; 'excess' = D0..D+2 mean over the baseline."""
    w = wide.reindex(window(DIWALI[year]))
    ok = w.columns[(w.notna().mean() >= 0.5) & (w.median() > 10)]
    if len(ok) == 0:
        return None
    city = w[ok].median(axis=1) if len(ok) >= 3 else w[ok].mean(axis=1)
    base = float(city.iloc[:48].mean())
    fest = city.iloc[48:144]
    peak_i = None if fest.isna().all() else int(48 + np.nanargmax(fest.to_numpy()))
    clear = None
    if peak_i is not None:
        smooth = city.rolling(6, center=True, min_periods=3).mean()
        hit = smooth.iloc[peak_i + 1:]
        hit = hit[hit <= base * 1.25]
        if len(hit):
            clear = int(REL[city.index.get_loc(hit.index[0])])
    out = {"year": year, "date": DIWALI[year], "source": source, "n_stations": int(len(ok)), "rel": REL.tolist(), "city": nums(city),
           "baseline": round(base, 1), "peak_rel": None if peak_i is None else int(REL[peak_i]),
           "peak_value": None if peak_i is None else round(float(city.iloc[peak_i]), 1), "clear_rel": clear,
           "excess": round(float(city.iloc[72:144].mean() / base), 2) if base > 0 else None, "targets": {}}
    for name, loc in (targets or {}).items():
        if name in ok:
            out["targets"][loc] = nums(w[name])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with Train.csv, Test.csv, Stations.csv, AdditionalHistory/")
    ap.add_argument("--pair", nargs=2, metavar=("LOW", "HIGH"), type=Path, required=True, help="the two forecast CSVs (ID, Target)")
    ap.add_argument("--weather", choices=["api", "cache"], default="api")
    ap.add_argument("--cache", type=Path, help="with --weather cache: folder of src.external parquet pulls")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data.json")
    a = ap.parse_args()
    st = pd.read_csv(a.data / "Stations.csv", dtype={"location_id": "string"})
    grid = load_train(a.data)
    t_end = grid.index.max()
    today = pd.Timestamp.now(IST).normalize()
    start, end = today, today + pd.Timedelta(days=6, hours=23)
    print(f"Train ends {t_end:%d %b %Y %H:%M}; outlook window {start:%d %b} .. {end:%d %b}")

    # Outlook from the pair files: low leg, high leg, central = low / 0.95 (the unscaled forecast)
    test = load_test(a.data)
    low = pd.read_csv(a.pair[0], dtype={"ID": "string"}).set_index("ID")["Target"]
    high = pd.read_csv(a.pair[1], dtype={"ID": "string"}).set_index("ID")["Target"]
    tt = test.assign(low=test["ID"].map(low).to_numpy(), high=test["ID"].map(high).to_numpy())
    tt = tt[(tt["hour_start"] >= start) & (tt["hour_start"] <= end)]
    hours = pd.date_range(start, end, freq="h")
    outlook, daily = {}, {}
    for loc in SHORT:
        g = tt[tt["location_id"] == loc].set_index("hour_start").reindex(hours)
        mid = g["low"] / 0.95
        outlook[loc] = {"t": iso(hours), "low": nums(g["low"]), "high": nums(g["high"]), "mid": nums(mid)}
        rows = []
        for day, gg in pd.DataFrame({"low": g["low"], "high": g["high"], "mid": mid}).groupby(hours.normalize()):
            m = gg["mid"].to_numpy()
            blk = [np.nanmean(m[i:i + 4]) for i in range(0, max(1, len(m) - 3))]
            i0 = int(np.nanargmax(blk)) if len(blk) and np.isfinite(blk).any() else 0
            rows.append({"date": day.strftime("%Y-%m-%d"), "low": round(float(gg["low"].mean()), 1), "high": round(float(gg["high"].mean()), 1),
                         "mid": round(float(gg["mid"].mean()), 1), "worst": [int(gg.index[i0].hour), int(gg.index[min(i0 + 3, len(m) - 1)].hour) + 1]})
        daily[loc] = rows

    obs_idx = pd.date_range(t_end - pd.Timedelta(days=14) + pd.Timedelta(hours=1), t_end, freq="h")
    observed = {loc: {"t": iso(obs_idx), "v": nums(grid[loc].reindex(obs_idx))} for loc in SHORT}

    # Ventilation = boundary-layer height x 10 m wind (m2/s). Day labels use the night hours (20:00-07:00), when the layer is
    # shallow and PM2.5 builds up; the daytime BLH spike would otherwise dominate a daily mean.
    vent, refs, stamp = {}, [], ""
    vidx = pd.date_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=1), freq="h")
    for _, s in st.iterrows():
        w, stamp = weather(a.weather, s.latitude, s.longitude, a.cache)
        vc = w["boundary_layer_height"] * w["wind_speed_10m"] / 3.6
        vent[s.location_id] = {"t": iso(vidx), "vc": nums(vc.reindex(vidx)), "blh": nums(w["boundary_layer_height"].reindex(vidx)),
                               "wind": nums(w["wind_speed_10m"].reindex(vidx))}
        night = vc[(vc.index.hour >= 20) | (vc.index.hour < 7)]
        refs.append(night.loc[REF[0]:REF[1]].resample("D").mean())
        for row in daily[s.location_id]:
            d = pd.Timestamp(row["date"], tz=IST)
            row["vc"] = round(float(vc.loc[d: d + pd.Timedelta(hours=23)].mean()), 0)
            row["vc_night"] = round(float(night.loc[d - pd.Timedelta(hours=4): d + pd.Timedelta(hours=6)].mean()), 0)
    t1, t2 = [round(float(x), 0) for x in pd.concat(refs, axis=1).mean(axis=1).quantile([1 / 3, 2 / 3])]
    for loc in SHORT:
        for row in daily[loc]:
            row["label"] = "stagnant" if row["vc_night"] < t1 else ("mixed" if row["vc_night"] < t2 else "ventilated")

    # Diwali lens
    years = []
    h1820 = history_2018_2020(a.data)
    for y in (2018, 2019):
        r = diwali_year(y, h1820, "CPCB history 2018-2020 (MPCB stations)")
        if r:
            years.append(r)
    r = diwali_year(2024, history_2024(a.data), "XKDR archive 2024 (all Mumbai stations)", NAME_TO_LOC)
    if r:
        years.append(r)
    other = load_other(a.data)
    both = grid.rename(columns={v: k for k, v in NAME_TO_LOC.items()}).join(
        other.pivot_table(index="hour_start", columns="station_name", values="pm25", aggfunc="mean"), how="outer")
    for y in (2025, 2026):
        if both.index.max() >= pd.Timestamp(DIWALI[y], tz=IST) - pd.Timedelta(days=2):
            r = diwali_year(y, both, "challenge Train + other Mumbai stations", NAME_TO_LOC)
            if r:
                years.append(r)
    print("Diwali years:", [(r["year"], r["n_stations"], r["peak_rel"], r["peak_value"], r["clear_rel"], r["excess"]) for r in years])
    ex = [r["excess"] for r in years if r["n_stations"] >= 3 and r["excess"]]
    d26 = pd.Timestamp(DIWALI[2026], tz=IST)
    for loc in SHORT:
        for row in daily[loc]:
            k = (pd.Timestamp(row["date"], tz=IST) - d26).days
            if -1 <= k <= 2:
                row["diwali"] = "Diwali eve" if k == -1 else ("Diwali" if k == 0 else f"Diwali +{k}")
    diwali_note = (f"On past Diwalis with 3+ stations the festival days ran {min(ex):.1f}x to {max(ex):.1f}x the pre-festival level, "
                   "peaking after midnight; the weather-driven band does not include that.") if ex else ""

    # Map: targets (latest Train hour) + other Mumbai stations (latest provided hour); coordinates from the XKDR lookup
    look = pd.read_csv(a.data / "AdditionalHistory" / "XKDR_All_Stations.csv", usecols=["station_name", "latitude", "longitude"])
    look = look.drop_duplicates("station_name").set_index("station_name")
    pts = []
    for _, s in st.iterrows():
        col = grid[s.location_id].dropna()
        pts.append({"name": s.station_name, "lat": float(s.latitude), "lon": float(s.longitude), "target": True, "id": s.location_id,
                    "latest_t": col.index[-1].strftime("%Y-%m-%dT%H:%M"), "latest_v": round(float(col.iloc[-1]), 1), "n_hours": int(col.size)})
    for name, g in other.dropna(subset=["pm25"]).groupby("station_name"):
        if name in look.index:
            g = g.sort_values("hour_start")
            pts.append({"name": name, "lat": float(look.loc[name, "latitude"]), "lon": float(look.loc[name, "longitude"]), "target": False,
                        "latest_t": g["hour_start"].iloc[-1].strftime("%Y-%m-%dT%H:%M"), "latest_v": round(float(g["pm25"].iloc[-1]), 1), "n_hours": int(len(g))})

    out = {"generated_ist": pd.Timestamp.now(IST).strftime("%Y-%m-%d %H:%M"), "train_end_ist": t_end.strftime("%Y-%m-%d %H:%M"),
           "weather_pull": stamp, "origin": a.pair[0].name.split("origin")[-1].replace(".csv", "") if "origin" in a.pair[0].name else "",
           "today": today.strftime("%Y-%m-%d"), "window": {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")},
           "stations": [{"id": s.location_id, "name": s.station_name, "short": SHORT[s.location_id], "lat": float(s.latitude), "lon": float(s.longitude)}
                        for _, s in st.iterrows()],
           "observed": observed, "outlook": outlook, "daily": daily, "ventilation": vent, "vc_terciles": [t1, t2],
           "vc_note": f"terciles of the night-time (20:00-07:00) ventilation, {REF[0]} to {REF[1]}",
           "diwali": years, "diwali_note": diwali_note, "map": pts,
           "sources": ["Challenge data: CPCB/CAAQMS via OpenAQ, XKDR, Rohan Rao/CPCB-derived (CC-BY-SA 4.0, Zindi)",
                       "Weather: Open-Meteo forecast and historical-forecast APIs (CC-BY 4.0), listed on ClimateVerse",
                       "Forecast band: the two legs of our hedged submission (x0.95 and x1.4 of a five-model average)"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.out} ({a.out.stat().st_size / 1024:.0f} KB); weather {stamp}; night ventilation terciles {t1:.0f} / {t2:.0f} m2/s")


if __name__ == "__main__":
    main()
