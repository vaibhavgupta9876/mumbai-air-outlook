# Mumbai Air Outlook

An additional submission to the **Google Mumbai Data Challenge: Air Quality Mapping & Forecasting** (Zindi, 2026) for the
Most Creative Submission category. It turns the challenge's forecasting task into something a resident can act on.

**Live page:** https://vaibhavgupta9876.github.io/mumbai-air-outlook/ (one static HTML file plus `data.json`)

## What it shows

1. **Outlook.** The next seven days of hourly PM2.5 for the four challenge stations (Malad West, Borivali East, Byculla,
   Shivaji Nagar). The blue band is the pair of forecasts we actually submit, one scaled low and one scaled high around a
   five-model average. Under it, the ventilation index (boundary-layer height x 10 m wind) that sets the daily level, and
   day cards with the band, a stagnant / mixed / ventilated label from the night-time ventilation, and the worst four hours.
   Diwali eve to Diwali +2 are flagged: the weather forecast knows nothing about fireworks.
2. **Diwali lens.** Every Diwali in the challenge data (2018, 2019, 2024, 2025, and 2026 once the final data release covers
   it) lined up on one clock, D-3 to D+5, as the median Mumbai station per hour. A table gives the pre-festival level, the
   peak and the hour the smoothed level is back within 25% of normal. On the three multi-station years the festival days
   ran 1.6 to 1.9 times the pre-festival level, peaking after midnight.
3. **Map.** The four forecast stations and the other Mumbai stations in the challenge history, coloured by their latest
   provided reading on the Indian AQI PM2.5 bands.

## How it is built

```
build_data.py  --data <challenge files>  --pair forecast/low.csv forecast/high.csv  --weather api  -->  data.json
index.html     reads data.json; Google Charts for the plots, Leaflet + OpenStreetMap tiles for the map
```

`build_data.py` is standalone (pandas and numpy only). Inputs:

- the challenge files from Zindi: `Train.csv`, `Test.csv`, `Stations.csv` and the unzipped `AdditionalHistory/`
  (CPCB/CAAQMS via OpenAQ, the XKDR 2024 archive, the Rohan Rao CPCB-derived 2018-2020 set; CC-BY-SA 4.0);
- Open-Meteo's forecast API (31 past days + 16 ahead) and historical-forecast API (15 Oct to 15 Dec 2025, the reference
  period for the ventilation terciles). Open-Meteo is CC-BY 4.0 and is listed in the ClimateVerse challenge catalogue;
- the two legs of our hedged submission pair in `forecast/` (plain CSVs with `ID,Target`).

No other source is used. In particular, no live station readings beyond the released challenge files.

### Reproduce

In Google Colab: open `notebook.ipynb` (Colab badge below), upload the four challenge files when asked, run all cells.
It clones this repository, builds `data.json` from the live Open-Meteo API and shows the page inline.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaibhavgupta9876/mumbai-air-outlook/blob/main/notebook.ipynb)

Locally:

```
pip install pandas numpy
python build_data.py --data challenge --pair forecast/low.csv forecast/high.csv --weather api --out data.json
python -m http.server 8000      # then open http://localhost:8000
```

### Decisions worth knowing

- **Band, not interval.** The band is the two files we submit, not a confidence interval from a formula. In our backtests the
  daily level is what a six-day forecast gets wrong most, so the band spans the level; the hourly shape inside a day is
  well predicted. Central line = low leg / 0.95 (the unscaled average).
- **Night-time ventilation.** Daily-mean ventilation is dominated by the afternoon boundary layer and labels every day
  "ventilated". The night hours (20:00 to 07:00), when the layer is shallow and PM2.5 builds up, separate the days.
  Terciles come from 15 Oct to 15 Dec 2025.
- **Median station.** The Diwali city line is the median across stations with at least 50% coverage of the window and a
  window median above 10 ug/m3 (one station reported 1 to 8 ug/m3 through October 2025: a failing sensor, not clean air).
- **Sensor faults.** Isolated single-station spikes (over 200, over four times the station's own 7-hour median and over
  three times the other stations that hour) are blanked. City-wide events such as Diwali pass because every station rises
  together.

## Tools

Google Charts (plots), Google Colab (reproducible build), Google Maps links from each station marker, Leaflet with
OpenStreetMap tiles (map), pandas / numpy (data). The forecast itself comes from the challenge entry: Chronos-2
zero-shot with weather covariates, a weather-driven gradient-boosting model and covariate variants, averaged.

## Licence and disclaimer

Code: MIT. Data: as per each source above; the page attributes them in its footer. This is not an official advisory: for
health decisions use CPCB and BMC bulletins.
