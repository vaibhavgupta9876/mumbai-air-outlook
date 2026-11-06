# Mumbai Air Outlook

An additional submission to the **Google Mumbai Data Challenge: Air Quality Mapping & Forecasting** (Zindi, 2026) for the
Most Creative Submission category. It turns the challenge's forecasting task into something a resident can act on.

**Live page:** https://vaibhavgupta9876.github.io/mumbai-air-outlook/ (one static HTML file plus `data.json`, refreshed
every morning by a GitHub Action)

## What it shows

- **Today, at a glance.** Four tiles, one per station: today's forecast band, its Indian AQI category, whether the night
  is stagnant or ventilated, the cleanest and worst hours, and how today compares with last week's measurements.
- **Plan your day.** Seven rows of 24 hourly cells coloured on the Indian AQI scale from the central forecast, the
  cleanest three hours outlined, and two sentences of advice per day. On Diwali days a thin second strip shows the hours
  scaled by what past Diwalis did, because the weather forecast knows nothing about fireworks.
- **Outlook.** The hourly band for the next seven days after the last fourteen days measured, and the ventilation index
  (boundary-layer height x 10 m wind) that sets the daily level.
- **Diwali lens.** Every Diwali in the challenge data (2018, 2019, 2024, 2025, and 2026 as it unfolds) lined up on one
  clock, D-3 to D+5, as the median Mumbai station per hour; a table with the pre-festival level, the peak and the hour the
  smoothed level is back within 25% of normal. On the three multi-station years the festival days ran 1.6 to 1.9 times
  the pre-festival level, peaking after midnight.
- **Why ventilation.** The evidence: for every station-day since February 2025, the night-time ventilation against that
  day's mean PM2.5, by season. Stagnant nights run about a fifth higher than ventilated ones in the October to February
  season. Honest about the size of the effect.
- **Map.** The four forecast stations with today's band, and the other Mumbai stations in the challenge history with their
  last 30 days, on the Indian AQI colours.

## How it is built

```
build_data.py  --weather api   challenge/ + forecast/low.csv + forecast/high.csv + Open-Meteo  -->  data.json
index.html     reads data.json; Google Charts for the plots, Leaflet + OpenStreetMap tiles for the map
```

`build_data.py` is standalone (pandas and numpy only). Inputs, all in this repository or fetched live:

- `challenge/`: the challenge files from Zindi, `Train.csv`, `Test.csv`, `Stations.csv` and `AdditionalHistory/`
  (CPCB/CAAQMS via OpenAQ, the XKDR 2024 archive, the Rohan Rao CPCB-derived 2018-2020 set). The challenge rules license
  this data CC-BY-SA 4.0 for any purpose, which is why it can ship here;
- Open-Meteo's forecast API (31 past days + 16 ahead) and historical-forecast API (from 1 Feb 2025). Open-Meteo is CC-BY
  4.0 and is listed in the ClimateVerse challenge catalogue;
- `forecast/low.csv` and `forecast/high.csv`: the two legs of our hedged challenge submission (plain `ID,Target` CSVs).

No other source. In particular, no live station readings beyond the released challenge files.

The day-by-day advice is written from the forecast numbers by a template, or by Gemini (`gemini-2.5-flash`) when a
`GEMINI_API_KEY` is present at build time; the page itself never calls an API.

### Reproduce

In Google Colab, nothing to upload: open `notebook.ipynb` and run all cells. It clones this repository, builds `data.json`
from the live Open-Meteo API, draws the Diwali lens and shows the page inline.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaibhavgupta9876/mumbai-air-outlook/blob/main/notebook.ipynb)

Locally:

```
pip install pandas numpy
python build_data.py --weather api --out data.json
python -m http.server 8000      # then open http://localhost:8000
```

The GitHub Action in `.github/workflows/refresh.yml` runs the same build every morning at 08:30 IST and commits
`data.json`, so the ventilation outlook and the day labels stay current between forecast uploads.

### Decisions worth knowing

- **Band, not interval.** The band is the two files we submit, not a confidence interval from a formula. In our backtests
  the daily level is what a six-day forecast gets wrong most, so the band spans the level; the hourly shape inside a day is
  well predicted. Central line = low leg / 0.95 (the unscaled average).
- **Night-time ventilation.** Daily-mean ventilation is dominated by the afternoon boundary layer and labels every day
  "ventilated". The night hours (20:00 to 07:00), when the layer is shallow and PM2.5 builds up, separate the days.
  Terciles come from 15 Oct to 15 Dec 2025.
- **Median station.** The Diwali city line is the median across stations with at least 50% coverage of the window and a
  window median above 10 ug/m3 (one station reported 1 to 8 ug/m3 through October 2025: a failing sensor, not clean air).
- **Sensor faults.** Isolated single-station spikes (over 200, over four times the station's own 7-hour median and over
  three times the other stations that hour) are blanked. City-wide events such as Diwali pass because every station rises
  together.
- **Diwali scenario.** The thin strip on festival days multiplies the forecast by the mid-point of the past festival
  excess (1.6 to 1.9). It is a scenario, labelled as such, not a forecast.

## Tools

Google Charts (plots), Google Colab (reproducible build), Gemini (optional advice text), Google Maps links from each
station marker, GitHub Actions and Pages (daily refresh and hosting), Leaflet with OpenStreetMap tiles (map), pandas and
numpy (data). The forecast itself comes from the challenge entry: Chronos-2 zero-shot with weather covariates, a
weather-driven gradient-boosting model and covariate variants, averaged.

## Licence and disclaimer

Code: MIT. Challenge data: CC-BY-SA 4.0 (Zindi; CPCB/CAAQMS, OpenAQ, XKDR, Rohan Rao). Weather: Open-Meteo, CC-BY 4.0.
The page attributes them in its footer. This is not an official advisory: for health decisions use CPCB and BMC bulletins.
