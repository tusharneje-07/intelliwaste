# IntelliWaste

Synthetic LSTM waste-bin forecasting with citizen verification and an interactive web dashboard.

## Overview

IntelliWaste is a reproducible research prototype for hourly fill-level prediction across 12 synthetic waste bins. It trains a Keras LSTM on chronological hourly history, evaluates continuous fill predictions and five fill-status bands, and optionally fuses low-confidence predictions with citizen verification.

The repository also includes a Flask API and React dashboard. The dashboard provides operational summaries, a real OpenFreeMap/MapLibre basemap with synthetic bin markers and zone overlays, prediction testing, citizen fusion calculations, model diagnostics, and generated evaluation visualizations.

The data and metrics are synthetic demonstrations, not real-world municipal performance. The optional ResNet50 interface is classification-only and is never used in fill-level fusion.

## Features

- Reproducible hourly synthetic history for 12 bins across four zones.
- 24-hour lookback LSTM with one-hour forecast horizon.
- Chronological training, validation, and test evaluation.
- Continuous MAE/RMSE metrics and five-band fill-status metrics.
- Confidence threshold workflow with optional citizen verification.
- Reliability-weighted LSTM and citizen fusion with step-by-step formulas.
- Practical 7–48 hour time-to-full planning horizon.
- Flask JSON API for summaries, bins, predictions, fusion, evaluation, and artifacts.
- Responsive React + TypeScript dashboard with Recharts.
- Real road/building basemap using OpenFreeMap and MapLibre, without an API key.
- Synthetic zone overlays, clickable bin markers, map zoom controls, and drag interaction.
- Optional three-class ResNet50 waste classifier status flow.

## Quick Start

### Prerequisites

- Python `>=3.10,<3.14`
- [`uv`](https://docs.astral.sh/uv/)
- [Bun](https://bun.sh/)
- A browser with network access for OpenFreeMap map tiles

### Install Python dependencies

From the repository root:

```bash
uv sync
```

### Generate data and train the model

The model artifact is stored locally under `model/` and is ignored by Git. Generate the synthetic data and train it with:

```bash
uv run python train.py
```

### Start the backend

In one terminal:

```bash
uv run python app/backend/app.py
```

The Flask API runs at <http://localhost:5000>.

### Start the frontend

In a second terminal:

```bash
cd app/frontend
bun install
bun run dev
```

Open <http://localhost:5173>.

## Usage

1. Open the dashboard to review fill summaries, zones, markers, bin history, and time-to-full estimates.
2. Select **Run a test**, choose a bin, and run the existing LSTM prediction.
3. Review the confidence decision and always-visible calculation steps.
4. Enable the low-confidence demo only when testing citizen verification.
5. Select a citizen and fill range, then submit verification to calculate reliability weights and the fused result.
6. Open **Model evaluation** to review training history, metrics, model structure, and generated visualizations.

The main API routes are:

```text
GET  /api/health
GET  /api/summary
GET  /api/bins
GET  /api/bins/<bin_id>
GET  /api/citizens
POST /api/predict
POST /api/fusion
GET  /api/evaluation
GET  /api/artifacts/<path>
POST /api/classify
```

The map uses synthetic marker coordinates because the dataset contains no GPS locations. OpenFreeMap supplies the road and building context without requiring an API key; the map keeps attribution visible and hides basemap location labels.

## Project Structure

```text
intelliwaste/
├── intelliwaste.py              # Shared generation, preprocessing, model, and fusion logic
├── train.py                     # Data generation and LSTM training
├── test.py                      # Evaluation, fusion, and visualization generation
├── explain.py                   # Console walkthrough for the selected bin
├── pyproject.toml               # Python dependencies and version constraints
├── data/                        # Synthetic CSV inputs
├── model/                       # Local trained Keras artifact, ignored by Git
├── results/                     # Training and test outputs
├── app/
│   ├── README.md                # Web application run guide
│   ├── backend/app.py           # Flask API
│   └── frontend/
│       ├── src/App.tsx          # React dashboard
│       ├── src/styles.css       # Dashboard and map styling
│       ├── package.json         # Frontend scripts and dependencies
│       └── vite.config.ts       # Vite dev server and API proxy
├── PRD/                         # Product requirements
└── LICENSE                      # MIT license
```

## Development and Testing

Validate the Python entry points:

```bash
uv run python -m py_compile intelliwaste.py train.py test.py explain.py app/backend/app.py
```

Run the frontend type check and production build:

```bash
cd app/frontend
bun run check
bun run build
```

After starting Flask, check the API:

```bash
curl http://localhost:5000/api/health
curl http://localhost:5000/api/summary
curl http://localhost:5000/api/bins
```

Regenerate the experimental outputs when the data or model changes:

```bash
uv run python train.py
uv run python test.py
uv run python explain.py
```

## Documentation

- [Web application guide](app/README.md)
- [Product requirements](PRD/)
- [Agent and repository instructions](AGENT.md)
- [MIT license](LICENSE)

## Contributing

Contributions should preserve the synthetic-data methodology, keep model inputs and outputs reproducible, and avoid adding personal or production sensor data. Run the Python compilation checks and frontend build before submitting changes. Update the relevant README when setup commands, API routes, or user-visible workflows change.

## License

This project is licensed under the [MIT License](LICENSE).
