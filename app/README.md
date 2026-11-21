# IntelliWaste Web Application

The web application combines a Flask JSON API with a React + TypeScript
dashboard over the existing IntelliWaste CSV artifacts and Keras model.

## Run

From the repository root:

```bash
uv sync
uv run python train.py
uv run python app/backend/app.py
```

In another terminal:

```bash
cd app/frontend
bun install
bun run dev
```

Open <http://localhost:5173>.

The map uses OpenFreeMap and MapLibre for real road/building context without an
API key. Basemap location labels are hidden, while IntelliWaste synthetic bin
markers and zone overlays remain interactive. The backend does not train a new
model during startup; it loads `model/intelliwaste_lstm.keras` and applies the
shared preprocessing in `intelliwaste.py`.

## Frontend checks

```bash
cd app/frontend
bun run check
bun run build
```
