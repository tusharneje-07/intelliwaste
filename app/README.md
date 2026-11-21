# IntelliWaste Web Application

React + TypeScript frontend with a Flask API over the existing IntelliWaste
CSV artifacts and Keras model.

## Run

From the repository root:

```bash
uv sync
uv run python app/backend/app.py
```

In another terminal:

```bash
cd app/frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

The map is an illustrative React Konva schematic because the dataset contains
no geographic coordinates. Recharts provides the history and evaluation
charts. The backend does not train a new model; it loads
`model/intelliwaste_lstm.keras` and applies the existing preprocessing in
`intelliwaste.py`.
