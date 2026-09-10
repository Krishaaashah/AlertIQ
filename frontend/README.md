# AlertIQ Dashboard (frontend)

React + Vite dashboard for the AlertIQ backend — live alert metrics,
model status, drift status, and a research/architecture walkthrough.

## Development

```bash
npm install
npm run dev
```

Opens at [http://localhost:5173](http://localhost:5173) by default. The
dashboard expects the FastAPI backend to be running at
`http://localhost:8000` (see the root `README.md` to start it, or run
`docker compose up` from the project root to start both together).

## Structure

```text
src/
|-- App.jsx                  # Top-level layout/routing
|-- components/
|   |-- Charts.jsx            # Metric charts (recharts)
|   |-- LiveFeed.jsx          # Live alert decision feed
|   |-- NeuralScene.jsx       # 3D visualization (react-three-fiber)
|   `-- ResearchPage.jsx      # Architecture / methodology walkthrough
`-- data/
    `-- simulationData.js     # Local demo data for offline/no-backend view
```

## Build

```bash
npm run build
```

Outputs static files to `dist/`, served by `frontend/Dockerfile` via nginx
in the Docker setup (see the root `docker-compose.yml`).

## Lint

```bash
npm run lint
```
