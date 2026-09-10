# Config

AlertIQ splits configuration into two layers, deliberately:

| Layer | Location | What lives there | Who changes it |
| --- | --- | --- | --- |
| **Deployment config** | `config/.env.example` | Host, port, CORS origins, frontend API base URL | Whoever is deploying/running the service |
| **Pipeline config** | `src/config.py` | Alert rule thresholds, reward values, drift thresholds, RL hyperparameters, file paths | Whoever is tuning the ML/RL system itself |

Deployment settings are environment-specific and change per machine —
they belong in an `.env` file, not committed source. Pipeline settings
(like the asymmetric reward structure) are safety-relevant constants that
the model, the RL agent, and the test suite (`tests/test_config.py`) all
have to agree on — keeping them as one importable Python module avoids
having those numbers drift out of sync between code and config.

To use the deployment config:

```bash
cp config/.env.example .env
# edit .env as needed
```

`docker-compose.yml` at the project root reads from this same `.env` file.
