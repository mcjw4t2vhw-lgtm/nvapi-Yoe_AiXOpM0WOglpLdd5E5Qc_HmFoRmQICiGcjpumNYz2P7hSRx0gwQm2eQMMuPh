# nvapi

Query NVIDIA GPU inventory, VRAM pressure, and driver info. Uses `nvidia-smi` when it is installed; otherwise a deterministic mock backend so tests and local demos still run.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Library

```python
from nvapi import NvApi

api = NvApi.mock()          # always available
# api = NvApi.auto()        # nvidia-smi if present, else mock
# api = NvApi.nvidia_smi()  # require a real driver

for gpu in api.list_gpus():
    print(gpu.index, gpu.name, gpu.status, gpu.memory.used_percent)

print(api.summary())
print(api.driver())
```

GPU status:

| Status | When |
|--------|------|
| `healthy` | temp &lt; 80 °C and VRAM &lt; 90 % |
| `degraded` | temp ≥ 80 °C or VRAM ≥ 90 % |
| `critical` | temp ≥ 90 °C or VRAM ≥ 98 % |

## CLI

```bash
nvapi --backend mock list
nvapi --backend mock get 0
nvapi --backend mock summary
nvapi --backend mock driver
```

## HTTP API

```bash
uvicorn nvapi.api:app --host 0.0.0.0 --port 8000
```

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | Inventory roll-up; `ok` is false if any GPU is critical |
| GET | `/gpus` | All devices |
| GET | `/gpus/{index}` | One device (`404` if missing) |
| GET | `/driver` | Driver version and backend name |

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
```

## Layout

```
nvapi/          library, CLI, FastAPI app
tests/          pytest suite
```

## License

[MIT](LICENSE)
