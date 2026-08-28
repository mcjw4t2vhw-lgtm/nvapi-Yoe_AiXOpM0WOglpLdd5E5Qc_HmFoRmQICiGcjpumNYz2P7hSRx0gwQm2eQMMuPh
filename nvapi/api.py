"""HTTP surface for GPU inventory."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from nvapi.errors import GpuNotFoundError, NvApiError
from nvapi.service import NvApi


def create_app(api: NvApi | None = None) -> FastAPI:
    service = api or NvApi.auto()
    app = FastAPI(title="nvapi", version="1.1.0")
    app.state.nvapi = service

    @app.exception_handler(GpuNotFoundError)
    async def missing_gpu(_request: Request, exc: GpuNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(NvApiError)
    async def nvapi_error(_request: Request, exc: NvApiError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, object]:
        summary = service.summary()
        return {
            "ok": summary.worst_status != "critical",
            "summary": summary.model_dump(),
        }

    @app.get("/driver")
    def driver() -> dict[str, object]:
        return service.driver().model_dump()

    @app.get("/gpus")
    def gpus() -> list[dict[str, object]]:
        return [gpu.model_dump() for gpu in service.list_gpus()]

    @app.get("/gpus/{index}")
    def gpu(index: int) -> dict[str, object]:
        if index < 0:
            raise HTTPException(status_code=422, detail="index must be >= 0")
        return service.get_gpu(index).model_dump()

    return app


app = create_app()
