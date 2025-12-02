"""FastAPI server for serving SLM models."""

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from slm_builder.models.base import ModelFactory, generate_text
from slm_builder.utils import get_logger

logger = get_logger(__name__)


class PredictRequest(BaseModel):
    """Request model for prediction."""

    prompt: str = Field(description="Input prompt")
    max_length: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    num_return_sequences: int = Field(default=1, ge=1, le=10)


class PredictResponse(BaseModel):
    """Response model for prediction."""

    generated_text: str
    prompt: str
    model: str


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction."""

    prompts: List[str] = Field(description="List of input prompts")
    max_length: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    """Response model for batch prediction."""

    results: List[PredictResponse]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_info: Dict[str, Any]


class SLMServer:
    """FastAPI server for SLM inference."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        """Initialize server.

        Args:
            model_dir: Directory with trained model
            device: Device to run inference on
        """
        self.model_dir = model_dir
        self.device = device
        self.app = FastAPI(
            title="SLM Builder Server",
            description="API for serving Small/Specialized Language Models",
            version="0.1.0",
        )

        # Load model
        logger.info("Loading model for serving", model_dir=model_dir, device=device)
        self.model, self.tokenizer = ModelFactory.load_model_and_tokenizer(model_dir, device=device)
        self.model.eval()

        self.model_info = ModelFactory.get_model_info(self.model)
        logger.info("Model loaded", info=self.model_info)

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model_loaded=True,
                model_info=self.model_info,
            )

        @self.app.post("/predict", response_model=PredictResponse)
        async def predict(request: PredictRequest):
            """Generate text from a prompt."""
            try:
                logger.debug("Prediction request", prompt=request.prompt[:50])

                generated = generate_text(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=request.prompt,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    device=self.device,
                )

                return PredictResponse(
                    generated_text=generated,
                    prompt=request.prompt,
                    model=self.model_dir,
                )
            except Exception as e:
                logger.error("Prediction failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/batch_predict", response_model=BatchPredictResponse)
        async def batch_predict(request: BatchPredictRequest):
            """Generate text for multiple prompts."""
            try:
                results = []
                for prompt in request.prompts:
                    generated = generate_text(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        max_length=request.max_length,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        device=self.device,
                    )
                    results.append(
                        PredictResponse(
                            generated_text=generated,
                            prompt=prompt,
                            model=self.model_dir,
                        )
                    )

                return BatchPredictResponse(results=results)
            except Exception as e:
                logger.error("Batch prediction failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/metrics")
        async def metrics():
            """Get server metrics."""
            import psutil

            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "model_info": self.model_info,
            }


def start_server(
    model_dir: str,
    host: str = "0.0.0.0",
    port: int = 8080,
    device: str = "cpu",
    reload: bool = False,
):
    """Start the FastAPI server.

    Args:
        model_dir: Directory with trained model
        host: Host address
        port: Port number
        device: Device for inference
        reload: Enable auto-reload (development)
    """
    import uvicorn

    server = SLMServer(model_dir=model_dir, device=device)

    logger.info("Starting server", host=host, port=port)

    uvicorn.run(
        server.app,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fastapi_server.py <model_dir> [host] [port]")
        sys.exit(1)

    model_dir = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080

    start_server(model_dir, host, port)
