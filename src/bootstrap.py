from fastapi import FastAPI, APIRouter
from src.config._container import Application
import datetime
from contextlib import asynccontextmanager


def create_app(application: Application) -> FastAPI:
    """기본 FastAPI 앱 생성 및 동기적 설정"""
    environment = application.environment()

    app = FastAPI(
        title="DECODED LLM API",
        description="API for text generation using Hugging Face models",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        debug=not environment.APP_ENV == "test",
    )
    return app


async def initialize_async_resources(application: Application):
    """비동기 리소스 초기화"""
    pass


def configure_routes(app: FastAPI, application: Application) -> FastAPI:
    logger = application.logger()
    logger.info("start configure routes")

    router = APIRouter()
    # API 라우터 설정
    router.include_router(application.router().register_routes(APIRouter()))
    app.include_router(router)

    logger.info("configure routes completed")

    @app.get("/", tags=["Root"])
    async def read_root():
        return {"message": "WELCOME TO DECODED AI SERVER", "status": "running"}

    @app.get("/health", tags=["Health Check"])
    async def health_check():
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
            }

    return app


@asynccontextmanager
async def create_lifespan(app: FastAPI, application: Application):
    pass


def bootstrap(application: Application) -> FastAPI:
    """전체 애플리케이션 부트스트래핑"""
    app = create_app(application)
    logger = application.logger()
    configure_routes(app, application)
    logger.info("configure routes completed")

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        async with create_lifespan(app_, application):
            yield

    app.lifespan = lifespan
    return app
