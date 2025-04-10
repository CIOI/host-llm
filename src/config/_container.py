from dependency_injector import containers
from dependency_injector.containers import WiringConfiguration
from dependency_injector.providers import (
    Dependency,
    Callable,
    Singleton,
)
from src.controllers.huggingface import HuggingFaceController
from ._environment import Environment
from ._logger import LoggerService, get_logger
from src.services.cache_service import CacheService
from src.services.llm_service import LLMService
from src.routes.endpoints.text_generation import TextGenerationRouter


class Application(containers.DeclarativeContainer):
    wiring_config = WiringConfiguration(
        auto_wire=False,
        modules=[
            "preprocessing",
            "services",
        ],
    )
    environment = Dependency(
        Environment,
        default=Environment.from_env_file(),
    )
    logger: Callable[LoggerService] = Callable(
        get_logger,
        environment=environment,
    )

    huggingface_controller = Singleton(
        HuggingFaceController,
        environment=environment,
    )
    cache_service = Singleton(
        CacheService,
        environment=environment,
    )
    llm_service = Singleton(
        LLMService,
        environment=environment,
        cache_service=cache_service,
        huggingface_controller=huggingface_controller,
    )
    router = Singleton(
        TextGenerationRouter,
        llm_service=llm_service,
        logger=logger,
    )
