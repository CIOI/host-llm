from dependency_injector.containers import (
    WiringConfiguration,
    DeclarativeContainer,
)
from dependency_injector.providers import (
    Dependency,
    Callable,
    Singleton,
    Container,
)
from src.controllers.huggingface import HuggingFaceController
from ._environment import Environment
from ._logger import LoggerService, get_logger
from src.models.llm_models import ModelRegistry
from src.services.cache_service import CacheService
from src.services.llm_service import LLMService
from src.routes.endpoints.text_generation import TextGenerationRouter


class HuggingFaceContainer(DeclarativeContainer):
    environment = Dependency(Environment)
    cache_service = Dependency(CacheService)
    logger = Dependency(LoggerService)

    model_registry = Singleton(
        ModelRegistry,
    )
    controller = Singleton(
        HuggingFaceController,
        environment=environment,
        model_registry=model_registry,
        logger=logger,
    )
    llm_service = Singleton(
        LLMService,
        environment=environment,
        cache_service=cache_service,
        huggingface_controller=controller,
    )
    router = Singleton(
        TextGenerationRouter,
        llm_service=llm_service,
        logger=logger,
    )


class Application(DeclarativeContainer):
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
    cache_service = Singleton(
        CacheService,
        environment=environment,
    )
    huggingface = Container(
        HuggingFaceContainer,
        environment=environment,
        cache_service=cache_service,
        logger=logger,
    )
