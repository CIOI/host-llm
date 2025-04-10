import uvicorn
from src.config._container import Application
from src.bootstrap import bootstrap
import logging

application = Application()
env = application.environment().APP_ENV
# 로거 설정
logger = application.logger()
environment = Application.environment()
# 환경변수 로드
logger.info(f"Current settings: {environment}")

# 모든 로거의 기본 레벨을 INFO로 설정
logging.getLogger().setLevel(logging.INFO)

# 특정 라이브러리 로거의 레벨도 INFO로 설정
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("selenium").setLevel(logging.INFO)

# 애플리케이션 컨테이너 생성 및 앱 부트스트랩 안정성을 위해 동기로 제작
app = bootstrap(
    application,
)


if __name__ == "__main__":
    """
    Run the AI API server.

    Args:
    - --env: The environment to run the server in. Default is "dev".
    """

    print(f"🤖 LLM API is running in environment: {env}")

    workers = 1 if env == "test" else 4
    uvicorn.run(
        "src.app:app",
        host=environment.HOST,
        port=environment.PORT,
        reload=environment.DEBUG,
        workers=workers,
        log_level="info",
    )
