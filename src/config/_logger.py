import logging
import sys
from ._environment import Environment
import functools
from fastapi import HTTPException


class LoggerService:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """기본 로거를 설정합니다."""
        logger = logging.getLogger("app")

        # 이미 핸들러가 있으면 설정 생략
        if logger.handlers:
            return logger

        # 로그 레벨 설정
        if self.environment and hasattr(self.environment, "log_level"):
            log_level = getattr(
                logging,
                self.environment.log_level,
                logging.INFO,
            )
        else:
            log_level = logging.INFO

        logger.setLevel(log_level)

        # 콘솔 핸들러 추가
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def info(self, message):
        """정보 수준 로그를 기록합니다."""
        self.logger.info(message)

    def error(self, message, exc_info=None):
        """오류 수준 로그를 기록합니다."""
        self.logger.error(message, exc_info=exc_info)

    def warning(self, message):
        """경고 수준 로그를 기록합니다."""
        self.logger.warning(message)

    def debug(self, message):
        """디버그 수준 로그를 기록합니다."""
        self.logger.debug(message)

    def critical(self, message):
        """심각한 오류 수준 로그를 기록합니다."""
        self.logger.critical(message)


# container.py에서 사용할 팩토리 함수
def get_logger(environment=None):
    logger_service = LoggerService(environment)
    return logger_service


def http_error_handler(func):
    """HTTP 요청 처리 데코레이터: 예외 처리 자동화"""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            # 클라이언트 인스턴스를 종료하지 않고 로깅만 수행
            self.logger.error(
                f"HTTP request error in {func.__name__}: {str(e)}",
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error in HTTP request: {str(e)}",
            )

    return wrapper
