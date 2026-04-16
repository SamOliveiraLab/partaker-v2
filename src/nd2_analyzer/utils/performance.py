import functools
import logging
import time
from typing import Callable

# Performance logger setup
perf_logger = logging.getLogger("performance")
perf_logger.setLevel(logging.INFO)


def timing_decorator(func_name: str = None):
    """Decorator to measure and log function execution time"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                name = func_name or func.__name__
                perf_logger.info(f"{name}: {execution_time:.4f}s")

                return result
            except Exception as e:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                name = func_name or func.__name__
                perf_logger.error(f"{name}: {execution_time:.4f}s (FAILED: {e})")
                raise

        return wrapper

    return decorator


class TimingContext:
    """Context manager for timing code blocks"""

    def __init__(self, operation_name: str):
        self.name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        execution_time = end_time - self.start_time
        if exc_type is None:
            perf_logger.info(f"{self.name}: {execution_time:.4f}s")
        else:
            perf_logger.error(f"{self.name}: {execution_time:.4f}s (FAILED: {exc_val})")
