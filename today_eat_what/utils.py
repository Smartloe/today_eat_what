import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple, TypeVar

T = TypeVar("T")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def load_dotenv(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in os.environ:
            os.environ[key.strip()] = value.strip()


def run_with_timeout(func: Callable[..., T], timeout: float, *args: Any, **kwargs: Any) -> T:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)


def run_with_retry(
    func: Callable[..., T],
    attempts: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[type[BaseException], ...] = (Exception,),
    *args: Any,
    **kwargs: Any,
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as exc:  # pragma: no cover - defensive
            if attempt == attempts:
                raise
            time.sleep(delay)
    raise RuntimeError("unreachable retry loop")
