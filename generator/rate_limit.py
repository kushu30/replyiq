import time
from functools import wraps

from google.api_core.exceptions import ResourceExhausted


def retry_with_backoff(max_retries: int = 3, base_delay: int = 15):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except ResourceExhausted:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = base_delay * (attempt + 1)
                    print(f"Rate limit hit, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            return func(*args, **kwargs)
        return wrapper
    return decorator