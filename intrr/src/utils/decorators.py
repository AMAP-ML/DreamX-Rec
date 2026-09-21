import errno
import os
import signal
import time
from functools import wraps
from typing import Callable, Optional, Tuple, Type, TypeVar

from src.utils.pylogger import RankedLogger

T = TypeVar("T")  # return type

logger = RankedLogger(__name__)


class TimedOutException(Exception):
    pass


class RetriesFailedException(Exception):
    pass


class __RetriableTimeoutException(Exception):
    pass

def timeout(
    seconds=10,
    error_message=os.strerror(errno.ETIME),
    timeout_action_func=None,
    exception_thrown_on_timeout=TimedOutException,
    **timeout_action_func_params,
):
    """
    Decorator to exit a program when a function execution exceeds a specified timeout.

    This decorator exits the program when the decorated function timed out, and executes
    timeout_action_func before exiting. The timeout_action_func can be useful for cases
    like environment clean up (e.g., ray.shutdown()). Another way to handle timeouts,
    especially when there are multiple threads or child threads, is to run `func` with
    multiprocessing. However, multiprocessing does not work with Ray.

    References to the current timeout implementation:
    - https://docs.python.org/3/library/signal.html
    - https://www.saltycrane.com/blog/2010/04/using-python-timeout-decorator-uploading-s3/

    Parameters:
    - seconds (int): Timeout duration in seconds. Defaults to 10.
    - error_message (str): Error message to log on timeout. Defaults to os.strerror(errno.ETIME).
    - timeout_action_func (callable, optional): Function to execute before exiting on timeout.
    - exception_thrown_on_timeout (Exception): Exception to raise on timeout. Defaults to TimedOutException.
    - **timeout_action_func_params: Arbitrary keyword arguments passed to timeout_action_func.
    """

    def decorator(func):
        def _handler(signum, frame):
            logger.info(error_message)
            if timeout_action_func:
                timeout_action_func(**timeout_action_func_params)
            raise exception_thrown_on_timeout()

        def wrapper(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # cancel the alarm
                signal.alarm(0)
                # reinstall the old signal handler
                signal.signal(signal.SIGALRM, old)
            return result

        return wraps(func)(wrapper)

    return decorator


def retry(
    exception_to_check: Type = Exception,
    tries: int = 5,
    delay_s: int = 3,
    backoff: int = 2,
    max_delay_s: Optional[int] = None,
    fn_execution_timeout_s: Optional[int] = None,
    deadline_s: Optional[int] = None,
    should_throw_original_exception: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that can be added around a function to retry incase it fails i.e. throws some exceptions

    Args:
        exception_to_check (Optional[Type]): the exception to check. may be a tuple of
        exceptions to check. Defaults to Exception. i.e. catches everything
        tries (Optional[int]): [description]. number of times to try (not retry) before giving up. Defaults to 5.
        delay_s (Optional[int]): [description]. initial delay between retries in seconds. Defaults to 3.
        backoff (Optional[int]): [description]. backoff multiplier e.g. value of 2 will double the delay
            each retry. Defaults to 2.
        max_delay_s (Optional[int]): [description]. maximum delay between retries in seconds. Defaults to None.
        fn_execution_timeout_s (Optional[int]): Maximum time given before a single function
            execution should time out. Defaults to None.
        deadline_s (Optional[int]): [description]. Total time in seconds to spend retrying, fails if exceeds
            this time. Note this timeout can also stop the first execution, so ensure to provide
            a lot of extra room so retries can actually take place.
            If timeout occurs, src.common.utils.timeout.TimedOutException is raised.
            Defaults to None.
        should_throw_original_exception (Optional[bool]): Defaults to False.
    """

    def deco_retry(f) -> Callable[..., T]:
        @wraps(f)
        def f_retry(*args, **kwargs) -> T:  # type: ignore[type-var]
            mtries, mdelay = tries, delay_s

            def fn(*args, **kwargs) -> T:
                if fn_execution_timeout_s is not None:
                    timeout_individual_fn_call_decorator = timeout(
                        seconds=fn_execution_timeout_s,
                        exception_thrown_on_timeout=__RetriableTimeoutException,
                    )
                    return timeout_individual_fn_call_decorator(f)(*args, **kwargs)
                return f(*args, **kwargs)

            acceptable_exceptions: Tuple[Type[Exception], ...] = (
                exception_to_check
                if isinstance(exception_to_check, tuple)
                else (exception_to_check,)
            )
            acceptable_exceptions = acceptable_exceptions + (
                __RetriableTimeoutException,
            )

            ret_val: T
            while mtries >= 0:
                try:
                    ret_val = fn(*args, **kwargs)
                    break
                except TimedOutException:
                    raise
                except acceptable_exceptions as e:
                    if mtries == 0:
                        # Failed for the final time
                        logger.exception(f"Failed for the last time: {e}")
                        if should_throw_original_exception:
                            raise  # Reraise original exception
                        raise RetriesFailedException(
                            f"Retry failed, permanently failing {f.__module__}:{f.__name__}, see logs for {e}"
                        )
                    msg = f"{e}, Retrying {f.__module__}:{f.__name__} in {mdelay} seconds..."
                    logger.warning(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay = (
                        min(mdelay * backoff, max_delay_s)
                        if max_delay_s
                        else mdelay * backoff
                    )

            return ret_val

        if deadline_s is not None:
            global_retry_timeout_decorator = timeout(seconds=deadline_s)
            return global_retry_timeout_decorator(f_retry)

        return f_retry

    return deco_retry
