# from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call
import platform
import signal
from contextlib import contextmanager

from pynisher import WallTimeoutException


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise WallTimeoutException("Timed out!")
    # from https://stackoverflow.com/questions/1854/how-to-identify-which-os-python-is-running-on/58071295#58071295
    if platform.system() == "Linux":
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        print("Time Limit Enforcement only available on Linux", flush=True)
        yield