"""
Small utilities for testing.
"""
import threading
import signal
import nose
import time
import os
import sys
import gc


# A decorator to run tests only when numpy is available
try:
    import numpy as np

    def with_numpy(func):
        """A decorator to skip tests requiring numpy."""
        return func

except ImportError:
    def with_numpy(func):
        """A decorator to skip tests requiring numpy."""
        def my_func():
            raise nose.SkipTest('Test requires numpy')
        return my_func
    np = None


# we use memory_profiler library for memory consumption checks
try:
    from memory_profiler import memory_usage

    def with_memory_usage(func):
        """A decorator to skip tests requiring memory_profiler."""
        return func

    def memory_used(func, *args, **kwargs):
        """Compute memory usage when executing func."""
        gc.collect()
        ref_mem = memory_usage(max_usage=True)
        mem_use = memory_usage((func, args, kwargs),
                               interval=.001, timeout=1, max_usage=True)
        return mem_use[0] - ref_mem

except ImportError:
    def with_memory_usage(func):
        """A decorator to skip tests requiring memory_profiler."""
        def dummy_func():
            raise nose.SkipTest('Test requires memory_profiler.')
        return dummy_func

    def memory_used(func, *args, **kwargs):
        """Return an invalid value."""
        return -1
    memory_usage = None

# A utility to kill the test runner in case a multiprocessing assumption
# triggers an infinite wait on a pipe by the master process for one of its
# failed workers

_KILLER_THREADS = dict()


def setup_autokill(module_name, timeout=30):
    """Timeout based suiciding thread to kill the test runner process

    If some subprocess dies in an unexpected way we don't want the
    parent process to block indefinitely.
    """
    if "NO_AUTOKILL" in os.environ or "--pdb" in sys.argv:
        # Do not install the autokiller
        return

    # Renew any previous contract under that name by first cancelling the
    # previous version (that should normally not happen in practice)
    teardown_autokill(module_name)

    def autokill():
        pid = os.getpid()
        print("Timeout exceeded: terminating stalled process: %d" % pid)
        os.kill(pid, signal.SIGTERM)

        # If were are still there ask the OS to kill ourself for real
        time.sleep(0.5)
        print("Timeout exceeded: killing stalled process: %d" % pid)
        os.kill(pid, signal.SIGKILL)

    _KILLER_THREADS[module_name] = t = threading.Timer(timeout, autokill)
    t.start()


def teardown_autokill(module_name):
    """Cancel a previously started killer thread"""
    killer = _KILLER_THREADS.get(module_name)
    if killer is not None:
        killer.cancel()
