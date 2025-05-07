import platform
import resource
import logging


# ref https://stackoverflow.com/questions/41105733/limit-ram-usage-to-python-program
def _memory_limit_ratio(ratio):
    """Limit max memory usage to half."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # Convert KiB to bytes, and divide in two to half
    resource.setrlimit(resource.RLIMIT_AS, (int(ratio * _get_memory() * 1024), hard))


def _get_memory():
    with open("/proc/meminfo", "r") as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                free_memory += int(sline[1])
    return free_memory  # KiB


def limit_memory(ratio=0.75):
    def decorator(function):
        def wrapper(*args, **kwargs):
            _memory_limit_ratio(ratio)
            try:
                return function(*args, **kwargs)
            except MemoryError as e:
                mem = _get_memory() / 1024 / 1024
                print("Remain: %.2f GB" % mem)
                logging.error("\n\nERROR: Memory Exception\n")
                raise e

        return wrapper

    return decorator
