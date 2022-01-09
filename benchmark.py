import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time_ns()
        result = func(*args, **kwargs)
        end = time.time_ns()
        execution = end-start
        return execution, result
    return wrapper