import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        execution_time = (end-start)
        return "{:.2f}".format(execution_time), result
    return wrapper