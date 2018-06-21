import settings
import time
import logging
from functools import wraps

PROF_DATA = {}

def calltimeprofile(logger):
    def decorator(fn):
        @wraps(fn)
        def with_profiling(*args, **kwargs):
            func_name = fn.__name__
            #log = logging.getLogger("stpe1_ndsb")
            arg_names = fn.__code__.co_varnames
            params = dict(
                args=dict(zip(arg_names, args)),
                kwargs=kwargs)
            argstr = ', '.join(['{}={}'.format(str(k), repr(v)) for k, v in params.items()])
            
            start_time = time.time()

            ret = fn(*args, **kwargs)

            elapsed_time = time.time() - start_time

            logger.info("Call {0} with {1} elpase time: ------------- {2}".format(func_name, argstr, elapsed_time))

            if fn.__name__ not in PROF_DATA:
                PROF_DATA[fn.__name__] = [0, []]
            PROF_DATA[fn.__name__][0] += 1
            PROF_DATA[fn.__name__][1].append(elapsed_time)
            #input('Press any key to continue...')
            return ret

        return with_profiling
    return decorator

def print_prof_data(logger):
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        total_time = sum(data[1])
        logger.info("Function {0} called {1} times. ".format(fname, data[0]))
        logger.info("Execution time max: {0}, average: {1}, total: {2} ".format(max_time, avg_time, total_time))
        #print "Function %s called %d times. " % (fname, data[0]),
        #print 'Execution time max: %.3f, average: %.3f' % (max_time, avg_time)

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
