import time


# Get time to generate output name
def gettrainname(name):
    '''
    Get the train name with the training start full date.
    Arguments:
    @name (string): network name
    Returns: full_name (string)
    '''
    tm = time.gmtime()
    st_mon = str(tm.tm_mon) if tm.tm_mon > 9 else '0'+ str(tm.tm_mon)
    st_day = str(tm.tm_mday) if tm.tm_mday > 9 else '0'+ str(tm.tm_mday)
    st_hour = str(tm.tm_hour) if tm.tm_hour > 9 else '0'+ str(tm.tm_hour)
    st_min = str(tm.tm_min) if tm.tm_min > 9 else '0'+ str(tm.tm_min)
    tm_str = str(tm.tm_year) + st_mon + st_day + '_' + st_hour + st_min
    # log name - eg: both
    return tm_str + '_' + name