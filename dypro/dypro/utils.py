from calendar import monthrange
from math import ceil
import logging

logging.basicConfig(level=logging.ERROR)

SECONDS_OF_DAY = 24*60*60

def limit(value, min=None, max=None):
    """ If value is lower than min, return min. If value is greater than max, return max. Else return value

    Args:
        value ([type]): [description]
        min ([type], optional): [description]. Defaults to None.
        max ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """  

    if (min != None and max != None):
        returnValue = min if (value<min) else (max if (value>max) else value)
    elif (min != None):
        returnValue = min if (value<min) else value
    elif (max != None):
        returnValue = max if (value>max) else value
    else:
        returnValue = value

    return returnValue


def daysOfMonth(year:int, month:int) -> int:
    """[summary]

    Args:
        year (int): [description]
        month (int): choose the month from 0 (january) to 11 (december)

    Returns:
        int: number of days on correspondent month
    """    

    return monthrange(year, month+1)[1]

def secondsOfMonth(year:int, month:int) -> int:
    return daysOfMonth(year, month+1) * SECONDS_OF_DAY

def nearestSample(value, min, max, period=1):
    
    if value < min:
        output = min

    elif value > max:
        output = max

    else:
        i_samp = round((value - min)/period)
        output = limit(min + period * i_samp, min, max)

    # logging.debug(f"{value} {min} {max} -> {output}")
    
    return output

def sampling(min, max, period=1):
    """Function to generate a discretization of interval, given a step.

    Args:
        min ([type]): minimum value of interval
        max ([type]): maximum value of interval
        period (optional): Size of step. Defaults to 1.

    Yields:
        [type]: returns all discretized values of interval
    """    
    _maxOfRange = ceil((max-min)/period)
    
    for i in range(_maxOfRange + 1):
        value = min + period*i
        value = value if value<max else max

        yield value
