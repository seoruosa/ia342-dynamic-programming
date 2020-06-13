from collections.abc import Hashable
import functools

class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''

    def __init__(self, func):
        self.func = func
        _cache = self._cache = {}
    
    def __call__(self, *args):
        if not isinstance(args, Hashable):
            # uncacheable. a list, for instance
            # better to not cache than blow up
            return self.func(*args)
        if args in self._cache:
            return self._cache[args]
        else:
            value = self.func(*args)
            self._cache[args] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)
    
    def cache(self):
        return self._cache
    
    def update(self, argMap:dict):
        self._cache.update(argMap)

if __name__ == "__main__":
    @memoized
    def fibonacci(n):
        "Return the nth fibonacci number."
        if n in (0, 1):
            return n
        return fibonacci(n-1) + fibonacci(n-2)

    print(fibonacci(2))
    fibonacci.update({(1,): 1, (0,): 0, (2,): 1, (3,): 2, (4,): 3, (5,): 5, (6,): 8, (7,): 13, (8,): 21, (9,): 34, (10,): 55, (11,): 89, (12,): 144})
    print(fibonacci.cache())
    print(fibonacci(100))
    print(fibonacci.cache())