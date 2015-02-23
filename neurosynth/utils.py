import warnings

def deprecated(*args):
    """ Deprecation warning decorator. Takes optional deprecation message,
    otherwise will use a generic warning. """
    def wrap(func):
        def wrapped_func(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return func(*args, **kwargs)
        return wrapped_func

    if len(args) == 1 and callable(args[0]):
        msg = "Function '%s' will be deprecated in future versions of " \
            "Neurosynth." % args[0].__name__
        return wrap(args[0])
    else:
        msg = args[0]
        return wrap
