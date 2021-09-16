def name_as(name):
    def wrapper(func):
        func.__name__ = name
        return func

    return wrapper
