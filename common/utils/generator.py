from itertools import izip, product


def dict_generator(dicts):
    return (dict(izip(dicts, x)) for x in product(*dicts.itervalues()))


def tuple_generator(l1, l2):
    return product(*[l1, l2])

