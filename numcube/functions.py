def apply(cube, func, *args):
    return cube.apply(func, *args)


def reduce(cube, func, axis=None, keep=None, group=None, sort_grp=True):
    return cube.reduce(func, axis, keep, group, sort_grp)


def intersect(cubes):
    # TODO
    raise NotImplementedError


def envelope(cubes):
    # TODO
    raise NotImplementedError
