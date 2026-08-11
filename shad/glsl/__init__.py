
import os
from functools import cache

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_registered_includes = {}

def register_include(name, string):
    _registered_includes[name] = string


@cache
def load_glsl(name):
    if name in _registered_includes:
        return _registered_includes[name]
    
    for extension in ('.glsl', '', '.GLSL'):
        for folder in ((), ('include',)):
            file = os.path.join(THIS_DIR, *folder, name + extension)

            if os.path.isfile(file):
                with open(file, 'r')  as f:
                    data = f.read()
                    return data
    raise FileNotFoundError(f'No file named {name}')

__all__ = ['load_glsl', 'register_include']