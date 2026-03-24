
# Copyright (C) 2021 Jeacom
# Jean3dimensional@gmail.com
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

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