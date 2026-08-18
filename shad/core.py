
import re
from math import ceil, log2
from dataclasses import dataclass
from .glsl import load_glsl
import gpu
from gpu.capabilities import max_texture_size_get
from gpu.types import (GPUShaderCreateInfo,
                       GPUStageInterfaceInfo,
                       GPUUniformBuf,
                       GPUTexture,
                       Buffer,
                       GPUShader)
import numpy as np
import inspect
from functools import wraps, cache
from copy import deepcopy
from textwrap import dedent, indent
import sys

_default_include = ('2d_indexing',)


class UsageFlag(str):
    @staticmethod
    def _validate_subitem(subitem):
        if isinstance(subitem, UsageFlag):
            if subitem._subitems:
                raise ValueError('nesting usage flags is not supported')
            return subitem
        elif isinstance(subitem, str):
            return UsageFlag(subitem, usage='STRING')

        raise TypeError('invalid subitem type', type(subitem))

    def __new__(cls, value, *args, **kwargs):
        instance = super().__new__(UsageFlag, value)  # pyright: ignore
        return instance

    def __init__(self, value, subitems=(), size: tuple | int = None, usage: str = None, np_dt: np.dtype = None):
        self._value = value
        self._subitems = tuple(self._validate_subitem(item) for item in subitems)
        self._size = size
        self._usage = usage
        self._np_dt = np_dt

    def __or__(self, value):
        if isinstance(value, (UsageFlag, tuple, str)):
            return self.__getitem__(value)
        else:
            raise ValueError(f'unsupported combination UsageFlag | {type(value)}')

    def __getitem__(self, key):
        if isinstance(key, int) and self._size is not None:
            return UsageFlag(self._value, self._subitems, key, self._usage, self._np_dt)
        elif isinstance(key, tuple):
            return UsageFlag(self._value, [*self._subitems, *key], self._size, self._usage, self._np_dt)
        elif isinstance(key, (UsageFlag, str)):
            return UsageFlag(self._value, (*self._subitems, key), self._size, self._usage, self._np_dt)
        else:
            raise ValueError(f'unsupported combination UsageFlag[{type(key)}]')

    def get_filledin_usages(self):
        filled_in = {self._usage: self._value}
        for subitem in self._subitems:
            if subitem._usage in filled_in:
                raise ValueError(f'duplicate usage for {self}, {self.to_string(usage=True)}')
            filled_in[subitem._usage] = subitem._value
        return filled_in

    def get_size(self):
        all_sizes = [self._size, *(subitem._size for subitem in self._subitems)]
        all_sizes = set(all_sizes) - {None, }
        if len(all_sizes) > 1:
            raise ValueError('multiple sizes found for ' + self.to_string())
        return all_sizes.pop() if all_sizes else 0

    def to_string(self, usage=False) -> str:
        ret = self._usage if usage else self._value

        if self._size not in (None, 0, 1):
            ret = f'{ret}[{self._size}]'

        if self._subitems:
            joined_subitems = ', '.join(item.to_string(usage) for item in self._subitems)
            return f'{ret}[{joined_subitems}]'

        return ret

    def __repr__(self):
        return self.to_string(usage=False)


#autopep8: off

# NOTE: blender has no buffer type exposed in the api that can be used with those.
# No idea what to do so leave them disabled.
# FLOAT_BUFFER          = UsageFlag('FLOAT_BUFFER',      size=None, usage='IMAGE_TYPE')
# INT_BUFFER            = UsageFlag('INT_BUFFER',        size=None, usage='IMAGE_TYPE')
# UINT_BUFFER           = UsageFlag('UINT_BUFFER',       size=None, usage='IMAGE_TYPE')

# Image Types
FLOAT_1D              = UsageFlag('FLOAT_1D',          size=None, usage='IMAGE_TYPE')
FLOAT_1D_ARRAY        = UsageFlag('FLOAT_1D_ARRAY',    size=None, usage='IMAGE_TYPE')
FLOAT_2D              = UsageFlag('FLOAT_2D',          size=None, usage='IMAGE_TYPE')
FLOAT_2D_ARRAY        = UsageFlag('FLOAT_2D_ARRAY',    size=None, usage='IMAGE_TYPE')
FLOAT_3D              = UsageFlag('FLOAT_3D',          size=None, usage='IMAGE_TYPE')
FLOAT_CUBE            = UsageFlag('FLOAT_CUBE',        size=None, usage='IMAGE_TYPE')
FLOAT_CUBE_ARRAY      = UsageFlag('FLOAT_CUBE_ARRAY',  size=None, usage='IMAGE_TYPE')
INT_1D                = UsageFlag('INT_1D',            size=None, usage='IMAGE_TYPE')
INT_1D_ARRAY          = UsageFlag('INT_1D_ARRAY',      size=None, usage='IMAGE_TYPE')
INT_2D                = UsageFlag('INT_2D',            size=None, usage='IMAGE_TYPE')
INT_2D_ARRAY          = UsageFlag('INT_2D_ARRAY',      size=None, usage='IMAGE_TYPE')
INT_3D                = UsageFlag('INT_3D',            size=None, usage='IMAGE_TYPE')
INT_CUBE              = UsageFlag('INT_CUBE',          size=None, usage='IMAGE_TYPE')
INT_CUBE_ARRAY        = UsageFlag('INT_CUBE_ARRAY',    size=None, usage='IMAGE_TYPE')
UINT_1D               = UsageFlag('UINT_1D',           size=None, usage='IMAGE_TYPE')
UINT_1D_ARRAY         = UsageFlag('UINT_1D_ARRAY',     size=None, usage='IMAGE_TYPE')
UINT_2D               = UsageFlag('UINT_2D',           size=None, usage='IMAGE_TYPE')
UINT_2D_ARRAY         = UsageFlag('UINT_2D_ARRAY',     size=None, usage='IMAGE_TYPE')
UINT_3D               = UsageFlag('UINT_3D',           size=None, usage='IMAGE_TYPE')
UINT_CUBE             = UsageFlag('UINT_CUBE',         size=None, usage='IMAGE_TYPE')
UINT_CUBE_ARRAY       = UsageFlag('UINT_CUBE_ARRAY',   size=None, usage='IMAGE_TYPE')
SHADOW_2D             = UsageFlag('SHADOW_2D',         size=None, usage='IMAGE_TYPE')
SHADOW_2D_ARRAY       = UsageFlag('SHADOW_2D_ARRAY',   size=None, usage='IMAGE_TYPE')
SHADOW_CUBE           = UsageFlag('SHADOW_CUBE',       size=None, usage='IMAGE_TYPE')
SHADOW_CUBE_ARRAY     = UsageFlag('SHADOW_CUBE_ARRAY', size=None, usage='IMAGE_TYPE')
DEPTH_2D              = UsageFlag('DEPTH_2D',          size=None, usage='IMAGE_TYPE')
DEPTH_2D_ARRAY        = UsageFlag('DEPTH_2D_ARRAY',    size=None, usage='IMAGE_TYPE')
DEPTH_CUBE            = UsageFlag('DEPTH_CUBE',        size=None, usage='IMAGE_TYPE')
DEPTH_CUBE_ARRAY      = UsageFlag('DEPTH_CUBE_ARRAY',  size=None, usage='IMAGE_TYPE')

_2D_IMAGE_FORMATS = {}

def _make_2d_format(imformat) -> tuple:
    typemap = {'UI' : UINT_2D,  'I'  : INT_2D,
               '8'  : FLOAT_2D, 'F'  : FLOAT_2D,
               '16F': FLOAT_2D, '16' : UINT_2D}

    for tk, imtype in typemap.items():
        if imformat._value.endswith(tk):
            imformat_2d = imtype[imformat]
            _2D_IMAGE_FORMATS[imformat._value] = imformat_2d
            return imformat, imformat_2d

    raise ValueError('un2difiable type')


# Format types & 2D image formats
RGBA8UI, RGBA8UI_2D    = _make_2d_format(UsageFlag('RGBA8UI',           size=None, usage='IMAGE_FORMAT'))
RGBA8I, RGBA8I_2D      = _make_2d_format(UsageFlag('RGBA8I',            size=None, usage='IMAGE_FORMAT'))
RGBA8, RGBA8_2D        = _make_2d_format(UsageFlag('RGBA8',             size=None, usage='IMAGE_FORMAT'))
RGBA32UI, RGBA32UI_2D  = _make_2d_format(UsageFlag('RGBA32UI',          size=None, usage='IMAGE_FORMAT'))
RGBA32I, RGBA32I_2D    = _make_2d_format(UsageFlag('RGBA32I',           size=None, usage='IMAGE_FORMAT'))
RGBA32F, RGBA32F_2D    = _make_2d_format(UsageFlag('RGBA32F',           size=None, usage='IMAGE_FORMAT'))
RGBA16UI, RGBA16UI_2D  = _make_2d_format(UsageFlag('RGBA16UI',          size=None, usage='IMAGE_FORMAT'))
RGBA16I, RGBA16I_2D    = _make_2d_format(UsageFlag('RGBA16I',           size=None, usage='IMAGE_FORMAT'))
RGBA16F, RGBA16F_2D    = _make_2d_format(UsageFlag('RGBA16F',           size=None, usage='IMAGE_FORMAT'))
RGBA16, RGBA16_2D      = _make_2d_format(UsageFlag('RGBA16',            size=None, usage='IMAGE_FORMAT'))
RG8UI, RG8UI_2D        = _make_2d_format(UsageFlag('RG8UI',             size=None, usage='IMAGE_FORMAT'))
RG8I, RG8I_2D          = _make_2d_format(UsageFlag('RG8I',              size=None, usage='IMAGE_FORMAT'))
RG8, RG8_2D            = _make_2d_format(UsageFlag('RG8',               size=None, usage='IMAGE_FORMAT'))
RG32UI, RG32UI_2D      = _make_2d_format(UsageFlag('RG32UI',            size=None, usage='IMAGE_FORMAT'))
RG32I, RG32I_2D        = _make_2d_format(UsageFlag('RG32I',             size=None, usage='IMAGE_FORMAT'))
RG32F, RG32F_2D        = _make_2d_format(UsageFlag('RG32F',             size=None, usage='IMAGE_FORMAT'))
RG16UI, RG16UI_2D      = _make_2d_format(UsageFlag('RG16UI',            size=None, usage='IMAGE_FORMAT'))
RG16I, RG16I_2D        = _make_2d_format(UsageFlag('RG16I',             size=None, usage='IMAGE_FORMAT'))
RG16F, RG16F_2D        = _make_2d_format(UsageFlag('RG16F',             size=None, usage='IMAGE_FORMAT'))
RG16, RG16_2D          = _make_2d_format(UsageFlag('RG16',              size=None, usage='IMAGE_FORMAT'))
R8UI, R8UI_2D          = _make_2d_format(UsageFlag('R8UI',              size=None, usage='IMAGE_FORMAT'))
R8I, R8I_2D            = _make_2d_format(UsageFlag('R8I',               size=None, usage='IMAGE_FORMAT'))
R8, R8_2D              = _make_2d_format(UsageFlag('R8',                size=None, usage='IMAGE_FORMAT'))
R32UI, R32UI_2D        = _make_2d_format(UsageFlag('R32UI',             size=None, usage='IMAGE_FORMAT'))
R32I, R32I_2D          = _make_2d_format(UsageFlag('R32I',              size=None, usage='IMAGE_FORMAT'))
R32F, R32F_2D          = _make_2d_format(UsageFlag('R32F',              size=None, usage='IMAGE_FORMAT'))
R16UI, R16UI_2D        = _make_2d_format(UsageFlag('R16UI',             size=None, usage='IMAGE_FORMAT'))
R16I, R16I_2D          = _make_2d_format(UsageFlag('R16I',              size=None, usage='IMAGE_FORMAT'))
R16F, R16F_2D          = _make_2d_format(UsageFlag('R16F',              size=None, usage='IMAGE_FORMAT'))
R16, R16_2D            = _make_2d_format(UsageFlag('R16',               size=None, usage='IMAGE_FORMAT'))

del _make_2d_format

# Special formats
R11F_G11F_B10F        = UsageFlag('R11F_G11F_B10F',    size=None, usage='IMAGE_FORMAT')
DEPTH32F_STENCIL8     = UsageFlag('DEPTH32F_STENCIL8', size=None, usage='IMAGE_FORMAT')
DEPTH24_STENCIL8      = UsageFlag('DEPTH24_STENCIL8',  size=None, usage='IMAGE_FORMAT')
SRGB8_A8              = UsageFlag('SRGB8_A8',          size=None, usage='IMAGE_FORMAT')
RGB16F                = UsageFlag('RGB16F',            size=None, usage='IMAGE_FORMAT')
SRGB8_A8_DXT1         = UsageFlag('SRGB8_A8_DXT1',     size=None, usage='IMAGE_FORMAT')
SRGB8_A8_DXT3         = UsageFlag('SRGB8_A8_DXT3',     size=None, usage='IMAGE_FORMAT')
SRGB8_A8_DXT5         = UsageFlag('SRGB8_A8_DXT5',     size=None, usage='IMAGE_FORMAT')
RGBA8_DXT1            = UsageFlag('RGBA8_DXT1',        size=None, usage='IMAGE_FORMAT')
RGBA8_DXT3            = UsageFlag('RGBA8_DXT3',        size=None, usage='IMAGE_FORMAT')
RGBA8_DXT5            = UsageFlag('RGBA8_DXT5',        size=None, usage='IMAGE_FORMAT')
DEPTH_COMPONENT32F    = UsageFlag('DEPTH_COMPONENT32F',size=None, usage='IMAGE_FORMAT')
DEPTH_COMPONENT24     = UsageFlag('DEPTH_COMPONENT24', size=None, usage='IMAGE_FORMAT')
DEPTH_COMPONENT16     = UsageFlag('DEPTH_COMPONENT16', size=None, usage='IMAGE_FORMAT')

READ                  = UsageFlag('READ',              size=None, usage='QUAL_READ')
WRITE                 = UsageFlag('WRITE',             size=None, usage='QUAL_WRITE')
NO_RESTRICT           = UsageFlag('NO_RESTRICT',       size=None, usage='QUAL_NORESTR')

FLOAT                 = UsageFlag('FLOAT', size=0, usage='GLTYPE', np_dt=np.dtype(np.float32))
VEC2                  = UsageFlag('VEC2',  size=0, usage='GLTYPE', np_dt=np.dtype((np.float32, 2)))
VEC3                  = UsageFlag('VEC3',  size=0, usage='GLTYPE', np_dt=np.dtype((np.float32, 3)))
VEC4                  = UsageFlag('VEC4',  size=0, usage='GLTYPE', np_dt=np.dtype((np.float32, 4)))

# dont support MAT3 for UBOStruct as its 3x3 size becomes problematic for alignment
# TODO: make a special edge case for this type in UBOStruct.
MAT3                  = UsageFlag('MAT3',  size=0, usage='GLTYPE') #, np_dt=np.dtype((np.float32, (3, 3))))

MAT4                  = UsageFlag('MAT4',  size=0, usage='GLTYPE', np_dt=np.dtype((np.float32, (4, 4))))
UINT                  = UsageFlag('UINT',  size=0, usage='GLTYPE', np_dt=np.dtype(np.uint32))
UVEC2                 = UsageFlag('UVEC2', size=0, usage='GLTYPE', np_dt=np.dtype((np.uint32, 2)))
UVEC3                 = UsageFlag('UVEC3', size=0, usage='GLTYPE', np_dt=np.dtype((np.uint32, 3)))
UVEC4                 = UsageFlag('UVEC4', size=0, usage='GLTYPE', np_dt=np.dtype((np.uint32, 4)))
INT                   = UsageFlag('INT',   size=0, usage='GLTYPE', np_dt=np.dtype(np.int32))
IVEC2                 = UsageFlag('IVEC2', size=0, usage='GLTYPE', np_dt=np.dtype((np.uint32, 2)))
IVEC3                 = UsageFlag('IVEC3', size=0, usage='GLTYPE', np_dt=np.dtype((np.uint32, 3)))
IVEC4                 = UsageFlag('IVEC4', size=0, usage='GLTYPE', np_dt=np.dtype((np.uint32, 4)))
BOOL                  = UsageFlag('BOOL',  size=0, usage='GLTYPE', np_dt=np.dtype(np.uint32))

# aliases that match the custom types defined by blender for glsl
# just for consistency.
FLOAT2 = VEC2
FLOAT3 = VEC3
FLOAT4 = VEC4
UINT2 = UVEC2
UINT3 = UVEC3
iINT2 = IVEC2
iINT3 = IVEC3
iINT4 = IVEC4

FLAT                  = UsageFlag('FLAT',              size=None, usage='INTERPOLATION')
SMOOTH                = UsageFlag('SMOOTH',            size=None, usage='INTERPOLATION')
NO_PERSPECTIVE        = UsageFlag('NO_PERSPECTIVE',    size=None, usage='INTERPOLATION')

# Shader stage qualifiers
VIN                   = UsageFlag('VIN',               size=None, usage='VIN')
VOUT                  = UsageFlag('VOUT',              size=None, usage='VOUT')
FRAG_OUT              = UsageFlag('FRAG_OUT',          size=None, usage='FRAG_OUT')

# Blending qualifiers
SRC_0                 = UsageFlag('SRC_0',             size=None, usage='BLEND_MODE')
SRC_1                 = UsageFlag('SRC_1',             size=None, usage='BLEND_MODE')

# ubos are now a class that can be passed like an UsageFlag object
# UBO                  = UsageFlag('UNIFORM_BUFFER',     size=None, usage='UNIFORM_BUFFER')


_ALL_USAGE_FLAGS = {name: flag for name, flag in locals().items() if isinstance(flag, UsageFlag)}


#autopep8: on


def _get_include(filenames, extra_source=''):
    files = []
    for fname in filenames:
        files.append(f'// {fname}')
        files.append(load_glsl(fname))
    files.append('\n\n')
    files.append(extra_source)
    return '\n\n'.join(files)


class UBOStructMeta(type):

    def __getitem__(cls, size):
        if isinstance(size, int):
            new_cls = type(cls.__name__, (cls,), dict(cls.__dict__))
            new_cls._size = size
            return new_cls
        raise ValueError('invalid type for defining a UBOStruct array, must be an int')


class UBOStruct(np.ndarray, metaclass=UBOStructMeta):
    '''
    Define an UBO using type annotations.
    Much like blender, it doesnt check every possible alignment issue that might happen, but tries its best.
    Its the developer responsibility to follow the std140 packing rules.

    https://developer.blender.org/docs/handbook/guidelines/glsl/#packing-rules

    '''
    # _data_arr = None
    # _dirty = True
    # _ubo = None
    # _matrix_elems = None
    _size = 1
    # _np_dt = 1
    # _substructs = None
    # _compiled_typedef_string = None

    def __init_subclass__(cls):
        annots = inspect.get_annotations(cls)

        dtype_elems = []
        typedef_lines = ['\nstruct ' + cls.__name__ + '{']
        seen_substructs = set()

        cls.struct_elems = []
        cls._matrix_elems = set()

        # Compile a typedef string and generate a numpy type from annotaions
        struct_size = 0

        type_conversion_map = {
            # blender defines its own types for glsl structs that get automatically converted
            # during the glsl cross-compilation
            # packed_float3 is partigularly advized over float3 in the documentation
            FLOAT: 'float',
            VEC2: 'float2',
            VEC3: 'packed_float3',
            VEC4: 'float4',
            UINT: 'uint',
            UVEC2: 'uint2',
            UVEC3: 'uint3',
            UVEC4: 'uint4',
            INT: 'int',
            IVEC2: 'int2',
            IVEC3: 'int3',
            IVEC4: 'int4',
            MAT3: 'mat3',
            MAT4: 'mat4',
        }

        def emit_struct_member(type, name, size):
            if type not in type_conversion_map:
                raise ValueError(f'unrecognized basic type {repr(type)}')
            type = type_conversion_map[type]

            if size > 1:
                size_str = f'[{int(size)}]'
            else:
                size_str = ''

            typedef_lines.append(f'    {type} {name}{size_str};')

        def emit_struct_def(definition):
            typedef_lines.insert(0, definition)

        def add_dtype_elem(name, dt, size):
            cls.struct_elems.append((name, dt, size))
            if v._size > 1:
                dtype_elems.append((name, dt * size))
            else:
                dtype_elems.append((name, dt))

        def emit_element(name, v):
            nonlocal struct_size
            if v._value.startswith('MAT'):
                cls._matrix_elems.add(name)

            if v._np_dt:
                itemsize = int(v._np_dt.itemsize)
                size = max(v._size, 1)
                elem_size = itemsize * size

                if size > 1 and v._value not in {'UVEC4', 'IVEC4', 'VEC4'}:
                    raise ValueError(f'{repr(name)}, Only arrays of 4-element types are supported due '
                                     'to the 16-byte alignment requirement.')

                alignment = {
                    4: 4,
                    8: 8,
                    12: 16,
                    16: 16,
                    64: 16,

                }.get(itemsize, 16)

                if alignment is None:
                    raise ValueError(f'Element')

                misalignment = struct_size % alignment

                if not misalignment == 0:
                    raise ValueError(f'{repr(name)} element misaligned for base aligment of {alignment} bytes. Off by'
                                     f' {misalignment} bytes, maybe insert {(alignment - misalignment) // 4} padding elements?')

                emit_struct_member(type=v._value, name=name, size=v._size)
                struct_size += elem_size

                add_dtype_elem(name, v._np_dt, v._size)
            else:
                raise ValueError('invalid UsageFlag annotation', v)

        for name, v in annots.items():
            if isinstance(v, UsageFlag) and v._usage == 'GLTYPE':
                if v._value == 'MAT3':
                    raise ValueError('MAT3 is currently unsupported by UBOStruct due to alignment issues')

                emit_element(name, v)

            elif isinstance(v, type) and issubclass(v, UBOStruct):
                v: UBOStruct = v
                add_dtype_elem(name, v._np_dt, v._size)
                struct_size += int(v._np_dt.itemsize) * max(v._size, 1)

                if v not in seen_substructs:
                    emit_struct_def(v.get_ubo_typedef())
                    seen_substructs.add(v)
                    seen_substructs |= v._substructs

                emit_struct_member(type=v.__name__, name=name, size=v._size)

            else:
                raise ValueError('UBOStructure annotations must all be declared using GLTYPE UsageFlags')

        if not struct_size % 16 == 0:
            raise ValueError(f'UBOstructure size must add up to a multiple of 16 bytes, currently {struct_size} bytes.'
                             f'Maybe insert {4 - (struct_size % 16) // 4} padding elements?')

        typedef_lines.append('};\n')

        guard_name = f'{cls.__name__}_STRUCT_DEF_'

        cls._compiled_typedef_string = '\n'.join(
            [
                f'#ifndef {guard_name}',
                f'#define {guard_name}',
                *typedef_lines,
                '#endif'
            ]
        )

        cls._np_dt = np.dtype(dtype_elems)
        cls._substructs = frozenset(seen_substructs)

        return super().__init_subclass__()

    @classmethod
    def get_ubo_typedef(cls):
        return cls._compiled_typedef_string

    def __new__(cls, **kwargs):
        return np.zeros(1, dtype=cls._np_dt).view(dtype=cls._np_dt, type=cls)

    def __setitem__(self, name, value):
        # Special case for matrix types:
        # it seems like they are stored in column major order in blender's UBOs
        # its annoying, so reorder for consistency
        self._dirty = True
        s = super().__getitem__(0)
        s[name] = value
        if name in self._matrix_elems:
            s[name] = s[name].T

    def __getitem__(self, name):
        return super().__getitem__(0)[name]

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<UBOStruct: {self.__class__.__name__} {str(super().__getitem__(0))}>'

    def __init__(self, **kwargs):
        super().__init__()
        self._dirty = False
        for k, v in kwargs.items():
            try:
                self[k] = v
            except KeyError:
                raise NameError(k, 'not defined')
        self._ubo = GPUUniformBuf(self)

    def set_values(self, **kwargs):
        s = super().__getitem__(0)

        for k, v in kwargs.items():
            if hasattr(v, '__len__'):
                s[k][:len(v)] = v
            else:
                s[k] = v

        self._dirty = True
        return self

    def force_update(self):
        self._ubo.update(self)
        self._dirty = False
        return self._ubo

    def get_ubo(self):
        if self._dirty:
            self._ubo.update(self)
            self._dirty = False
        return self._ubo


def ubo_struct(cls):
    '''
    decorator to convert a class to UBOStruct.
    This is merelly syntactic sugar, decorators read visually better sometimes.

    Much like blender, it doesnt check every possible alignment issue that might happen, but tries its best
    its the developer responsibility to follow the std140 packing rules.

    https://developer.blender.org/docs/handbook/guidelines/glsl/#packing-rules
    '''
    return type(cls.__name__, (cls, UBOStruct), {**cls.__dict__})


def ceil_pot(x):
    return 2 ** (ceil(x) - 1).bit_length()


def calc_pot_size(length):
    '''
    Finds the smallest power of two rectangle of at least 16x16 that fits a given length
    '''
    if length <= 0:
        return 16, 16

    w = ceil_pot(length ** 0.5)
    h = w // 2 if (w * w >= length * 2) else w
    return max(w, 16), max(h, 16)


class Texture:
    '''
    simple wrapper for GPUTexture, to support some custom syntax

    has .R and .W flags to be used by compute_inline_partial 
    for specifying readony and write only
    '''

    readonly: 'Texture'
    '''
    Proxy object interpreted as readonly by compute_inline_partial
    '''

    writeonly: 'Texture'
    '''
    Proxy object interpreted as writeonly by compute_inline_partial
    '''

    format: str
    '''
    Might not be the actual format of the texture. This is the format the data should actually be interpreted as.
    '''

    _inline_format_code: str
    '''
    flag used by compute_inline_partial to detect both the format and read/write qualifiers
    '''

    def __init__(self, size, format, data=None):
        if isinstance(size, int):
            size = calc_pot_size(size)

        if data is not None:
            if isinstance(data, Buffer):
                self._gpu_texture = GPUTexture(size, format=format, data=data)
            else:
                self.__dict__ = Texture.from_array(data, format=format, size=size).__dict__
                return

        else:
            self._gpu_texture = GPUTexture(size, format=format)

        self.format = self._gpu_texture.format
        self.size = size
        self.width, self.height = size
        self._inline_format_code = self.format

        self.readonly = Texture.__new__(Texture)
        self.writeonly = Texture.__new__(Texture)
        self.readonly.__dict__ = self.__dict__.copy()
        self.writeonly.__dict__ = self.__dict__.copy()
        self.readonly._inline_format_code = self.format + '_R'
        self.writeonly._inline_format_code = self.format + '_W'

    def read(self):
        return self._gpu_texture.read()

    def clear(self, format, value=(0, 0, 0, 1)):
        return self._gpu_texture.clear(format=format, value=value)

    @staticmethod
    @cache
    def _upload_buffer():
        class UploadBuffer(UBOStruct):
            data: UVEC4[16384 // (4 * 4)]
        return UploadBuffer()

    @classmethod
    @cache
    def _upload_compute_ubo(cls):
        # unorthodox, I know, but the regular texture uploading api, i.e. `data=buffer` refuses to give preddictable results
        # likely some kind of synchronization issue, or maybe I'm using it wrong.
        # I didn't dig through the source to know.
        # anyways, breaking up the data into chunks and uploading via uniform buffer seems to work, albeit slower.
        return compute_program(
            local_size=(64, 1, 1),
            shader_inputs={'tex': RGBA32UI_2D, 'ubo_data': type(cls._upload_buffer()),
                           'elem_size': INT,
                           'start': INT,
                           'len': INT},
            code='''//glsl
            void main(){
                uint idx = gl_GlobalInvocationID.x;
                if (idx >= len) return;
                uvec4 texel = uvec4(0);
                for (int i=0; i<elem_size; i++){
                    uint itemidx = idx * elem_size + i;
                    texel[i] = ubo_data.data[itemidx / 4][itemidx % 4];
                }
                store(tex, idx + start, texel);
            }
            '''
        )

    @staticmethod
    @cache
    def _read_compute():
        return compute_program(
            local_size=(8, 8, 1),
            shader_inputs={'inptex': RGBA32UI_2D, 'outtex': R32F_2D, 'n_channels': INT},
            code='''//glsl
            void main(){
                uvec2 outsize = uvec2(imageSize(outtex));
                uvec2 texcoord = gl_GlobalInvocationID.xy;

                if (texcoord.x >= outsize.x || texcoord.y >= outsize.y) return;
                
                uint idx = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * outsize.x;
                uvec4 texel = load(inptex, idx / n_channels);
                imageStore(outtex, ivec2(texcoord), uintBitsToFloat(uvec4(texel[idx % n_channels])));
            }
            '''
        )

    @staticmethod
    def _format_and_channels(format):
        if (format_match := re.match(r'(RGBA|RG|R)(32F|32UI|32I)$', format)) is None:
            raise ValueError('unsupported format', format)
        n_channels, data_format = format_match.groups()
        n_channels = ['', 'R', 'RG', '', 'RGBA'].index(n_channels)

        return n_channels, data_format

    @classmethod
    def from_array(cls, arr: np.ndarray, format, size=None, elem_size=None):
        # GPUTexture(..., data=buffer) was failing randomly for large textures
        # so a workaround was needed.
        # This implementation is using an UBO to upload data to a texture
        # seems to be more robust, but unfortunatelly also kinda slow.

        if not any(arr.dtype == d for d in (np.float32, np.uint32, np.int32)):
            raise ValueError('unsupported dtype: {arr.dtype}')

        n_channels, data_format = cls._format_and_channels(format)

        if elem_size is None:
            elem_size = n_channels
        elif elem_size > n_channels:
            raise ValueError(f'element size {elem_size} greater than number of channels {n_channels}')

        uarr = arr.view(np.uint32).reshape(-1)
        arr_size = len(uarr)
        tex_size = arr_size // elem_size

        if not arr_size % elem_size == 0:
            raise ValueError(f'length of array not divisible by element size {elem_size}')

        if size is None:
            size = calc_pot_size(tex_size)

        if tex_size > np.prod(size):
            raise ValueError(f'size {size} too small to fit array of size {len(uarr)}')

        tex = Texture(size, format=format)
        tex.clear('UINT', value=(0, 0, 0, 0))

        upload = cls._upload_compute_ubo()
        ubo = cls._upload_buffer()
        ubo_data = ubo['data'].reshape(-1)
        ubo_len = len(ubo_data)

        start = 0
        while start < len(uarr):
            arr_slice = uarr[start: start + ubo_len]
            elements = len(arr_slice) // elem_size

            ubo_data[:len(arr_slice)] = arr_slice
            ubo.force_update()

            upload(ceil(elements / 64), 1, 1,
                   tex=tex,
                   ubo_data=ubo,
                   elem_size=elem_size,
                   start=start // elem_size,
                   len=elements)

            start += elements * elem_size

        return tex

    def read_array(self, linear=True):
        # Trying to read any format other than a float Texture or R32UI returns garbage data
        # Since thats the case, lets read a float texture and perform bit-level reinterpretation

        formats = {'32F': ('FLOAT', np.float32),
                   '32I': ('INT', np.int32),
                   '32UI': ('UINT', np.uint32)}

        n_channels, data_format = self._format_and_channels(self.format)
        fmt, dt = formats[data_format]

        output_tex = Texture(size=calc_pot_size(np.prod(self.size) * n_channels), format='R32F')
        read_compute = self._read_compute()

        ow, oh = output_tex.size
        read_compute(ceil(ow / 8), ceil(oh * 2),
                     inptex=self, outtex=output_tex, n_channels=n_channels)

        data = output_tex.read()
        tot_len = int(np.prod(data.dimensions))
        data1 = Buffer(fmt, (tot_len,), data)  # little trick to avoid expensive data copies.

        if linear:
            rshape = (-1, n_channels) if n_channels > 1 else (-1,)
        else:
            rshape = (*self.size, n_channels) if n_channels > 1 else self.size

        return np.frombuffer(data1, dtype=dt)[:np.prod(self.size) * n_channels].reshape(*rshape)


def glsl_dedent(code):
    return dedent(re.sub('//glsl|//hlsl', '\n', code))


@dataclass
class ShaderBuilder:
    shader_inputs: dict | None = None
    defines: dict | None = None
    vertex_source: str = ''
    fragment_source: str = ''
    compute_source: str = ''
    typedef_source: str = ''

    include_source: str = ''
    include: list | tuple = ()
    local_size: list | tuple = (1, 1, 1)

    _shader: GPUShader = None
    _dispatch_kwargs_fn = None
    _shader_type = None

    def build(self):
        cr_info = self._cr_info = GPUShaderCreateInfo()
        vert_stage = self._vert_stage = GPUStageInterfaceInfo('vertex_interface')
        tex_slot = 0
        vin_slot = 0
        frag_out_slot = 0
        uniform_buffer_slot = 0
        vout_flag = False

        kwargs_inputs_mapping = {}

        ubo_typedef_strings = []
        seen_ubos = set()

        for name, input_type in self.shader_inputs.items() if self.shader_inputs else {}:
            if isinstance(input_type, UsageFlag):

                size = input_type.get_size()
                filled_in_usages = input_type.get_filledin_usages()

                usage_format = set(filled_in_usages.keys())

                # push constant / uniform
                if usage_format == {'GLTYPE'}:
                    if input_type == UINT:
                        # UINT refuses to work, likely unsupported
                        raise ValueError(f'{name}: unsupported type UINT for shaders, use INT instead')

                    cr_info.push_constant(filled_in_usages['GLTYPE'], name, size=size)
                    # first_letter = filled_in_usages['GLTYPE'][0]
                    # kwargs_inputs_mapping[name] = 'uniform_int' if first_letter in 'UIB' else 'uniform_float'
                    try:
                        kwargs_inputs_mapping[name] = {
                            BOOL: 'uniform_bool',
                            INT: 'uniform_int',
                            FLOAT: 'uniform_float',

                            VEC2: 'uniform_float',
                            VEC3: 'uniform_float',
                            VEC4: 'uniform_float',

                            IVEC2: 'uniform_int',
                            IVEC3: 'uniform_int',
                            IVEC4: 'uniform_int',

                            UVEC2: 'uniform_int',
                            UVEC3: 'uniform_int',
                            UVEC4: 'uniform_int',

                            MAT3: 'uniform_float',
                            MAT4: 'uniform_float',
                        }[filled_in_usages['GLTYPE']]

                    except KeyError:
                        raise ValueError(f'unrecogized push_constant {filled_in_usages['GLTYPE']}')

                # image variables
                elif len(usage_format & {'IMAGE_TYPE', 'IMAGE_FORMAT'}) == 2:
                    qualifiers = {filled_in_usages[k] for k in filled_in_usages if k.startswith('QUAL_')}
                    qualifiers = qualifiers or {'READ', 'WRITE'}

                    cr_info.image(tex_slot, filled_in_usages['IMAGE_FORMAT'], filled_in_usages['IMAGE_TYPE'], name,
                                  qualifiers=qualifiers)

                    kwargs_inputs_mapping[name] = 'image'
                    tex_slot += 1

                # Texture uniform
                elif usage_format == {'IMAGE_TYPE'}:
                    cr_info.sampler(tex_slot, filled_in_usages['IMAGE_TYPE'], name)
                    kwargs_inputs_mapping[name] = 'uniform_sampler'
                    tex_slot += 1

                # Vertex attribute
                elif usage_format == {'VIN', 'GLTYPE'}:
                    cr_info.vertex_in(vin_slot, filled_in_usages['GLTYPE'], name)
                    vin_slot += 1

                # Vertex output
                elif usage_format == {'VOUT', 'GLTYPE'} or\
                        usage_format == {'VOUT', 'GLTYPE', 'INTERPOLATION'} or\
                        usage_format == {'GLTYPE', 'INTERPOLATION'}:

                    interp_mode = filled_in_usages.get('INTERPOLATION', 'SMOOTH')

                    if interp_mode == 'FLAT':
                        vert_stage.flat(filled_in_usages['GLTYPE'], name)

                    elif interp_mode == 'SMOOTH':
                        vert_stage.smooth(filled_in_usages['GLTYPE'], name)

                    elif interp_mode == 'NO_PERSPECTIVE':
                        vert_stage.no_perspective(filled_in_usages['GLTYPE'], name)

                    else:
                        raise ValueError('unrecognized interpolation mode')

                    vout_flag = True

                # fragment output
                elif usage_format == {'GLTYPE', 'FRAG_OUT'} or\
                        usage_format == {'GLTYPE', 'FRAG_OUT', 'BLEND_MODE'}:
                    cr_info.fragment_out(frag_out_slot, filled_in_usages['GLTYPE'], name,
                                         blend=filled_in_usages.get('BLEND_MODE', 'NONE'))

                    frag_out_slot += 1

                # maybe reactivate later, not useful for now.
                # elif usage_format == {'UNIFORM_BUFFER', 'STRING'}:
                #     cr_info.uniform_buf(uniform_buffer_slot, filled_in_usages['STRING'], name)
                #     kwargs_inputs_mapping[name] = 'uniform_block'
                #     uniform_buffer_slot += 1

                else:
                    raise ValueError(f'unknown usage format {input_type.to_string(usage=True)} for {input_type}')

            # Probably UBOStruct, it quacks with the same accent
            elif hasattr(input_type, 'get_ubo_typedef'):
                if input_type not in seen_ubos:
                    seen_ubos.add(input_type)
                    ubo_typedef_strings.append(input_type.get_ubo_typedef())

                cr_info.uniform_buf(uniform_buffer_slot, input_type.__name__, name)
                kwargs_inputs_mapping[name] = 'uniform_block'
                uniform_buffer_slot += 1

            else:
                raise ValueError(f'invalid shader input declaration: "{name}": {input_type}')

        if not isinstance(self.include, (list, tuple)):
            raise ValueError('include must be list or tuple')

        _merged_include = _get_include((*_default_include, *self.include), self.include_source)

        if self.defines:
            for name, value in self.defines.items():
                cr_info.define(name, str(value))

        if self.compute_source:
            header = '# define COMP_SHADER 1\n'
            code = '\n\n'.join((header, _merged_include + self.compute_source))
            cr_info.compute_source(code)

            if isinstance(self.local_size, int):
                local_size = (self.local_size,)
            local_size = (*self.local_size, 1, 1, 1)[:3]
            cr_info.local_group_size(*local_size)

            lx, ly, lz = local_size
            cr_info.define('COMPUTE_TOTAL_INVOCATIONS_', str(lx * ly * lz))
            cr_info.define('COMPUTE_X_INVOCATIONS_', str(lx))
            cr_info.define('COMPUTE_Y_INVOCATIONS_', str(ly))
            cr_info.define('COMPUTE_Z_INVOCATIONS_', str(lz))

            self._shader_type = 'COMPUTE'

        else:
            self._shader_type = 'PIPELINE'

        if self.vertex_source:
            header = '# define VERT_SHADER 1\n'
            code = '\n\n'.join((header, _merged_include, glsl_dedent(self.vertex_source)))
            cr_info.vertex_source(code)

        if self.fragment_source:
            header = '# define FRAG_SHADER 1\n'
            code = '\n\n'.join((header, _merged_include, glsl_dedent(self.fragment_source)))
            cr_info.fragment_source(code)

        if self.typedef_source or ubo_typedef_strings:
            code = '\n\n'.join((self.typedef_source, *ubo_typedef_strings))
            cr_info.typedef_source(code)

        if vout_flag and vin_slot > 0:
            cr_info.vertex_out(vert_stage)

        shader = self._shader = gpu.shader.create_from_info(cr_info)

        self._dispatch_kwargs_fn = kwargs_inputs_mapping

        params_set = self._make_params_setter(shader, kwargs_inputs_mapping)
        self.params_set = params_set

        return shader, params_set, self._shader_type

    def _make_params_setter(self, shader, kwargs_inputs_mapping):

        setter_functions = {}

        def ubo_setter(k, v):
            if hasattr(v, 'get_ubo'):
                return shader.uniform_block(k, v.get_ubo())
            else:
                return shader.uniform_block(k, v)

        def sampler_setter(k, v):
            return shader.uniform_sampler(k, getattr(v, '_gpu_texture', v))

        def image_setter(k, v):
            return shader.image(k, getattr(v, '_gpu_texture', v))

        def dummy_setter(k, v):
            '''sometimes a shader optmizes out a uniform, and its hella annoying. pretend it still exists'''
            pass

        for k, fn_name in kwargs_inputs_mapping.items():
            match fn_name:
                case 'uniform_block':
                    setter_functions[k] = ubo_setter

                case 'uniform_sampler':
                    setter_functions[k] = sampler_setter

                case 'image':
                    setter_functions[k] = image_setter

                case 'uniform_int' | 'uniform_float' | 'uniform_bool':
                    setter_functions[k] = getattr(shader, fn_name)

                    try:
                        setter_functions[k](k, 0)

                    except ValueError:
                        # uniform not present, likely optimzed out.
                        setter_functions[k] = dummy_setter

                case _:
                    setter_functions[k] = getattr(shader, fn_name)

        def params_set(kwargs):
            kwf = setter_functions
            try:
                for k, v in kwargs.items():
                    kwf[k](k, v)
            except KeyError:
                raise NameError('unexpected keyword argumment', k)

        return params_set


def _make_shader_fn(code_strings={}, local_size=(1, 1, 1), include=(), include_source='', typedef_source='', shader_inputs={}, defines={}):
    compute_code = code_strings.get('compute', None)
    vert_code = code_strings.get('vertex', None)
    frag_code = code_strings.get('fragment', None)

    shader_builder = ShaderBuilder(compute_source=compute_code,
                                   vertex_source=vert_code,
                                   fragment_source=frag_code,
                                   shader_inputs=shader_inputs,
                                   defines=defines,
                                   include=[*include],
                                   include_source=include_source,
                                   typedef_source=typedef_source,
                                   local_size=local_size)

    shader, params_set, _ = shader_builder.build()

    return shader, params_set


def compute_program(*, code=None, local_size=None, include=(), include_source='', typedef_source='', shader_inputs={}, defines={}):
    if not code:
        raise ValueError('shader code has not been provided')

    if not local_size:
        raise ValueError('local_size parameter required')

    shader, params_set = _make_shader_fn(code_strings={'compute': code}, local_size=local_size, include=include,
                                         include_source=include_source, typedef_source=typedef_source,
                                         shader_inputs=shader_inputs,
                                         defines=defines)

    def shader_fn(*num_workgroups, **kwargs):
        shader.bind()
        params_set(kwargs)
        gpu.compute.dispatch(shader, *(*num_workgroups, 1, 1)[:3])

    shader_fn.shader = shader
    return shader_fn


def shader_program(*, vert_code=None, frag_code=None, include=(), include_source='', typedef_source='', shader_inputs={}, defines={}):
    if not vert_code and frag_code:
        raise ValueError('shader code has not been provided')

    shader, params_set = _make_shader_fn(code_strings={'vertex': vert_code, 'fragment': frag_code}, local_size=None, include=include,
                                         include_source=include_source, typedef_source=typedef_source,
                                         shader_inputs=shader_inputs,
                                         defines=defines)

    def shader_fn(batch, **kwargs):
        shader.bind()
        params_set(kwargs)
        batch.draw(shader)

    def instanced(batch, instance_start=0, instance_count=0, **kwargs):
        shader.bind()
        params_set(kwargs)
        batch.draw_instanced(shader,
                             instance_start=instance_start,
                             instance_count=instance_count)

    shader_fn.shader = shader
    shader_fn.instanced = instanced
    shader_fn.params_set = params_set

    return shader_fn


@wraps(compute_program)
def compute_inline_partial(line_no_cache=False, **kw_comp_params):
    '''
    ## example:
    tex = GPUTexture((10, 10), format='RGBA32F')
    inline_compute = compute_partial_inline(local_size=(8, 8, 1))
    inline_compute('imageStore(tex, ivec2(gl_GlobalInvocationID.xy), vec4(value));',
                10, 10, 1, tex=tex, value=(1, 2, 3, 4))
    '''
    # kw_comp_params = deepcopy(kw_comp_params)
    # if 'local_size' not in kw_comp_params:
    #     raise ValueError('local_size parameter required')

    cache = {}

    int_types = (int, np.integer)
    list_types = (list, tuple, np.ndarray)
    number_types = (int, float, bool, np.integer, np.floating, np.bool)

    texture_formats = _2D_IMAGE_FORMATS.copy()

    for key, value in list(texture_formats.items()):
        texture_formats[key + '_R'] = value | READ
        texture_formats[key + '_W'] = value | WRITE

    def arg_type(argname, argval):

        if isinstance(argval, number_types):
            if isinstance(argval, (bool, np.bool)):
                return (argname, BOOL)

            elif isinstance(argval, int_types):
                return (argname, INT)

            return (argname, FLOAT)

        if isinstance(argval, (GPUTexture, Texture)):
            if argval.format not in texture_formats:
                raise ValueError('unsupported texture')
            return (argname, texture_formats[getattr(argval, '_inline_format_code', argval.format)])

        if isinstance(argval, UBOStruct):
            return argname, type(argval)

        if isinstance(argval, list_types):
            l = len(argval)
            if not l in (2, 3, 4):
                raise ValueError(f'ivalid vector size for argument {argname} {l}')

            vals = list(argval)
            is_ingeger_type = all(isinstance(x, int_types) for x in vals)

            return (argname, ((VEC2, VEC3, VEC4), (IVEC2, IVEC3, IVEC4))[is_ingeger_type][l - 2])

        raise ValueError(f'"{argname}": invalid argument type {type(argval)}')

    def compute_inline(*num_invocations, code=None, wrap_main=True, include=(), include_source='', typedef_source='', defines={}, line_no_cache=line_no_cache, local_size=None, **kwargs):
        local_size = local_size or kw_comp_params.get('local_size', ())
        local_size = (*local_size, 1, 1, 1)[:3]
        local_x, local_y, local_z = local_size

        if not code:
            raise ValueError('code argument not provided')

        signature = None
        cache_key = None

        if line_no_cache:
            frame = sys._getframe(1)
            cache_key = (frame.f_lineno, frame.f_code.co_filename, local_size, tuple(include),
                         include_source, typedef_source, local_size, tuple(defines.items()))

        else:
            signature = frozenset((arg_type(*item) for item in kwargs.items()))
            cache_key = (local_size, tuple(include), signature,
                         include_source, typedef_source, local_size, tuple(defines.items()))

        if cache_key not in cache:
            if not signature:
                signature = frozenset((arg_type(*item) for item in kwargs.items()))

            if wrap_main:
                code = dedent('''
                    void main(){
                        if (gl_GlobalInvocationID.x >= num_invocations.x || 
                            gl_GlobalInvocationID.y >= num_invocations.y ||
                            gl_GlobalInvocationID.z >= num_invocations.z) return;
                        CODE
                    }
                ''').replace('CODE', indent(glsl_dedent(code), ' ' * 8))

            else:
                code = dedent('''
                bool boundscheck(){
                    return (gl_GlobalInvocationID.x < num_invocations.x && 
                            gl_GlobalInvocationID.y < num_invocations.y &&
                            gl_GlobalInvocationID.z < num_invocations.z);
                }
                              
                CODE
                ''').replace('CODE', glsl_dedent(code))

            params = kw_comp_params.copy()

            extra_inputs = dict(signature)
            extra_inputs['num_invocations'] = IVEC3
            extra_inputs.update(kw_comp_params.get('shader_inputs', {}))

            params['shader_inputs'] = extra_inputs
            params['include'] = (*params.get('include', ()), *include)
            params['include_source'] = params.get('include_source', include_source)
            params['typedef_source'] = params.get('typedef_source', typedef_source)
            params['defines'] = {**params.get('defines', {}), **defines}
            params['local_size'] = local_size
            params['code'] = code

            program = compute_program(**params)

            cache[cache_key] = program

        else:
            program = cache[cache_key]

        inv_x, inv_y, inv_z = (*num_invocations, 1, 1, 1)[:3]
        return program(
            ceil(inv_x / local_x),
            ceil(inv_y / local_y),
            ceil(inv_z / local_z),
            **kwargs,
            num_invocations=(inv_x, inv_y, inv_z))

    compute_inline._cache = cache  # maybe useful, who knows

    return compute_inline


__all__ = ['ShaderBuilder',
           'UsageFlag',
           'UBOStruct',
           'ubo_struct',
           'Texture',
           'compute_program',
           'compute_inline_partial',
           'shader_program',
           'calc_pot_size',
           'ceil_pot',
           'glsl_dedent',
           *_ALL_USAGE_FLAGS]  # pyright: ignore
