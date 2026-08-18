import bpy
import numpy as np
import gpu

from . common import apptimer
from . shad.core import Texture, compute_inline_partial, R32I, R32F, R32UI, UVEC4, glsl_dedent, ubo_struct
from . shad.glsl import register_include
from math import ceil
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterable

LAYER_LIB = '''//glsl

#ifndef LAYER_ACC

#define LAYER_ACC 0
#define LAYER_IDX 1
#define LAYER_MASK 2
#define LAYER_GAIN 3
#define LAYER_BIAS 4
#define LAYER_ACTIV 5
#define LAYER_NROWS 8

#define LAYER_GET_DATA(layer_data, row, idx) imageLoad(layer_data, ivec2(idx, row)).x
#define LAYER_STR_DATA(layer_data, row, idx, data) imageStore(layer_data, ivec2(idx, row), ivec4(data, 0, 0, 0))
#define LAYER_GET_BIT(layer_data, idx) ((uint(LAYER_GET_DATA(layer_data, LAYER_MASK, (idx) / 32)) >> uint((idx) % 32)) & 1u)
#define LAYER_SET_BIT(layer_data, idx) (imageAtomicOr(layer_data, ivec2(idx / 32, LAYER_MASK), 1 << (idx % 32)))

// the mask row is packed and leaves some empty space that can be better utilized
// assumes the layer is at least 2 neurons wide.
#define LAYER_COUNTER_ID(layer_data) ivec2(imageSize(layer_data).x - 1u, LAYER_MASK)
#define LAYER_COUNTER_INC(layer_data, val) imageAtomicAdd(layer_data, LAYER_COUNTER_ID(layer_data), val)
#define LAYER_COUNTER_GET(layer_data) imageLoad(layer_data, LAYER_COUNTER_ID(layer_data)).x
#define LAYER_COUNTER_SET(layer_data, val) imageStore(layer_data, LAYER_COUNTER_ID(layer_data), ivec4(val, 0, 0, 0));

#define INDEX_INACTIVE -1

// fixed point

struct NeuronActivity{
    float acc;
    float bias;
    float gain;
    float activation;
    float gain_activation;
};

NeuronActivity neuron_calc(float acc, float bias, float gain){
    NeuronActivity r;
    r.acc = acc;
    r.bias = bias;
    r.gain = gain;
    r.activation = acc + bias;
    r.gain_activation = r.activation * gain;
    return r;
}

#define LAYER_ACTIVITY_CALC(layer_data, bias_data, idx) neuron_calc(\\
        AS_FP(LAYER_GET_DATA(layer_data, LAYER_ACC, idx)),\\
        intBitsToFloat(LAYER_GET_DATA(bias_data, LAYER_BIAS, idx)),\\
        intBitsToFloat(LAYER_GET_DATA(bias_data, LAYER_GAIN, idx)))

#define LAYER_CALC_VALUE_BIAS(layer_data, bias_data, idx) (         \\
    (AS_FP(LAYER_GET_DATA(layer_data, LAYER_ACC, idx)) +           \\
    intBitsToFloat(LAYER_GET_DATA(bias_data, LAYER_BIAS, idx))) *  \\
    intBitsToFloat(LAYER_GET_DATA(bias_data, LAYER_GAIN, idx)))


#define FXQ 11
#define FX_ONE (1 << FXQ)
#define AS_FX(x) int((x) * float(FX_ONE))
#define AS_FP(x) (float(x) * (1.0 / FX_ONE))

int mul_fx(int a, int b){
    //Limited for multiplication of values <= 1.0
    int r = a * b;
    r += FX_ONE >> 1;
    return r >> FXQ;
}

int mul_fxfp(int a, float f){
    return int(float(a) * float(f));
}

#endif
'''

FXQ = 11
FX_ONE = (1 << FXQ)


def as_fp(x): return x / FX_ONE
def as_fx(x): return np.asarray(x * FX_ONE, dtype=np.uint32)


register_include('layer', LAYER_LIB)

LAYER_OFFSETS = dict(
    acc=0,
    idx=1,
    mask=2,
    gain=3,
    bias=4,
    activ=5,
)
LAYER_NROWS = 8


compute_inline = compute_inline_partial(local_size=(128, 1, 1), include=['random_lib', 'inf', 'layer'], line_no_cache=True)


def _bits2tex(data):
    packed = np.packbits(data, bitorder='little')
    packed.resize(ceil(len(packed) / 4) * 4)
    packed = packed.view(np.uint32)
    return Texture.from_array(packed, format=R32UI)


@dataclass
class OuterProdExpr:
    '''
    Dataclass for syntatic sugar on matrix methods
    '''
    items: Iterable
    samples: int | None = None

    def __rmul__(self, other: int | float):
        return self * other

    def __mul__(self, other: int | float):
        return OuterProdExpr([(a, b, f * other) for a, b, f in self.items], self.samples)

    def __add__(self, other: 'OuterProdExpr'):
        return OuterProdExpr([*self.items, *other.items], self.samples)

    def __sub__(self, other: 'OuterProdExpr'):
        return OuterProdExpr([*self.items, *((a, b, -f) for a, b, f in other.items)], self.samples)

    def __neg__(self):
        return self * -1

    def __getitem__(self, samples):
        return OuterProdExpr(self.items, samples)


@dataclass
class PendingProjectionBase:
    synapses: 'Synapses'
    layer_a: 'Layer'
    side: str


class PendingRightProjection(PendingProjectionBase):
    def __rshift__(self, layer_b: 'Layer'):
        return self.synapses.aproj(self.layer_a, layer_b, 1.0)


class PendingLeftProjection(PendingProjectionBase):
    def __lshift__(self, layer_b: 'Layer'):
        return self.synapses.bproj(self.layer_a, layer_b, 1.0)


@ubo_struct
class UploadUbo:
    # 16kb // 4 bytes // vec4
    data: UVEC4[16384 // (4 * 4)]


_upload_ubos = [UploadUbo() for _ in range(4)]


def prepare_upload_ubos(arr):
    if not isinstance(arr, np.ndarray) or not arr.dtype == np.uint32:
        raise ValueError('array must be a uint32 ndarray')

    l_arr = len(arr)
    if l_arr > 4096 * len(_upload_ubos):
        raise ValueError('array too big')

    for i in range(len(_upload_ubos)):
        start = i * 4096
        end = (i + 1) * 4096
        if start >= l_arr:
            break

        slice = arr[start:end]

        _upload_ubos[i]['data'].reshape(-1)[:len(slice)] = slice
        _upload_ubos[i].force_update()

    return {
        f'upload_ubo{i}': _upload_ubos[i]
        for i in range(len(_upload_ubos))
    }


UPLOAD_LIB = '''//glsl
#ifndef UPLOAD_LIB
#define UPLOAD_LIB

#define UPLOAD_SIZE 4096

uint get_upload_item(uint idx) {
    uint v = idx >> 2;
    uint e = idx & 3u;
    uint chunk_size = UPLOAD_SIZE / 4u;
    
    switch (idx / UPLOAD_SIZE) {
        case 0: return upload_ubo0.data[v][e];
        case 1: return upload_ubo1.data[v - chunk_size][e];
        case 2: return upload_ubo2.data[v - chunk_size * 2][e];
        case 3: return upload_ubo3.data[v - chunk_size * 3][e];
        default: return 0;
    }
}

#endif
'''

register_include('upload_ubo', UPLOAD_LIB)


class Layer:
    '''
    Layer is a multiplexed stack, each row contain data about the neuron's state.
    offsets:
        0: int = matrix accumulator
        1: int = active indices
        2: uint = bitmask
        3: floatBitsToInt = gain constant
        4: floatBitsToInt = bias constant
        5: floatBitsToInt = activity value constant
    '''

    _tmp_buffer_tex = Texture((128, 1), format=R32I)

    @classmethod
    def _tmp_buffer(cls, size):
        if cls._tmp_buffer_tex.width > size:
            cls._tmp_buffer_tex = Texture((size, 1), format=R32I)
        return cls._tmp_buffer_tex

    def __init__(self, size, data=None, seed=1):
        self.size = size

        # texture size must be at least 2-wide to fit counter in the mask row
        if size < 2:
            raise ValueError('size must be at least 2')
        self.data = Texture((size, LAYER_NROWS), format='R32I', data=data)

        # replacecd with empty slot in data texture
        # self.counter = Texture((1, 1), format=R32I)
        # self.counter.clear('INT', (0,))

        self.seed = seed

        # assumed to always work when active_estimate == size
        # but works optimally if its set to the number of non-negative active indices
        # active indices must always be packed.
        self.active_estimate = self.size

        if data is None:
            self.clear(True)

    def dump(self):
        return self.data.read_array().reshape(-1, self.size)

    def load(self, buff):
        target_size = self.size * LAYER_NROWS

        buff_arr = np.frombuffer(buff, dtype=np.int32)

        if len(buff_arr) < target_size:
            raise ValueError(f'data too small: expected {target_size} elements, got {len(buff_arr)}')

        buff_arr = buff_arr[:target_size]
        self.data = Texture((self.size, LAYER_NROWS), format='R32I', data=buff_arr)

    def read(self):
        data = self.dump()
        output_data = {
            key: data[LAYER_OFFSETS[key]] for key in LAYER_OFFSETS
        }
        output_data['mask'] = data[LAYER_OFFSETS['mask']][:ceil(self.size / 32)]
        output_data['counter'] = data[LAYER_OFFSETS['mask']][-1]
        output_data['activ'] = output_data['activ'].view('f4')
        output_data['gain'] = output_data['gain'].view('f4')
        output_data['bias'] = output_data['bias'].view('f4')
        output_data['acc'] = output_data['acc'] / FX_ONE
        return output_data

    def pretty_bits(self, header=None, w=None):
        dot_width = w or ceil(self.size ** 0.5)
        char_width = ceil(dot_width / 2)
        char_height = ceil((self.size / dot_width) / 4)

        result_tex = Texture((char_width * char_height + 1, 1), format=R32UI)
        result_tex.clear('UINT', (0x2800,))

        header = 'Layer' if header is None else header

        compute_inline(
            self.active_estimate,
            result=result_tex,
            data=self.data,
            U_char_width=char_width,
            U_dot_width=dot_width,
            code='''//glsl
                if(gl_GlobalInvocationID.x == 0){
                    imageStore(result, 
                               ivec2(imageSize(result).x - 1, 0),
                               ivec4(LAYER_COUNTER_GET(data)));
                }
            
                int id = int(gl_GlobalInvocationID).x;
                int idx = LAYER_GET_DATA(data, LAYER_IDX, id);
                if (idx == INDEX_INACTIVE) return;

                ivec2 bit_coord = ivec2(idx % U_dot_width, idx / U_dot_width);
                ivec2 character_coord = bit_coord / ivec2(2, 4);
                int bit_offset = (bit_coord.x % 2) + 2 * (bit_coord.y % 4);

                int offset_map[8] = {
                    0, 3,
                    1, 4,
                    2, 5,
                    6, 7,
                };
                imageAtomicOr(result,
                            ivec2(character_coord.x + character_coord.y * U_char_width, 0),
                            1u << offset_map[bit_offset]);

            '''
        )

        result = np.frombuffer(result_tex.read(), dtype=np.uint32).reshape(-1)
        characters = result.tobytes().decode('utf-32-le')

        ANSI_GREEN = '\x1b[32m'
        ANSI_RESET = '\x1b[0m'

        active_count = result[-1]

        header = f'┌{header}({self.size}) | {active_count}\n'
        frame = ''.join(
            [
                header,
                '├', '─' * char_width, '┐\n',

                *(''.join(('│', ANSI_GREEN, characters[y * char_width:(y + 1) * char_width], ANSI_RESET, '│\n'))
                    for y in range(char_height)),

                '└', '─' * char_width, '┘'
            ]
        )
        return frame

    def pretty_pixels(self, w=None):
        w = w or ceil(self.size ** 0.5)
        n_lines = ceil(ceil(self.size / w) / 2)

        fg_code = '\x1b[38;2;000;000;000m'
        bg_code = '\x1b[48;2;000;000;000m'
        ansi_reset_endl = '\x1b[0m\n'
        block = chr(0x2580)

        n_blocks = w * n_lines * 2

        line_characters = (len(fg_code) + len(bg_code) + len(block)) * w + len(ansi_reset_endl)
        result_tex = Texture((line_characters, n_lines), format=R32F)
        compute_inline(
            w, n_lines,
            tex=result_tex,
            data=self.data,
            U_width=int(w),
            wrap_main=False,
            code='''//glsl
            
            uvec3 float_to_8bitdecimal(float x){
                x = clamp(x, 0.0, 1.0);
                uint ux = uint(round(x * 255.0));
                uvec3 ret;
                ret[0] = ux % 10u;
                ux /= 10u;
                ret[1] = ux % 10u;
                ux /= 10u;
                ret[2] = ux % 10u;
                return ret;
            }

            uint bg_code[19];
            uint fg_code[19];

            void rgb_set_bg(float val, int component){
                uvec3 digits = float_to_8bitdecimal(val);
                int offset = 7 + component * 4;
                bg_code[offset + 0] = 0x30u + digits.z;
                bg_code[offset + 1] = 0x30u + digits.y;
                bg_code[offset + 2] = 0x30u + digits.x;
            }
            void rgb_set_fg(float val, int component){
                uvec3 digits = float_to_8bitdecimal(val);
                int offset = 7 + component * 4;
                fg_code[offset + 0] = 0x30u + digits.z;
                fg_code[offset + 1] = 0x30u + digits.y;
                fg_code[offset + 2] = 0x30u + digits.x;
            }

            vec3 color_gradient(float val){
                float pval = clamp(val, 0.0, 1.0);
                float nval = clamp(-val, 0.0, 1.0);

                vec3 ncol = mix(
                    vec3(1.0, 0.0, 0.0),
                    vec3(1.0, 1.0, 0.0),
                    nval
                ) * pow(nval, 0.7);
                
                vec3 pcol = vec3(pval);

                return nval > pval? ncol : pcol;
                
            }

            void main(){
                // '\x1b[48;2;000;000;000m'
                bg_code = uint[19](27, 91, 52, 56, 59, 50, 59, 48, 48, 48, 59, 48, 48, 48, 59, 48, 48, 48, 109);
                // '\x1b[38;2;000;000;000m'
                fg_code = uint[19](27, 91, 51, 56, 59, 50, 59, 48, 48, 48, 59, 48, 48, 48, 59, 48, 48, 48, 109);
                //              '\x1b   [   4   8   ;   2   ;   0   0   0   ;   0   0   0   ;   0   0   0   m'

                uint reset_code[5] = {27, 91, 48, 109, 10};
                ivec2 cell = ivec2(gl_GlobalInvocationID.xy);

                int top = cell.x + cell.y * 2 * U_width;
                int bottom = cell.x + (cell.y * 2 + 1) * U_width;

                float top_val = intBitsToFloat(LAYER_GET_DATA(data, LAYER_ACTIV, top));
                float bottom_val = intBitsToFloat(LAYER_GET_DATA(data, LAYER_ACTIV, bottom));

                vec3 rgb;
                rgb = color_gradient(top_val);
                rgb_set_fg(rgb.r, 0);
                rgb_set_fg(rgb.g, 1);
                rgb_set_fg(rgb.b, 2);
                
                rgb = color_gradient(bottom_val);
                rgb_set_bg(rgb.r, 0);
                rgb_set_bg(rgb.g, 1);
                rgb_set_bg(rgb.b, 2);


                int code_offset_start = cell.x * (2 * 19 + 1);
                for (int i=0; i<19; i++){
                    imageStore(tex, ivec2(code_offset_start + i, cell.y), vec4(uintBitsToFloat(fg_code[i])));
                    imageStore(tex, ivec2(code_offset_start + 19 + i, cell.y), vec4(uintBitsToFloat(bg_code[i])));
                }
                imageStore(tex, ivec2(code_offset_start + 19 * 2, cell.y), vec4(uintBitsToFloat(0x2580u)));
                if (cell.x == 0){
                    code_offset_start = U_width * (2 * 19 + 1);
                    for (int i=0; i<5; i++){
                        imageStore(tex, ivec2(code_offset_start + i, cell.y), vec4(uintBitsToFloat(reset_code[i])));
                    }
                }
            }
            '''
        )

        result = np.frombuffer(result_tex.read(), dtype=np.uint32).reshape(-1)
        characters = result.tobytes().decode('utf-32-le')
        return characters


    def clear(self, all=False, *, bitmask=False, indices=False, accum=False, gain=False, bias=False, activation=False, counter=None):
        all = bool(all)

        compute_inline(
            self.size,
            data=self.data,
            U_accum=bool(accum) or all,
            U_indices=bool(indices) or all,
            U_bitmask=bool(bitmask) or all,
            U_gain=bool(gain) or all,
            U_bias=bool(bias) or all,
            U_activation=bool(activation) or all,
            U_counter=bool(counter is not None) or all,
            U_counter_val=int(counter or 0),

            code='''//glsl
                int idx = int(gl_GlobalInvocationID.x);
                if (U_accum) LAYER_STR_DATA(data, LAYER_ACC, idx, 0);
                if (U_indices) LAYER_STR_DATA(data, LAYER_IDX, idx, -1);
                if (U_bitmask && idx != LAYER_COUNTER_ID(data).x) LAYER_STR_DATA(data, LAYER_MASK, idx, 0);
                if (U_gain) LAYER_STR_DATA(data, LAYER_GAIN, idx, floatBitsToInt(1.0));
                if (U_bias) LAYER_STR_DATA(data, LAYER_BIAS, idx, floatBitsToInt(0.0));
                if (U_activation) LAYER_STR_DATA(data, LAYER_ACTIV, idx, floatBitsToInt(0.0));
                if (U_counter && idx == 0) LAYER_COUNTER_SET(data, U_counter_val);
            '''
        )
        return self

    def flatten(self, value=1.0):
        compute_inline(
            self.size,
            data=self.data,
            U_value=float(value),
            code='''//glsl
                uint idx = gl_GlobalInvocationID.x;
                uint bit = LAYER_GET_BIT(data, idx);
                float val = bool(bit)? U_value : 0;

                LAYER_STR_DATA(data, LAYER_ACC, idx, AS_FX(val));
                LAYER_STR_DATA(data, LAYER_ACTIV, idx, floatBitsToInt(val));
            '''
        )
        return self

    def set_acc_range(self, start=None, stop=None, step=None, *, val=1):
        start = start or 0
        stop = stop or self.size
        step = step or 1

        count = len(range(start, stop, step))
        compute_inline(
            count,
            data=self.data,
            U_start=start,
            U_step=step,
            U_val=float(val),
            code='''//glsl
                int idx = int(U_start + gl_GlobalInvocationID.x * U_step);
                if (idx >= imageSize(data).x) return;

                LAYER_STR_DATA(data, LAYER_ACC, idx, AS_FX(U_val));
                imageAtomicOr(data, ivec2(idx / 32, LAYER_MASK), 1 << (idx % 32));
            '''
        )
        return self

    def set_activ(self, activ_buff: np.ndarray):
        if len(activ_buff) != self.size:
            raise ValueError(f'activ_buff: size {len(activ_buff)} does not match layer of size {self.size}')

        if not isinstance(activ_buff, np.ndarray) or activ_buff.dtype != np.float32:
            activ_buff = np.asarray(activ_buff, dtype=np.float32)

        self.clear(counter=0)

        compute_inline(
            self.size,
            **prepare_upload_ubos(activ_buff.view(np.uint32)),
            layer_data=self.data,
            include=['upload_ubo'],

            code=LAYER_LIB + '''//glsl
                uint idx = int(gl_GlobalInvocationID.x);

                uint raw_val = get_upload_item(idx);
                float activ_val = uintBitsToFloat(raw_val);

                LAYER_STR_DATA(layer_data, LAYER_ACTIV, idx, raw_val);
                LAYER_STR_DATA(layer_data, LAYER_ACC, idx, AS_FX(activ_val));
                
                if (activ_val != 0.0) {
                    LAYER_SET_BIT(layer_data, idx);
                    LAYER_COUNTER_INC(layer_data, 1);
                }
            '''
        )

    def threshold(self, value, use_bias=False, block_normalize=None, flatten=1):
        self.clear(bitmask=True, indices=True, activation=True, counter=0)

        bias_data = self.data if not isinstance(use_bias, Layer) else use_bias.data
        if block_normalize is not None:
            size = self.size
            local_size = (block_normalize, 1, 1)
        else:
            size = self.size
            local_size = (32, 1, 1)

        compute_inline(
            size,
            local_size=local_size,
            U_value=float(value),
            U_flatten=float(flatten),
            data=self.data,
            bias_data=bias_data,
            U_use_bias=bool(use_bias),
            U_block_normalize=int(block_normalize or 0),
            include=['parallel_reduction'],
            wrap_main=False,
            code='''//glsl
                void main(){
                    int idx = int(gl_GlobalInvocationID.x);
                    float computed_threshold = 0;
                    float rescale = 1.0;

                    NeuronActivity activity = NeuronActivity(0, 0, 0, 0, 0);
                    if (boundscheck()){
                        activity = LAYER_ACTIVITY_CALC(data, bias_data, idx);
                    }

                    if (U_block_normalize > 0){
                        float total = parallel_add(max(0.0, U_use_bias ? activity.activation : activity.acc));

                        int block_start = (idx / U_block_normalize) * U_block_normalize;
                        float n_elems = min(num_invocations.x - block_start, U_block_normalize);
                        total /= n_elems;

                        computed_threshold = total * U_value;
                        rescale = 1.0 / (total == 0 ? 1.0 : total - computed_threshold);
                    }
                    else {
                        computed_threshold = U_value;
                    }

                    float final_val = U_use_bias ? activity.activation : activity.acc;

                    if (final_val <= computed_threshold) return;

                    if (U_flatten != 0.0){
                        final_val = U_flatten;
                    }else{
                        final_val -= computed_threshold;
                        final_val *= rescale;
                    }

                    if (boundscheck()){
                        int next_idx = LAYER_COUNTER_INC(data, 1);
                        LAYER_STR_DATA(data, LAYER_IDX, next_idx, int(idx));
                        LAYER_STR_DATA(data, LAYER_ACTIV, idx, floatBitsToInt(final_val));
                        LAYER_SET_BIT(data, idx);
                    }
                }
            '''
        )

        self.active_estimate = self.size
        return self

    def wta(self, block_size, use_bias=False, flatten=1.0, signed=False):

        self.active_estimate = int(self.size // block_size)
        self.clear(bitmask=True, indices=True, activation=True, counter=self.active_estimate)
        bias_data = self.data if not isinstance(use_bias, Layer) else use_bias.data

        self.seed += 1

        compute_inline(
            self.size,
            local_size=(block_size, 1, 1),
            data=self.data,
            bias_data=bias_data,
            U_use_bias=bool(use_bias),
            U_flatten=float(flatten),
            U_signed=bool(signed),
            U_seed=self.seed,
            
            include=['parallel_reduction'],
            wrap_main=False,
            code='''//glsl

                void main(){
                    uint idx = gl_GlobalInvocationID.x;
                    uint group_idx = gl_WorkGroupID.x;
                    NeuronActivity activity = NeuronActivity(0, 0, 0, 0, 0);

                    if (boundscheck()){
                        activity = LAYER_ACTIVITY_CALC(data, bias_data, idx);
                    }

                    float competition_key = (U_use_bias ? activity.gain_activation : activity.activation);

                    if (U_signed){
                        competition_key = abs(competition_key);
                    }

                    float winner_val = parallel_max(competition_key);
                    uint str_idx = any_nonzero(winner_val == competition_key? idx + 1 : 0) - 1;

                    // float second_winner = 0.0;
                    // if (U_flatten != 0){
                    //     second_winner = parallel_max(winner_val == competition_key? 0.0 : competition_key);
                    //     second_winner = any_nonzero(second_winner == competition_key? activity.activation: 0.0);
                    // }

                    if (str_idx == idx){
                        float final_val = U_flatten != 0.0 ? U_flatten : activity.activation;
                        // if (U_flatten == 0){
                        //     if (U_signed){
                        //         final_val -= abs(second_winner) * sign(final_val);
                        //     }
                        //     else{
                        //         final_val -= second_winner;
                        //     }
                        // }

                        LAYER_STR_DATA(data, LAYER_IDX, group_idx, idx);
                        LAYER_STR_DATA(data, LAYER_ACTIV, str_idx, floatBitsToInt(final_val));
                        LAYER_SET_BIT(data, str_idx);
                    }
                }
            '''
        )
        return self

    def acc_to_activ(self, sign_only=False):
        self.clear(counter=self.size)
        self.active_estimate = self.size
        compute_inline(
            self.size,
            layer_data=self.data,
            U_mask_size=ceil(self.size // 32),
            U_sign_only=bool(sign_only),

            code='''//glsl
                int idx = int(gl_GlobalInvocationID.x);
                float val = AS_FP(LAYER_GET_DATA(layer_data, LAYER_ACC, idx));
                if (U_sign_only){
                    val = sign(val);
                }
                LAYER_STR_DATA(layer_data, LAYER_ACTIV, idx, floatBitsToInt(val));
                LAYER_STR_DATA(layer_data, LAYER_IDX, idx, idx);
                if (idx < U_mask_size){
                    LAYER_STR_DATA(layer_data, LAYER_MASK, idx, 0xffffffff);
                }
            '''
        )

    def bias_update(self, wearout=0.0, recovery=0.0, bias_factor=0.0, neg_bias_factor=None, *, neg_layer=None, pos_layer=None):
        contrastive = bool(neg_layer)

        if neg_bias_factor is None:
            neg_bias_factor = -bias_factor

        if pos_layer is None:
            pos_layer = self
            bias_factor = 0.0

        if neg_layer is None:
            neg_layer = self
            neg_bias_factor = 0.0

        compute_inline(
            self.size,
            self_data=self.data,
            pos_data=pos_layer.data,
            neg_data=neg_layer.data,
            U_bias_fac=float(bias_factor),
            U_bias_neg_fac=float(neg_bias_factor),
            U_wearout=float(wearout),
            U_recovery=float(recovery),
            U_contrastive=bool(contrastive),
            code='''//glsl
                uint idx = gl_GlobalInvocationID.x;
                float pos_actv = intBitsToFloat(LAYER_GET_DATA(pos_data, LAYER_ACTIV, idx));
                float neg_actv = intBitsToFloat(LAYER_GET_DATA(neg_data, LAYER_ACTIV, idx));
                bool pos_bit = bool(LAYER_GET_BIT(pos_data, idx));
                bool neg_bit = bool(LAYER_GET_BIT(neg_data, idx));


                if (U_wearout != 0.0 || U_recovery != 0.0){
                    float gain = intBitsToFloat(LAYER_GET_DATA(self_data, LAYER_GAIN, gl_GlobalInvocationID.x));
                    float mix_fac = pos_bit ? U_wearout : U_recovery;
                    float side_fac = pos_bit ? 0.0 : 1.0;
                    float mix_enable = 1.0;
                    if (U_contrastive){
                        // pos,     neg,     enable?
                        // 0        0        1,   regular recovery
                        // 0        1        0   punish neuron by not recovering it
                        // 1        0        0   reward neuron by not wearing it down
                        // 1        1        1    regular wearout

                        mix_enable = float(int(pos_bit == neg_bit));
                    }

                    gain = mix(gain, side_fac, mix_fac * mix_enable);

                    LAYER_STR_DATA(self_data, LAYER_GAIN, gl_GlobalInvocationID.x, floatBitsToInt(gain));
                }
                if (U_bias_fac != 0.0 || U_bias_neg_fac != 0.0){
                    float bias = intBitsToFloat(LAYER_GET_DATA(self_data, LAYER_BIAS, gl_GlobalInvocationID.x));
                    bias += pos_actv * U_bias_fac + neg_actv * U_bias_neg_fac;

                    LAYER_STR_DATA(self_data, LAYER_BIAS, gl_GlobalInvocationID.x, floatBitsToInt(bias));

                }

            '''
        )
        return self

    def clear_bias(self):
        self.clear(bitmask=False, indices=False, accum=False, gain=True, bias=True, activation=False)
        return self

    def reset(self):
        self.clear(bitmask=True, indices=True, accum=True, gain=False, bias=False, activation=True, counter=0)
        return self

    def copy(self, to: 'Layer' = None) -> 'Layer':
        if to is None:
            return self.copy(to=Layer(self.size))

        if not to.size == self.size:
            raise ValueError('Target Layer object of different size')

        compute_inline(
            self.size,
            original=self.data,
            new=to.data,
            code='''//glsl
                int idx = int(gl_GlobalInvocationID.x);
                for (int ycoord = 0; ycoord<LAYER_NROWS; ycoord++){
                    ivec2 coord = ivec2(idx, ycoord);
                    imageStore(new, coord, imageLoad(original, coord));
                }
            '''
        )
        to.active_estimate = self.active_estimate
        to.seed = self.seed
        return to

    def _imask_op(self, other: 'Layer', op='|'):
        self.clear(bitmask=False, indices=True, accum=False, gain=False, counter=0)
        compute_inline(
            ceil(self.size / 32),
            this_data=self.data,
            other_data=other.data,
            defines={'OP': op},
            code='''//glsl
                int chunk = int(gl_GlobalInvocationID.x);
                uint a = uint(LAYER_GET_DATA(this_data, LAYER_MASK, chunk));
                uint b = uint(LAYER_GET_DATA(other_data, LAYER_MASK, chunk));
                uint c = a OP b;
                LAYER_STR_DATA(this_data, LAYER_MASK, chunk, c);

                int count = LAYER_COUNTER_INC(this_data, bitCount(c));

                while (c != 0){
                    int offset = findLSB(c);
                    c ^= 1u << offset;
                    LAYER_STR_DATA(this_data, LAYER_IDX, count, chunk * 32 + offset);
                    count += 1;
                }
            '''
        )

        # Over-estimate active count
        if op in '|^':
            self.active_estimate = min(self.size, self.active_estimate + other.active_estimate)

        elif op == '&':
            self.active_estimate = min(self.active_estimate, other.active_estimate)

        else:
            self.active_estimate = self.size

        return self

    def __iand__(self, other: 'Layer') -> 'Layer':
        return self._imask_op(other, op='&')

    def __ior__(self, other: 'Layer') -> 'Layer':
        return self._imask_op(other, op='|')

    def __ixor__(self, other: 'Layer') -> 'Layer':
        return self._imask_op(other, op='^')

    def __and__(self, other: 'Layer') -> 'Layer':
        return self.copy()._imask_op(other, op='&')

    def __or__(self, other: 'Layer') -> 'Layer':
        return self.copy()._imask_op(other, op='|')

    def __xor__(self, other: 'Layer') -> 'Layer':
        return self.copy()._imask_op(other, op='^')

    def _iaccum_op(self, other, op='+'):
        scalar = str(int(isinstance(other, (int, float))))

        # forbiden operations 1≃scalar 0=nonscalar
        if (scalar + op) in {'1/', '1+', '1-', '0/', }:
            return NotImplemented

        self.clear(indices=True, bitmask=True, counter=0)

        compute_inline(
            self.size,
            this_data=self.data,
            other=other.data if isinstance(other, Layer) else float(other),
            defines={'OP': op, 'SCALAR': scalar},
            code='''//glsl
                int idx = int(gl_GlobalInvocationID.x);
                float a = AS_FP(LAYER_GET_DATA(this_data, LAYER_ACC, idx));
                float x = intBitsToFloat(LAYER_GET_DATA(this_data, LAYER_ACTIV, idx));
            #if SCALAR == 1
                float c = a OP other;
                float d = x OP other;
            #else
                float oa = AS_FP(LAYER_GET_DATA(other, LAYER_ACC, idx));
                float ox = intBitsToFloat(LAYER_GET_DATA(other, LAYER_ACTIV, idx));
                float c = a OP oa;
                float d = x OP ox;
            #endif
                LAYER_STR_DATA(this_data, LAYER_ACC, idx, AS_FX(c));
                LAYER_STR_DATA(this_data, LAYER_ACTIV, idx, floatBitsToInt(d));
                if (d != 0){
                    LAYER_SET_BIT(this_data, idx);
                    int str_idx = LAYER_COUNTER_INC(this_data, 1);
                    LAYER_STR_DATA(this_data, LAYER_IDX, str_idx, idx);
                }
            '''
        )
        return self

    def __iadd__(self, other) -> 'Layer':
        return self._iaccum_op(other, '+')

    def __isub__(self, other) -> 'Layer':
        return self._iaccum_op(other, '-')

    def __imul__(self, other) -> 'Layer':
        return self._iaccum_op(other, '*')

    def __itruediv__(self, other) -> 'Layer':
        return self._iaccum_op(other, '/')

    def __add__(self, other) -> 'Layer':
        return self.copy()._iaccum_op(other, '+')

    def __sub__(self, other) -> 'Layer':
        return self.copy()._iaccum_op(other, '-')

    def __mul__(self, other) -> 'Layer':
        return self.copy()._iaccum_op(other, '*')

    def __truediv__(self, other) -> 'Layer':
        return self.copy()._iaccum_op(other, '/')

    def __matmul__(self, other):
        return OuterProdExpr([(self, other, 1)])

    def __rshift__(self, other):
        return PendingRightProjection(other, self, 'right')

    def __lshift__(self, other):
        return PendingLeftProjection(other, self, 'left')

    def __setitem__(self, key, val):
        if isinstance(key, slice):
            if key == slice(None) and isinstance(val, Layer):
                if not val.size == self.size:
                    raise ValueError('both layers must be the same size for a full copy')
                val.copy(to=self)
            else:
                self.set_acc_range(key.start, key.stop, key.step or 1, val=val)

        elif isinstance(key, int):
            self.set_acc_range(key, key + 1, 1, val=val)


class SynapsesBase(ABC):
    @abstractmethod
    def aproj(self, a: Layer, b: Layer, factor: float):
        raise NotImplementedError

    @abstractmethod
    def bproj(self, a: Layer, b: Layer, factor: float):
        raise NotImplementedError

    @abstractmethod
    def train(self, a, b, neg_a=None, neg_b=None, *, factor=1.0, neg_factor=None, use_activity=False, scatter_samples=None):
        raise NotImplementedError

    def __iadd__(self, other: OuterProdExpr):
        if len(other.items) == 2:
            x, y = other.items
            if x[-1] * y[-1] < 0:
                if x[-1] < y[-1]:
                    x, y = y, x

                (a, b, f), (an, bn, fn) = x, y

                self.train(a, b, an, bn, factor=f, neg_factor=fn, use_activity=True, scatter_samples=other.samples)
                return self

        for a, b, f in other.items:
            self.train(a, b, factor=f, use_activity=True, scatter_samples=other.samples)

        return self

    def __isub__(self, other: OuterProdExpr):
        self += -other
        return self

    def __mul__(self, other: float | int):
        if not isinstance(other, (float, int)):
            return NotImplemented

        return SynapsesExpr(synapses=self, factor=other, transpose=False)

    def __rmul__(self, other: float | int):
        return self * other

    def __truediv__(self, other: float | int):
        return self * (1.0 / other)

    def __neg__(self):
        return self * -1.0

    @property
    def T(self):
        return SynapsesExpr(synapses=self, factor=1.0, transpose=True)


class SynapsesExpr(SynapsesBase):
    def __init__(self, synapses: SynapsesBase, factor: float | int, transpose: bool):
        self.synapses = synapses
        self.factor = factor
        self.transpose = transpose

    def __repr__(self):
        fac_str = f"{self.factor} * " if self.factor not in {1.0, 1} else ''
        return f"({fac_str}{self.synapses!r}){('', '.T')[bool(self.transpose)]}"

    def aproj(self, a: Layer, b: Layer, factor: float):
        if self.transpose:
            return self.synapses.bproj(a=b, b=a, factor=factor * self.factor)

        return self.synapses.aproj(a=a, b=b, factor=factor * self.factor)

    def bproj(self, a: Layer, b: Layer, factor: float):
        if self.transpose:
            return self.synapses.aproj(a=b, b=a, factor=factor * self.factor)

        return self.synapses.bproj(a=a, b=b, factor=factor * self.factor)

    def train(self, a, b, neg_a=None, neg_b=None, *, factor=1.0, neg_factor=None, use_activity=False, scatter_samples=None):

        if self.transpose:
            a, b = b, a
            neg_a, neg_b = neg_b, neg_a

        if neg_factor is not None:
            neg_factor = neg_factor / self.factor

        return self.synapses.train(a=a, b=b, neg_a=neg_a, neg_b=neg_b,
                                   factor=factor / self.factor,
                                   neg_factor=neg_factor,
                                   use_activity=use_activity,
                                   scatter_samples=scatter_samples)

    def __mul__(self, other: int):
        return SynapsesExpr(synapses=self.synapses, factor=self.factor * other, transpose=self.transpose)

    @property
    def T(self):
        return SynapsesExpr(synapses=self.synapses, factor=self.factor, transpose=not self.transpose)


class Synapses(SynapsesBase):

    _get_set_include = '''//glsl
        int synapse_unpack_sign(int val){
            if (SIGNED){
                return (val << (32 - BITDEPTH)) >> (32 - BITDEPTH);
            }
            return val;
        }
    
        int synapse_value_clamp(int val){
            if (SIGNED){
                int val_range = int(BITMASK) ^ (1 << (BITDEPTH - 1));
                return clamp(val, -val_range, val_range);
            }
            return clamp(val, 0, int(BITMASK));
        }

        int get_synapse(int ai, int bi){
            ivec2 synapse_idx = ivec2(bi / CHUNK_SIZE, ai);
            uint chunk = imageLoad(matdata, synapse_idx).x;
            return synapse_unpack_sign(int((chunk >> ((bi % CHUNK_SIZE) * BITDEPTH)) & BITMASK));
        }
        
        void update_synapse(int ai, int bi, int synapse){
            uint clamped_synapse = uint(synapse_value_clamp(synapse)) & BITMASK;

            ivec2 synapse_idx = ivec2(bi / CHUNK_SIZE, ai);
            uint chunk = imageLoad(matdata, synapse_idx).x;
            
            while (true){
                
                uint modified_chunk = chunk;

                modified_chunk &= ~uint(BITMASK << (BITDEPTH * (bi % CHUNK_SIZE)));
                modified_chunk |= uint(clamped_synapse << (BITDEPTH * (bi % CHUNK_SIZE)));

                uint read_chunk = imageAtomicCompSwap(matdata, synapse_idx, chunk, modified_chunk);

                if (read_chunk == chunk) break;
                
                chunk = read_chunk;
            }
        }
    '''

    _train_include = _get_set_include + '''//glsl

        uint hash(uint x, uint y, uint z){
            return uxormix3(U_transpose_rng ? uvec3(y, x, z) : uvec3(x, y, z));
        }

        void increment_synapse(int ai, int bi, float change){
        
            float rng = uint2F(hash(ai, bi, U_seed));
            ivec2 synapse_idx = ivec2(bi / CHUNK_SIZE, ai);

            if (BITDEPTH == 1){

                uint synapse = 1u << (bi % CHUNK_SIZE);

                if (rng >= abs(change)) return;
                if (change > 0.0){
                    imageAtomicOr(matdata, synapse_idx, synapse);
                    return;
                }
                else {
                    imageAtomicAnd(matdata, synapse_idx, ~synapse);
                    return;
                }

            }
            else {

                int increment = int(abs(change)) + int(rng < fract(abs(change)));
                increment = change < 0? -increment : increment;

                if (increment == 0) return;

                int synapse = get_synapse(ai, bi);
                synapse += increment;
                update_synapse(ai, bi, synapse);

            }
        }
    '''

    def __init__(self, h, w, seed=1, bitdepth=2, signed=False):
        self.w = w
        self.h = h
        self.size = (w, h)
        self.signed = bool(signed)

        if w < 2:
            # Required to smuggle the activity counter into the end of the mask row.
            raise ValueError('Matrix width cant be less than 2')

        if signed and bitdepth == 1:
            raise ValueError('bitdepth of 1 cannot be signed')

        if bitdepth not in {1, 2, 4, 8, 16, 32}:
            raise ValueError('invalid bitdepth')

        self.bitdepth = bitdepth
        self.chunk_size = 32 // bitdepth
        self.bitmask = (1 << bitdepth) - 1

        # normalize scle_fac so our weights behave in 0-1 range
        if signed:
            self.scale_fac = 1.0 / (self.bitmask >> 1)

        else:
            self.scale_fac = 1.0 / self.bitmask

        self._defines = {'CHUNK_SIZE': self.chunk_size,
                         'BITDEPTH': self.bitdepth,
                         'SIGNED': ['false', 'true'][self.signed],
                         'BITMASK': '((1u << BITDEPTH) - 1u)'}

        if w % self.chunk_size != 0:
            raise ValueError(f'width must be a multiple of {self.chunk_size} for bitdepth={bitdepth}')

        self.data = Texture((w // self.chunk_size, h), format=R32UI)
        self.data.clear('UINT', (0,))

        self.seed = int(seed)
        self._transpose_rng = False
        self._transpose_mirror = None

    def load(self, buffer):
        size = np.prod(self.data.size)
        buffer = np.frombuffer(buffer).view(np.uint32)[:size]
        if len(buffer) < size:
            raise ValueError('data too small')
        self.data = Texture(self.data.size, format=R32UI, data=buffer)

    def dump(self):
        return self.data.read_array().reshape(-1, self.data.width)

    def set_row(self, row_idx: int, row_buff: np.ndarray):
        if not len(row_buff) == self.w:
            raise ValueError('row_buff: size does not match')
        if not (0 <= row_idx < self.h):
            raise ValueError('row_idx out of bounds')

        if not isinstance(row_buff, np.ndarray) or row_buff.dtype != np.uint32:
            row_buff = np.asarray(row_buff, dtype=np.uint32)

        compute_inline(
            self.data.width,
            U_true_w=int(self.w),
            U_row_idx=int(row_idx),
            **prepare_upload_ubos(row_buff),
            matdata=self.data,

            include=['upload_ubo'],
            include_source=self._get_set_include,
            defines=self._defines,

            code='''//glsl
                uint chunk = 0;
                for (uint i=0; i<CHUNK_SIZE; i++){
                    uint synapse_x = gl_GlobalInvocationID.x * CHUNK_SIZE + i;
                    if (synapse_x < U_true_w){
                        chunk |= ((synapse_value_clamp(int(get_upload_item(synapse_x)))) & BITMASK) << i * BITDEPTH;
                    }
                }
                imageStore(matdata, ivec2(gl_GlobalInvocationID.x, U_row_idx), ivec4(chunk));
            '''
        )

    def set_col(self, col_idx: int, col_buff: np.ndarray):
        if not len(col_buff) == self.h:
            raise ValueError('col_buff: size does not match')
        if not (0 <= col_idx < self.w):
            raise ValueError('col_idx out of bounds')

        if not isinstance(col_buff, np.ndarray) or col_buff.dtype != np.uint32:
            col_buff = np.asarray(col_buff, dtype=np.uint32)

        compute_inline(
            self.h,
            U_col_idx=int(col_idx),
            U_true_h=int(self.h),
            **prepare_upload_ubos(col_buff),
            matdata=self.data,

            include=['upload_ubo'],
            include_source=self._get_set_include,
            defines={**self._defines},

            code='''//glsl
                int ai = int(gl_GlobalInvocationID.x);
                if (ai < U_true_h){
                    int val = int(get_upload_item(uint(ai)));
                    update_synapse(ai, U_col_idx, val);
                }
            '''
        )

    def _get_line(self, idx: int, is_row: bool) -> np.ndarray:
        size = self.w if is_row else self.h
        result_tex = Texture((size, 1), format=R32F)
        result_tex.clear('FLOAT', (0,))

        compute_inline(
            size,
            matdata=self.data,
            result=result_tex,
            U_idx=int(idx),
            U_is_row=bool(is_row),
            U_size=int(size),
            include_source=self._get_set_include,
            defines=self._defines,
            code='''//glsl
                int i = int(gl_GlobalInvocationID.x);
                ivec2 co = ivec2(i, U_idx);
                co = U_is_row? co : co.yx;
                
                int val = get_synapse(co.y, co.x);
                imageStore(result, ivec2(i, 0), vec4(intBitsToFloat(val)));
            '''
        )
        data = result_tex.read()
        data1 = gpu.types.Buffer('FLOAT', (size,), data)
        return np.frombuffer(data1, dtype=np.int32)

    def get_row(self, row_idx: int) -> np.ndarray:
        if not (0 <= row_idx < self.h):
            raise ValueError('row_idx out of bounds')
        return self._get_line(row_idx, is_row=True)

    def get_col(self, col_idx: int) -> np.ndarray:
        if not (0 <= col_idx < self.w):
            raise ValueError('col_idx out of bounds')
        return self._get_line(col_idx, is_row=False)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.h}, {self.w}, bitdepth={self.bitdepth})'

    def init_random(self, density, min_val=0, max_val=1, seed=None):
        if not seed:
            self.seed += 1

        self.data.clear('UINT', (0,))

        if not self.signed:
            min_val = 0

        compute_inline(
            self.w,
            self.h,

            local_size=(16, 16, 1),

            data=self.data,
            U_seed=int(seed or self.seed),
            U_density=float(density),
            U_max_val=max_val / self.scale_fac,
            U_min_val=min_val / self.scale_fac,
            # pass as uniform so max_val doesnt get optmized out for bitdepth=1
            U_transpose_rng=bool(self._transpose_rng),
            defines=self._defines,

            code='''//glsl
                ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
                set_seed(uvec3((U_transpose_rng ? coord.yx : coord.xy), U_seed));

                float f = rndF();
                float f1 = rndF();

                uint synapse = (f < U_density? uint(round(mix(U_min_val, U_max_val, f1))) : 0u);
                ivec2 store_coord = ivec2(
                    coord.x / CHUNK_SIZE,
                    coord.y
                );
                synapse <<= (coord.x % CHUNK_SIZE) * BITDEPTH;
                imageAtomicOr(data, store_coord, synapse);
            '''
        )
        return self

    def aproj(self, a, b, factor):
        compute_inline(
            min(self.data.width, ceil(b.size / self.chunk_size)),
            a.active_estimate,

            local_size=(16, 32, 1),

            U_factor=float(factor) * self.scale_fac,
            a_data=a.data,
            b_data=b.data,
            matdata=self.data,
            defines=self._defines,

            include_source=self._get_set_include,

            wrap_main=False,


            code='''//glsl
            #define SHARED sha##red
            SHARED int shared_acc[16][CHUNK_SIZE];

            uint rotate_right(uint x, uint r) {
                r = r & 31u;
                return (x >> r) | (x << ((32u - r) & 31u));
            }

            void main(){
                int local_y = int(gl_LocalInvocationID.y);
                int local_x = int(gl_LocalInvocationID.x);

                if (local_y < CHUNK_SIZE) shared_acc[local_x][local_y] = 0;
                barrier();

                if (boundscheck()){
                    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
                    int row = LAYER_GET_DATA(a_data, LAYER_IDX, coord.y);
                    float input_activity = 0;

                    uint chunk = 0;
                    if (row != INDEX_INACTIVE && row < imageSize(matdata).y){
                        chunk = imageLoad(matdata, ivec2(coord.x, row)).x;
                        input_activity = intBitsToFloat(LAYER_GET_DATA(a_data, LAYER_ACTIV, row));
                        input_activity *= U_factor;
                    }

                    uint rotation = gl_LocalInvocationID.y;
                    chunk = rotate_right(chunk, rotation * BITDEPTH);

                    uint iter_bits = 0;

                    // --------------------------------------------
                    // Iterating logic that skips zeros in highly sparse synapses,
                    // just a special case for expected sparsity of lower bitdepths

                    if (BITDEPTH <= 4) {
                        // Compute iter_bits only for supported bitdepths
                        if (BITDEPTH == 1) {
                            iter_bits = chunk;
                        } else if (BITDEPTH == 2) {
                            iter_bits = (chunk & 0x55555555u) | ((chunk >> 1) & 0x55555555u);
                        } else if (BITDEPTH == 4) {
                            iter_bits = (chunk & 0x55555555u) | ((chunk >> 1) & 0x55555555u);
                            iter_bits = (iter_bits & 0x11111111u) | ((iter_bits >> 2) & 0x11111111u);
                        }

                        /// iter_bits on-bits aligns with the lsb of each synapse if its on
                        while (iter_bits != 0){
                            uint bit = uint(findLSB(iter_bits));
                            uint shared_idx = (bit / BITDEPTH + rotation) % CHUNK_SIZE;
                            iter_bits ^= 1u << bit;

                            if (BITDEPTH == 1) {
                                atomicAdd(shared_acc[local_x][shared_idx], AS_FX(input_activity));
                            } else {
                                int synapse = synapse_unpack_sign(int((chunk >> bit) & BITMASK));
                                atomicAdd(shared_acc[local_x][shared_idx], AS_FX(synapse * input_activity));
                            }
                        }
                    } else {
                        // Regular iterative loop alternative for bitdepth > 4
                        for (int i = 0; i < CHUNK_SIZE; i++){
                            int synapse = synapse_unpack_sign(int(chunk & BITMASK));
                            atomicAdd(shared_acc[local_x][(i + rotation) % CHUNK_SIZE], AS_FX(synapse * input_activity));
                            chunk >>= BITDEPTH;
                        }
                    }
                }

                // --------------------------------------------

                barrier();
                uint out_neuron = gl_GlobalInvocationID.x * CHUNK_SIZE + local_y;
                if (local_y < CHUNK_SIZE && out_neuron < imageSize(b_data).x){
                    if (shared_acc[local_x][local_y] != 0){
                        imageAtomicAdd(b_data, ivec2(out_neuron, LAYER_ACC), shared_acc[local_x][local_y]);
                    }
                };
            }
            '''
        )
        return b

    def bproj(self, a, b, factor):
        defines = {'SIZE_X': '8'}
        size_x = 8
        size_y = 8

        compute_inline(
            min(a.size, self.h),
            b.active_estimate,

            U_factor=float(factor) * self.scale_fac,
            a_data=a.data,
            b_data=b.data,
            matdata=self.data,
            defines={**self._defines, **defines},

            local_size=(size_x, size_y, 1),

            include_source=self._get_set_include,

            wrap_main=False,

            code='''//glsl
                #define SHARED sha##red
                SHARED int shared_acc[SIZE_X];

                void main(){
                    int local_x = int(gl_LocalInvocationID.x);
                    int local_y = int(gl_LocalInvocationID.y);

                    if (local_y == 0) shared_acc[local_x] = 0;

                    barrier();

                    if (boundscheck()){
                        int idx_a = int(gl_GlobalInvocationID.x);
                        int idx_b = LAYER_GET_DATA(b_data, LAYER_IDX, gl_GlobalInvocationID.y);

                        if (idx_b != INDEX_INACTIVE && (idx_b / CHUNK_SIZE) < imageSize(matdata).x){

                            uint chunk = imageLoad(matdata, ivec2(idx_b / CHUNK_SIZE, idx_a)).x;
                            int synapse = int((chunk >> ((idx_b % CHUNK_SIZE) * BITDEPTH)) & BITMASK);
                            synapse = synapse_unpack_sign(synapse);

                            if (synapse != 0){
                                float input_activity = intBitsToFloat(LAYER_GET_DATA(b_data, LAYER_ACTIV, idx_b));
                                atomicAdd(shared_acc[gl_LocalInvocationID.x], AS_FX(synapse * input_activity * U_factor));
                            }
                        }
                    }

                    barrier();
                    if (boundscheck()){
                        if (local_y == 0){
                            int acc_val = shared_acc[local_x];
                            if (acc_val != 0){
                                imageAtomicAdd(a_data, ivec2(gl_GlobalInvocationID.x, LAYER_ACC), acc_val);
                            }

                        }
                    }
                }
            '''
        )
        return a

    def sync_seed(self, other: 'Synapses', transpose=False):
        self.seed = other.seed
        self._transpose_rng = transpose

    def train(self, a, b, neg_a=None, neg_b=None, *, factor=1.0, neg_factor=None, use_activity=False, scatter_samples=None):
        self.seed += 1
        neg_factor = -factor if neg_factor is None else neg_factor

        factor /= self.scale_fac
        neg_factor /= self.scale_fac

        use_minus = False
        if neg_a and neg_b:
            if a.size != neg_a.size or b.size != neg_b.size:
                raise ValueError('positive and negative layers must be the same size')
            use_minus = True

        else:
            neg_a, neg_b = a, b
        # dispatch an excess of active threads.
        # many will do nothing if the difference between activation counts in positive and negative passes is too high,
        # but that rarely happens. assumes pos and neg layers are the same size
        # and that the active indices row in the layer will be padded with -1 for unused slots
        dispatch_x = max(neg_b.active_estimate, b.active_estimate)
        dispatch_y = max(neg_a.active_estimate, a.active_estimate)

        if scatter_samples is not None:
            if not self._transpose_rng:
                dispatch_y = scatter_samples
            else:
                dispatch_x = scatter_samples

        compute_inline(
            dispatch_x,
            dispatch_y,

            local_size=(128, 1, 1) if self._transpose_rng else (1, 128, 1),

            pos_a=a.data,
            pos_b=b.data,
            neg_a=neg_a.data,
            neg_b=neg_b.data,

            matdata=self.data,
            U_factor=float(factor),
            U_neg_factor=float(neg_factor),
            U_seed=self.seed,
            U_n_samples=int(scatter_samples or 0),
            U_use_activity=bool(use_activity),
            U_use_minus=bool(use_minus),
            U_transpose_rng=bool(self._transpose_rng),

            defines=self._defines,

            wrap_main=False,
            include_source=self._train_include,

            code='''//glsl

                bool matrix_boundary_check(int ai, int bi){
                    ivec2 size = imageSize(matdata);
                    return ai != INDEX_INACTIVE && bi != INDEX_INACTIVE && ai < size.y && bi < size.x * CHUNK_SIZE;
                }

                int strided_choice(float pool_size, float sample_size, int idx, int lane){

                    float bin_size = pool_size / sample_size;
                    
                    float global_shift = uint2F(uxormix3(uvec3(lane, U_seed, 1))) * pool_size;
                    float local_shift = uint2F(uxormix3(uvec3(lane, idx, U_seed))) * bin_size;

                    float offset = bin_size * idx + local_shift + global_shift;
                    return int(mod(offset, pool_size));
                }

                ivec2 strided_transpose_choice(int a_count, int b_count){
                    bool t = U_transpose_rng;
                    ivec2 glob = ivec2(gl_GlobalInvocationID.xy);

                    if (!t){
                        int choice = strided_choice(a_count, U_n_samples, glob.y, glob.x);
                        return ivec2(choice, glob.x);
                    }
                    else{
                        int choice = strided_choice(b_count, U_n_samples, glob.x, glob.y);
                        return ivec2(glob.y, choice);
                    }

                }

                bool use_samples_test(int num_samples, int a_count, int b_count){
                    if (U_transpose_rng){
                        return num_samples > 0 && num_samples < b_count;
                    }
                    else {
                        return num_samples > 0 && num_samples < a_count;
                    }
                }
                
                void main(){
                    int global_x = int(gl_GlobalInvocationID.x);
                    int global_y = int(gl_GlobalInvocationID.y);
                    
                    float pos_change_rescale = 1.0;
                    float neg_change_rescale = 1.0;

                    int a_count_pos = LAYER_COUNTER_GET(pos_a);
                    int a_count_neg = 0;
                    int b_count_pos = LAYER_COUNTER_GET(pos_b);
                    int b_count_neg = 0;
                    
                    if (U_use_minus){
                        a_count_neg = LAYER_COUNTER_GET(neg_a);
                        b_count_neg = LAYER_COUNTER_GET(neg_b);
                    }

                    if (boundscheck()){
                    
                        int ai_p;
                        int bi_p;

                        if (use_samples_test(U_n_samples, a_count_pos, b_count_pos)){
                            pos_change_rescale = float(U_transpose_rng ? b_count_pos : a_count_pos) / U_n_samples;
                            ivec2 choice_idx = strided_transpose_choice(a_count_pos, b_count_pos);
                            ai_p = LAYER_GET_DATA(pos_a, LAYER_IDX, choice_idx.x);
                            bi_p = LAYER_GET_DATA(pos_b, LAYER_IDX, choice_idx.y);

                        }
                        else{
                            ai_p = LAYER_GET_DATA(pos_a, LAYER_IDX, global_y);
                            bi_p = LAYER_GET_DATA(pos_b, LAYER_IDX, global_x);
                        }

                        if (matrix_boundary_check(ai_p, bi_p)){
                            float change;

                            if (U_use_activity){
                                float a_activ = intBitsToFloat(LAYER_GET_DATA(pos_a, LAYER_ACTIV, ai_p));
                                float b_activ = intBitsToFloat(LAYER_GET_DATA(pos_b, LAYER_ACTIV, bi_p));

                                change = a_activ * b_activ * U_factor;
                            
                                if (U_use_minus){
                                    float a_activ_neg = intBitsToFloat(LAYER_GET_DATA(neg_a, LAYER_ACTIV, ai_p));
                                    float b_activ_neg = intBitsToFloat(LAYER_GET_DATA(neg_b, LAYER_ACTIV, bi_p));
                                    change += a_activ_neg * b_activ_neg * U_neg_factor;
                                }

                            }
                            else{
                                change = U_factor;
                                if (U_use_minus){
                                    if (bool(LAYER_GET_BIT(neg_a, ai_p)) && bool(LAYER_GET_BIT(neg_b, bi_p))){
                                        change += U_neg_factor;
                                    }
                                }
                            }
                            increment_synapse(ai_p, bi_p, change * pos_change_rescale);
                        }
                    }
                    
                    if (!U_use_minus) return;

                    if (boundscheck()){
                        int ai_n;
                        int bi_n;

                        if (use_samples_test(U_n_samples, a_count_neg, b_count_neg)){
                            neg_change_rescale = float(U_transpose_rng ? b_count_neg : a_count_neg) / U_n_samples;

                            ivec2 choice_idx = strided_transpose_choice(a_count_neg, b_count_neg);
                            ai_n = LAYER_GET_DATA(neg_a, LAYER_IDX, choice_idx.x);
                            bi_n = LAYER_GET_DATA(neg_b, LAYER_IDX, choice_idx.y);
                            
                        } else {
                            ai_n = LAYER_GET_DATA(neg_a, LAYER_IDX, global_y);
                            bi_n = LAYER_GET_DATA(neg_b, LAYER_IDX, global_x);
                        }

                        if (matrix_boundary_check(ai_n, bi_n)){
                            float change;
                            uint pos_bit_ai = LAYER_GET_BIT(pos_a, ai_n);
                            uint pos_bit_bi = LAYER_GET_BIT(pos_b, bi_n);
                            
                            // check if this has already been computed
                            if (bool(pos_bit_bi) && bool(pos_bit_ai)) return;

                                if (U_use_activity){
                                    //if you are here, then this synapse is a unporcessed negative update;
                                    float a_activ_neg = intBitsToFloat(LAYER_GET_DATA(neg_a, LAYER_ACTIV, ai_n));
                                    float b_activ_neg = intBitsToFloat(LAYER_GET_DATA(neg_b, LAYER_ACTIV, bi_n));

                                    change = a_activ_neg * b_activ_neg * U_neg_factor;
                                } else {
                                    change = U_neg_factor;
                                }
                            increment_synapse(ai_n, bi_n, change * neg_change_rescale);
                        }
                    }
                }

            '''
        )
        return self

    def weight_erosion(self, a_mask: Layer, b_mask: Layer, factor: float, samples: int, anchor: str = 'b'):
        '''
        Weight Decay's weird cousin.
        '''
        self.seed += 1

        if anchor not in 'ab':
            raise ValueError('anchor must be eiter \'a\' or \'b\'')

        anchor: int = 'ab'.index(anchor)

        compute_inline(
            samples,
            a_mask=a_mask.data,
            b_mask=b_mask.data,
            matdata=self.data,
            U_factor=1.0 - float(factor),
            U_seed=int(self.seed),
            U_transpose_rng=bool(self._transpose_rng),
            U_anchor=anchor,
            dimensions=[self.w, self.h],
            include_source=self._train_include,
            defines=self._defines,

            code='''//glsl
                int idx = int(gl_GlobalInvocationID.x);
                uint h = uxormix3(uvec3(idx, U_seed, 1));
                uint x = xorshift(h);
                ivec2 co = ivec2(h, x);
                
                if (U_transpose_rng) co = co.yx;
                uint a_count = uint(LAYER_COUNTER_GET(a_mask));
                uint b_count = uint(LAYER_COUNTER_GET(b_mask));

                if (a_count == 0 || b_count == 0) return;

                if (U_anchor == 0){ // 'a'
                    co.y = LAYER_GET_DATA(a_mask, LAYER_IDX, co.y % a_count);
                    co.x %= dimensions.x;                    
                }
                else { // 'b'
                    co.x = LAYER_GET_DATA(b_mask, LAYER_IDX, co.x % b_count);
                    co.y %= dimensions.y;
                }

                uint a_bit = LAYER_GET_BIT(a_mask, co.y);
                uint b_bit = LAYER_GET_BIT(b_mask, co.x);

                if (bool(a_bit) && bool(b_bit)) return;

                float syn = get_synapse(co.y, co.x);
                float change = syn * U_factor - syn;

                increment_synapse(co.y, co.x, change);
                            
            '''
        )



