from .core import *
from . import glsl

compute_inline = compute_inline_partial(local_size=(128, 1, 1), include=['random_lib', 'chaintable'])

glsl.register_include(
    'chaintable',
    '''//glsl
        #ifndef CHAINTABLE_H
        #define CHAINTABLE_H

        #ifndef RND_LIB
        #error random_lib.glsl not included
        #endif

        void chain_table_store_item(uint key, int value, int node_idx){
            ivec2 table_size = imageSize(header);
            uint size = uint(table_size.x * table_size.y);
            int prev_node_idx = atomic_exchange_idx(header, key % size, node_idx);
            store(nodes, node_idx, ivec4(prev_node_idx, value, 0, 0));
        }

        uint hash_point(vec3 co, float cell_size, uint seed){
            co /= cell_size;
            co = floor(co);
            return ixormix4(ivec4(ivec3(co), seed));
        }

        void chain_table_store_spatial_hash(vec3 co, float cell_size, int seed, int value, int node_idx){
            chain_table_store_item(hash_point(co, cell_size, seed), value, node_idx);
        }

        #endif
    '''
)

import numpy as np

class ChainTable:
    def __init__(self, capacity):
        size = calc_pot_size(capacity)
        self.header = Texture(size, format=R32I)
        self.nodes = Texture(size, format=RG32I)
        self.params = {'header': self.header, 'nodes': self.nodes}
        self.clear()
    
    def clear(self):
        self.header.clear('INT', (-1, -1, -1, -1))
        self.nodes.clear('INT', (-1, -1, -1, -1))


# table = ChainTable(64)
# points = Texture.from_array(np.random.sample(64*4).astype(np.float32), format=RGBA32F)

# compute_inline(
#     64,
#     **table.params,
#     points=points,
#     code='''//glsl
#         int idx = int(gl_GlobalInvocationID.x);
#         vec4 point = load(points, idx);
#         chain_table_store_spatial_hash(point.xyz, 0.5, 1, idx, idx);
        
#     ''')
# print(table.header.read_array())
# print(table.nodes.read_array())
