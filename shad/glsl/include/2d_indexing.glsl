# ifndef IDX2COORD

uint morton_decode8(uint x){
    x = (x & 0x44u) >> 1 | (x & 0x11u);
    x = (x & 0x30u) >> 2 | (x & 0x03u);
    return x;
}

ivec2 idx2coord_block(uint idx, ivec2 imsize){
    uint w = uint(imsize.x);
    uint block_id = idx >> 8u;
    uint item_id = idx & 0xffu;
    ivec2 item_offset = ivec2(morton_decode8(item_id), morton_decode8(item_id >> 1u));
    
    uint wblock = w >> 4;
    uint w_mask = wblock - 1u; // assumes imsize.x is a power of two
    uint w_shift = uint(findLSB(wblock));

    return (ivec2(block_id & w_mask, block_id >> w_shift) << 4u) + item_offset;
}

ivec2 idx2coord_block(int idx, ivec2 imsize){
    return idx2coord_block(uint(idx), imsize);
}

ivec2 idx2coord_row(int idx, ivec2 imsize){
    return ivec2(idx % imsize.x, idx / imsize.x);
}
ivec2 idx2coord_row(uint idx, ivec2 imsize){
    return ivec2(idx % imsize.x, idx / imsize.x);
}

# define IDX2COORD(im, idx) idx2coord_row(idx, imageSize(im))
# define IDX2COORD_TEX(im, idx) idx2coord_row(idx, textureSize(im, 0))

# define load(im, idx) imageLoad(im, IDX2COORD(im, idx))
# define store(im, idx, data) imageStore(im, IDX2COORD(im, idx), data)
# define fetch(im, idx) texelFetch(im, IDX2COORD_TEX(im, idx), 0)

# define atomic_min_idx(im, idx, data) imageAtomicMin(im, IDX2COORD(im, idx), data)
# define atomic_max_idx(im, idx, data) imageAtomicMax(im, IDX2COORD(im, idx), data)
# define atomic_add_idx(im, idx, data) imageAtomicAdd(im, IDX2COORD(im, idx), data)
# define atomic_and_idx(im, idx, data) imageAtomicAnd(im, IDX2COORD(im, idx), data)
# define atomic_or_idx(im, idx, data) imageAtomicOr(im, IDX2COORD(im, idx), data)
# define atomic_xor_idx(im, idx, data) imageAtomicXor(im, IDX2COORD(im, idx), data)
# define atomic_compswap_idx(im, idx, compare, data) imageAtomicCompSwap(im, IDX2COORD(im, idx), compare, data)
# define atomic_exchange_idx(im, idx, data) imageAtomicExchange(im, IDX2COORD(im, idx), data)

# endif

