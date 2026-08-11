
# ifndef RND_LIB
# define RND_LIB

uint rnd_state = 12345678u;

float uint2F(uint x){
    x &= 0x007FFFFFu;
    x |= 0x3F800000u;
    return uintBitsToFloat(x) - 1.0;
}

uint xorshift(uint x){
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    return x;
}

float rndF(){
    rnd_state = xorshift(rnd_state);
    return uint2F(rnd_state);
}

uint rndUI(){
    rnd_state = xorshift(rnd_state);
    return rnd_state;
}


uint uxormix3(uvec3 x){
    uint a = x.x;
    a = (a ^ (a >> 16u)) * uint(369740719);
    a ^= x.y;
    a = (a ^ (a >> 16u)) * uint(743368733);
    a ^= x.z;
    a = (a ^ (a >> 16u)) * uint(424476911);
    a = (a ^ (a >> 16u)) * uint(741026501);
    return a;
}

uint ixormix4(ivec4 v){
    uvec4 x = uvec4(v);

    uint a = x.x ^ 123456789u;
    a = (a ^ (a >> 16u)) * uint(369740719);
    a ^= x.y;
    a = (a ^ (a >> 16u)) * uint(743368733);
    a ^= x.z;
    a = (a ^ (a >> 16u)) * uint(424476911);
    a ^= x.w;
    a = (a ^ (a >> 16u)) * uint(492513979);
    a = (a ^ (a >> 16u)) * uint(741026501);
    return a;
}

uint fxormix3(vec3 x){
    return uxormix3(floatBitsToUint(x));
}

void set_seed(vec3 inp){
    rnd_state = fxormix3(inp);
}

void uset_seed(uvec3 inp){
    rnd_state = uxormix3(inp);
}


float hashf(float f){
    uint x = floatBitsToUint(f);
    x += x << 10u;
    x ^= x >> 6u;
    x += x << 3u;
    x ^= x >> 11u;
    x += x << 15u;
    return abs(uint2F(x));
}

uint hashv3_ui(vec3 inp){
    uint x = floatBitsToUint(dot(inp, vec3(3.14159265359, 1.41421356237, 1.61803398874)));
    x += x << 10u;
    x ^= x >> 6u;
    x += x << 3u;
    x ^= x >> 11u;
    x += x << 15u;
    return x;
}


# endif