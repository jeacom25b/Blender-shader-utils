# ifndef MATH_FUNCS
# define MATH_FUNCS

# define NEG_INF uintBitsToFloat(0xFF800000)
# define EPS 1e-7f

float ensure_nonzero(float x){
    if (abs(x) <= EPS){
        return EPS;
    }
    return x;
}

bool almost_zerov3(vec3 x){
    x -= vec3(EPS);
    x = max(-sign(x), 0.0);
    return bool(x.x + x.y + x.z);
}
vec2 normalize_eps(vec2 x){
    float l = max(length(x), EPS);
    return x / l;
}
vec3 normalize_eps(vec3 x){
    float l = max(length(x), EPS);
    return x / l;
}
vec4 normalize_eps(vec4 x){
    float l = max(length(x), EPS);
    return x / l;
}

float estimate_curvature(vec3 p1, vec3 p2, vec3 n1, vec3 n2){
    vec3 dp = p2 - p1;
    vec3 dn = n2 - n1;
    return 2 * (dot(dn, dp) / max(dot(dp, dp), EPS));
}


struct VectorMatch{
    int index;
    float dprod;
    float dprod_abs;
};

VectorMatch best_match_vec(vec3 v, vec3 ref[2]){
    vec2 dprod = vec2(dot(v, ref[0]),
                      dot(v, ref[1]));

    vec2 dprod_abs = abs(dprod);

    VectorMatch r;
    r.index = int(dprod_abs[1] > dprod_abs[0]);
    r.dprod = dprod[r.index];
    r.dprod_abs = dprod_abs[r.index];
    return r;
}

void match_frames(vec3 ref1[2], vec3 ref2[2], out int indexes[2], out float signs[2], out int dominant_match){
    VectorMatch matches[2] = {
        best_match_vec(ref1[0], ref2),
        best_match_vec(ref1[1], ref2)
    };

    dominant_match = int(matches[1].dprod_abs > matches[0].dprod_abs);
    int other_match = 1 - dominant_match;
    
    if (matches[0].index == matches[1].index){
        matches[other_match].index = 1 - matches[dominant_match].index;
        matches[other_match].dprod = dot(ref1[other_match], ref2[matches[other_match].index]);
    }

    indexes[0] = matches[0].index;
    indexes[1] = matches[1].index;
    signs[0] = matches[0].dprod > 0? 1.0:-1.0;
    signs[1] = matches[1].dprod > 0? 1.0:-1.0;

}

vec3 project_point_line(vec3 point, vec3 line_point, vec3 line_normal){
    return line_point + line_normal * (dot(line_normal, point - line_point) / dot(line_normal, line_normal));
}

# endif