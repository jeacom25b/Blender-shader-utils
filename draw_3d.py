
import gpu
import bpy
import numpy as np
from mathutils import Vector, Matrix
from .shad.core import *
from gpu_extras.batch import batch_for_shader
import sys
import functools


__all__ = [
    "ViewData",
    "PolydotDraw",
    "PolylineDraw",
    "draw_lines_tex",
    "draw_dots_tex",
    "IMMDraw",
    "DrawHandler",
    "draw_handler",
]


def _z_offset_calc(z):
    # my math is almost certainly incorrect here but I dont care
    space = bpy.context.space_data
    view_scale = bpy.context.region_data.window_matrix[0][0]
    z = z * space.clip_start
    if bpy.context.region_data.is_perspective:
        z = z / view_scale
    else:
        z = z / view_scale * 0.0031
    return z


@ubo_struct
class ViewData:
    view_matrix: MAT4
    model_matrix: MAT4
    model_view_matrix: MAT4

    view_location: FLOAT4
    view_vector: FLOAT4

    model_view_location: FLOAT4
    model_view_vector: FLOAT4

    window_size: FLOAT2
    is_perspective: INT
    _pad: INT

    def update(self, context, model_matrix=Matrix.Identity(4)):
        area = context.area
        region = context.region_data
        vmat = region.view_matrix
        invvmat = vmat.inverted_safe()
        model_inv = model_matrix.inverted_safe()

        view_co = invvmat @ Vector((0, 0, 0))
        view_vec = -vmat[2].xyz

        model_view_co = model_inv @ view_co
        model_view_vec = view_vec
        model_view_vec.rotate(model_inv)
        model_view_vec.normalize()

        self['view_matrix'] = region.perspective_matrix
        self['model_matrix'] = model_matrix
        self['model_view_matrix'] = region.perspective_matrix @ model_matrix

        self['view_location'][:3] = view_co
        self['view_vector'][:3] = view_vec

        self['model_view_location'][:3] = model_view_co
        self['model_view_vector'][:3] = model_view_vec

        self['window_size'] = area.width, area.height
        self['is_perspective'] = region.is_perspective

        return self


class PolydotDraw:
    '''
    Point drawing shader constructor, takes in a `code` argumment as a string which must contain
    contain two implemented glsl functions:
        vec4 get_position(int point_id)
        vec4 get_color(int point_id)

    get_position() must return a vec4 where the .w component represents the size of the point.
    '''

    def __init__(self, *,
                 code,
                 include=[],
                 defines={},
                 shader_inputs={}):

        self._draw_points_ex = shader_program(
            include=include,
            defines=defines,
            shader_inputs={'view_data': ViewData,
                           'z_offset': FLOAT,
                           'vert_index': VIN[UINT],

                           'width_factor': FLOAT,
                           'color': VEC4,

                           'vcolor': VOUT[VEC4] | NO_PERSPECTIVE,
                           'point_uv': VOUT[VEC2] | NO_PERSPECTIVE,
                           'fragcol': FRAG_OUT[VEC4],
                           **shader_inputs},
            vert_code=f'#ifdef VERT_SHADER\n{code}\n#endif\n' + '''//glsl

            void main(){
                vec4 pos = get_position(gl_InstanceID); // prevent from vert stage getting optmized out
                
                vcolor = get_color(gl_InstanceID) * color;

                float point_size = pos.w;
                uint billboard_x = (vert_index >> 1) & 1;
                uint billboard_y = vert_index & 1;

                gl_Position = view_data.model_view_matrix * vec4(pos.xyz, 1.0f);

                point_uv = vec2(billboard_x, billboard_y);
                vec2 displace = (point_uv - vec2(0.5f)) * point_size * width_factor * 2;

                gl_Position.xy += (displace / view_data.window_size) * gl_Position.w;

                gl_Position.z -= z_offset / (1.0f + gl_Position.w);
            }
            ''',

            frag_code='''//glsl
            void main(){
                vec2 p = point_uv - vec2(0.5f);
                if (dot(p, p) > 0.25f) {discard; return;}
                fragcol = vcolor;
            }
            '''
        )

        self._point_batch = batch_for_shader(self._draw_points_ex.shader, 'TRIS', {'vert_index': [0, 1, 2, 3]},
                                             indices=[[0, 1, 2], [1, 3, 2]])

        self.params_set = self._draw_points_ex.params_set

    def __call__(self, num_points, z_offset=0.0, width_factor=1, color=(1, 1, 1, 1), view_data=None, **kwargs):
        self._draw_points_ex.instanced(self._point_batch, 0, num_points,
                                       width_factor=width_factor,
                                       color=color,
                                       view_data=view_data,
                                       z_offset=_z_offset_calc(z_offset),
                                       **kwargs)


class PolylineDraw:
    '''
    line drawing shader constructor, takes in a `code` argumment as a string which must contain
    contain two implemented glsl functions:
        vec4 get_position(int point_id)
        vec4 get_color(int point_id)

    point_id refers to the index of a point across all line segments where each line is represented
    by two consecutive indexes, for example line_0 = [0, 1] and line_1 = [2, 3], if strip=True is
    passed to __call__ during drawing, the last point_id of the previous line is the first point_id 
    of the next line, line_0 = [0, 1] and line_1 = [1, 2].

    get_position() must return a vec4 where the .w component represents the width of the line at that point.

    '''

    def __init__(self, *,
                 code,
                 include=[],
                 defines={},
                 shader_inputs={}):
        self._draw_lines_ex = shader_program(
            include=include,
            defines=defines,
            typedef_source='''struct Type {int value;};''',
            shader_inputs={'ModelViewProjectionMatrix': MAT4,
                           'view_data': ViewData,
                           'z_offset': FLOAT,
                           'feather': FLOAT,
                           'strip': BOOL,

                           'width_factor': FLOAT,
                           'color': VEC4,

                           'vert_index': VIN[INT],
                           'vcolor': VOUT[VEC4] | NO_PERSPECTIVE,
                           'line_coord': VOUT[VEC3] | NO_PERSPECTIVE,

                           'fragcol': FRAG_OUT[VEC4],
                           **shader_inputs},

            vert_code=f'#ifdef VERT_SHADER\n{code}\n #endif\n' + '''//glsl

            void main(){
                int segment_id = gl_InstanceID; // which line
                int segpoint_id = (vert_index >> 1) & 1; // which point in line
                int side_id = vert_index & 1; // which side of the strip
                int point_id = segment_id * 2 + segpoint_id; // which point among all points
                
                
                vec4 points[2];
                if (strip){
                    points[0] = get_position(segment_id);
                    points[1] = get_position(segment_id + 1);
                    vcolor = get_color(segment_id + segpoint_id);

                } else {
                    points[0] = get_position(segment_id * 2);
                    points[1] = get_position(segment_id * 2 + 1);
                    vcolor = get_color(point_id) * color;
                }
                

                // points that fall behind the camera do weird things.
                // clip to view plane to prevent it from summoning unseen eldrich forces
                
                vec2 window_size = view_data.window_size;
                vec3 view_location = view_data.model_view_location.xyz;
                vec3 view_vector = view_data.model_view_vector.xyz;
                bool is_perspective = bool(view_data.is_perspective);

                float pdv = dot(view_location - points[segpoint_id].xyz, view_vector);
                if (is_perspective && pdv > 0.0f){
                    vec3 pvec = points[1 - segpoint_id].xyz - points[segpoint_id].xyz;
                    float fac = (pdv + 0.0000001f) / dot(pvec, view_vector);
                    points[segpoint_id] = mix(points[segpoint_id], points[1 - segpoint_id], fac);
                    // NOTE TO FUTURE SELF: yes, this is wrong, should interpolate the color too for correctness
                    // but I dont care, minor issue and saves one function call.
                }

                vec4 sp_points[2] = { // normalized device coordinates
                    view_data.model_view_matrix * vec4(points[0].xyz, 1),
                    view_data.model_view_matrix * vec4(points[1].xyz, 1)
                };

                // direction of the line in NDC space
                vec2 delta = sp_points[1].xy / sp_points[1].w - sp_points[0].xy / sp_points[0].w;

                // correct aspect ratio of the thickness vector
                delta *= (window_size / vec2(max(window_size.x, window_size.y)));
                
                // vec2 normabs_delta = abs(normalize(delta));
                // float width_correction = max(normabs_delta.x, normabs_delta.y);
                // float point_width = points[segpoint_id].w / width_correction * width_factor;

                // vec2 sign_delta = sign(delta);
                // vec2 extrude_dir = vec2(-sign_delta.y, sign_delta.x);
                // extrude_dir[int(abs(delta.x) < abs(delta.y))] = 0.0;
                float point_width = points[segpoint_id].w * width_factor;
                vec2 extrude_dir = normalize(delta);
                extrude_dir = vec2(-extrude_dir.y, extrude_dir.x);

                extrude_dir = extrude_dir / window_size * point_width;
                
                extrude_dir *= (side_id == 0)? -1.0f : 1.0f; // extrude
                extrude_dir *= sp_points[segpoint_id].w;


                gl_Position = sp_points[segpoint_id] + vec4(extrude_dir, 0, 0);
                gl_Position.z -= z_offset / (1.0f + gl_Position.w);
                line_coord = vec3(side_id, segment_id, point_width);

            }
            ''',

            frag_code='''//glsl
            void main(){

                if (feather > 0){
                    float point_width = line_coord.z;
                    float fac = min(1.0f, point_width / feather * (1.0f - abs(line_coord.x * 2.0f - 1.0f)));
                    fragcol = vec4(vcolor.xyz, fac * vcolor.w);
                } else {
                    fragcol = vcolor;
                }
            }
            '''
        )

        self._line_batch = batch_for_shader(self._draw_lines_ex.shader, 'TRIS', {'vert_index': [0, 1, 2, 3]},
                                            indices=[[0, 1, 2], [1, 3, 2]])

        self.shader = self._draw_lines_ex.shader
        self._params_set = self._draw_lines_ex.params_set

    def __call__(self, num_points, z_offset=0.0, feather=1.5, width_factor=1, color=(1, 1, 1, 1), view_data=None, strip=False, **kwargs):

        if strip:
            num_lines = num_points - 1

        else:
            num_lines = num_points // 2

        self._draw_lines_ex.instanced(self._line_batch, 0, num_lines,
                                      view_data=view_data,
                                      z_offset=0 if z_offset == 0 else _z_offset_calc(z_offset),
                                      feather=feather,
                                      width_factor=width_factor,
                                      color=color,
                                      strip=strip,
                                      **kwargs)


draw_lines_tex = PolylineDraw(include=['2d_indexing'], shader_inputs={
    'points': FLOAT_2D[RGBA32F, READ],
    'colors': FLOAT_2D[RGBA32F, READ],
},
    code='''//glsl
        vec4 get_position(int point_id){
            return load(points, point_id);
        }

        vec4 get_color(int point_id){
            return load(colors, point_id);
        }
''')


draw_dots_tex = PolydotDraw(include=['2d_indexing'], shader_inputs={
    'points': FLOAT_2D[RGBA32F, READ],
    'colors': FLOAT_2D[RGBA32F, READ]
},
    code='''//glsl
    vec4 get_position(int point_id){
        return load(points, point_id);
    }

    vec4 get_color(int point_id){
        return load(colors, point_id);
    }

    '''
)
draw_dots_tex_single_color = PolydotDraw(include=['2d_indexing'], shader_inputs={
    'points': FLOAT_2D[RGBA32F, READ],
    'color': VEC4
},
    code='''//glsl
    vec4 get_position(int point_id){
        return load(points, point_id);
    }

    vec4 get_color(int point_id){
        return color;
    }

    '''
)


class IMMDraw:
    def __init__(self, max_capacity=1024):
        # TODO: extend this for automatic reallocation when capacity is reached
        # For now, we have to specify how many points will be allocated beforehand
        self.point_data = np.zeros((max_capacity, 4), dtype=np.float32)
        self.color_data = np.zeros((max_capacity, 4), dtype=np.float32)

        self.point_tex = None
        self.color_tex = None
        self.view_data = ViewData()
        self.i = 0

    def point(self, co, color, width=2):
        i = self.i
        self.point_data[i] = (*co, width)
        self.color_data[i] = color
        self.i += 1

    def clear(self):
        self.i = 0

    def update(self):
        self.point_tex = Texture.from_array(self.point_data, format=RGBA32F)
        self.color_tex = Texture.from_array(self.color_data, format=RGBA32F)

    def draw_as_lines(self, context, z_offset=0, feather=1, width_factor=1, color=(1, 1, 1, 1), matrix=Matrix.Identity(4)):
        self.view_data.update(context, model_matrix=matrix)
        draw_lines_tex(self.i,
                       z_offset=z_offset,
                       feather=feather,
                       width_factor=width_factor,
                       color=color,
                       colors=self.color_tex,
                       points=self.point_tex,
                       view_data=self.view_data)

    def draw_as_dots(self, context, z_offset=0, width_factor=1, color=(1, 1, 1, 1), matrix=Matrix.Identity(4)):
        self.view_data.update(context, model_matrix=matrix)
        draw_dots_tex(self.i,
                      z_offset=z_offset,
                      width_factor=width_factor,
                      color=color,
                      colors=self.color_tex,
                      points=self.point_tex,
                      view_data=self.view_data)


if False:
    '''
    Example usage for this stuff

    '''
    class TestDrawPoints(bpy.types.Operator):
        bl_idname = 'test.draw_points'
        bl_label = 'Draw Points'

        def draw_3d(self):
            gpu.state.depth_mask_set(True)
            gpu.state.depth_test_set('LESS_EQUAL')

            self.view_data.update(bpy.context)

            gpu.state.blend_set('ALPHA')
            gpu.state.point_size_set(20)
            draw_dots_tex(self.num_points,
                          points=self.points,
                          colors=self.colors,
                          z_offset=_z_offset_calc(0.05),
                          view_data=self.view_data,
                          )

            draw_lines_tex(self.num_points,
                           points=self.points,
                           colors=self.colors,
                           z_offset=_z_offset_calc(0.05),
                           feather=1,
                           strip=True,
                           view_data=self.view_data,
                           )

            gpu.state.depth_mask_set(False)
            gpu.state.depth_test_set('NONE')

        def invoke(self, context, event):
            self._handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_3d, (), "WINDOW", "POST_VIEW")
            N = 2000
            np.random.seed(0)

            items = np.arange(N).astype(np.float32)
            items *= 1 / 10

            self.points = np.ones((N, 4), dtype=np.float32)

            self.points[:, 0] = np.sin(items) * np.sin(items * 0.151351548)
            self.points[:, 1] = np.cos(items) * np.sin(items * 0.1976846154)
            self.points[:, 2] = np.cos(items) * np.sin(items * 0.348641397873)
            self.points[:, 3] = 4

            # self.points = np.random.sample(4 * N).astype(np.float32).reshape(N, 4)
            # self.points[:, 3] = 2
            # self.points[:2] = [[3, 3, 3, 5], [3, 3, 4, 5]]
            self.points = Texture.from_array(self.points, format='RGBA32F')

            self.colors = np.random.sample(4 * N).astype(np.float32).reshape(N, 4)
            self.colors[:, 3] = 1
            self.colors = Texture.from_array(self.colors, format='RGBA32F')

            self.num_points = N

            self.view_data = ViewData()

            context.window_manager.modal_handler_add(self)

            return {'RUNNING_MODAL'}

        def modal(self, context, event):
            if event.type == 'ESC':
                bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
                return {'FINISHED'}

            if event.type == 'NUMPAD_PLUS' and event.value == 'PRESS':
                self.N += 1
                return {'RUNNING_MODAL'}
            if event.type == 'NUMPAD_MINUS' and event.value == 'PRESS':
                self.N += -1
                return {'RUNNING_MODAL'}

            return {'PASS_THROUGH'}

    CLASSES = [TestDrawPoints]
