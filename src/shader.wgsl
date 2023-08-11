// Vertex shader
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}
// The @builtin(position) bit tells WGPU that this is the value we want
// to use as the vertex's clip coordinates.
// This is analogous to GLSL's gl_Position variable.
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}


@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}


// Fragment shader
@fragment
// The @location(0) bit tells WGPU to store the vec4 value returned
// by this function in the first color target
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>{
   return vec4<f32>(in.color, 1.0);
}
