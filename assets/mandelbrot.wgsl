@group(0) @binding(0) var output: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn checker_board(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    if (location.x + location.y) % 2 == 0 {
        textureStore(output, location, vec4<f32>(1., 1., 1., 1.));
    } else {
        textureStore(output, location, vec4<f32>(0., 0., 0., 1.));
    }
}
