struct Uniforms {
    time: u32,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

const MAX_ITERS: u32 = 200;
const DIVERGENCE_BOUND: f32 = 1.e5;

const POINT: vec2<f32> = vec2<f32>(0.743643887037151, 0.131825904205330);

struct Complex {
    x: f32,
    y: f32,
}

fn complex_add(a: Complex, b: Complex) -> Complex {
    return Complex(a.x + b.x, a.y + b.y);
}

fn complex_sq(z: Complex) -> Complex {
    return Complex(z.x * z.x - z.y * z.y, 2 * (z.x * z.y));
}

fn complex_mag2(z: Complex) -> f32 {
    return z.x * z.x + z.y * z.y;
}

fn divergence(c: Complex) -> f32 {
    var z: Complex = Complex(0., 0.);

    for (var i = 0u; i < MAX_ITERS; i++) {
        z = complex_add(complex_sq(z), c);
        let mag = complex_mag2(z);
        if mag >= DIVERGENCE_BOUND * DIVERGENCE_BOUND {
            return f32(i) - log(log(mag) / log(DIVERGENCE_BOUND)) / log(2);
        }
    }
    return -1.;
}

fn hsb2rgb(h: f32, s: f32, b: f32) -> vec3<f32> {
    let k = vec3<f32>(5.0, 3.0, 1.0);
    let p = abs(fract(vec3<f32>(h) + k / 6.0) * 6.0 - 3.0);
    let rgb = clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
    return b * mix(vec3<f32>(1.0), rgb, s);
}

fn to_mandel_coords(location: vec2<f32>) -> vec2<f32> {
    let f = pow(0.5, f32(uniforms.time) / 75.);
    return vec2<f32>((2. * location.x / 1280. - 1.) * f - POINT.x,
        (2. * location.y / 720. - 1.) * f - POINT.y);
}

fn mandelbrot_color(pos: vec2<f32>) -> vec4<f32> {
    let mandel_pos = to_mandel_coords(pos);
    let c: Complex = Complex(mandel_pos.x, mandel_pos.y);
    let d = divergence(c);
    if d < 0. {
        return vec4<f32>(0., 0., 0., 1.);
    }

    let decay = pow(1. - exp(-d / 20.), 2.);
    let angle = (d / 180. + .5) % 1.;
    let rgb = hsb2rgb(angle, 1., decay);
    return vec4<f32>(rgb.r, rgb.g, rgb.b, 1.);
}

fn anti_aliased_color(location: vec2<f32>) -> vec4<f32> {
    let c1 = mandelbrot_color(location + vec2<f32>(-0.125, -0.375));
    let c2 = mandelbrot_color(location + vec2<f32>(0.375, -0.125));
    let c3 = mandelbrot_color(location + vec2<f32>(-0.375, 0.125));
    let c4 = mandelbrot_color(location + vec2<f32>(0.125, 0.375));
    return (c1 + c2 + c3 + c4) / 4.;
}

@compute @workgroup_size(8, 8, 1)
fn checker_board(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    textureStore(output, location,
        anti_aliased_color(vec2<f32>(f32(location.x), f32(location.y))));
}
