struct Uniforms {
    time: u32,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

const MAX_ITERS: u32 = 2000;
const DIVERGENCE_BOUND: f32 = 1.e5;

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

fn mandelbrot_color(x: f32, y: f32) -> vec4<f32> {
    let c: Complex = Complex(x, y);
    let d = divergence(c);
    if d < 0. {
        return vec4<f32>(0., 0., 0., 1.);
    }

    let q = d / f32(MAX_ITERS);
    let r = sqrt(q);
    return vec4<f32>(r, q, q * q, 1.);
}

@compute @workgroup_size(8, 8, 1)
fn checker_board(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let f = pow(0.5, f32(uniforms.time) / 200.);
    let x = (2. * f32(invocation_id.x) / 1280. - 1.) * f - 0.5;
    let y = (2. * f32(invocation_id.y) / 720. - 1.) * f;

    textureStore(output, location, mandelbrot_color(x, y));
}
