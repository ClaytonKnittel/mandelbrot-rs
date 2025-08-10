
struct fp64 {
    high: f32,
    low: f32,
}

// Divide float number to high and low floats to extend fraction bits
fn split64(a: f32) -> fp64 {
    let c = (f32(1u << 12u) + 1.0) * a;
    let a_big = c - a;
    let a_hi = c * 1.0 - a_big;
    let a_lo = a * 1.0 - a_hi;
    return fp64(a_hi, a_lo);
}

fn fp64_to_f32(a: fp64) -> f32 {
    return a.high + a.low;
}

// Special sum operation when a > b
fn quickTwoSum(a: f32, b: f32) -> fp64 {
    let x = (a + b) * 1.0;
    let b_virt = (x - a) * 1.0;
    let y = b - b_virt;
    return fp64(x, y);
}

fn twoSum(a: f32, b: f32) -> fp64 {
    let x = (a + b);
    let b_virt = (x - a) * 1.0;
    let a_virt = (x - b_virt) * 1.0;
    let b_err = b - b_virt;
    let a_err = a - a_virt;
    let y = a_err + b_err;
    return fp64(x, y);
}

fn twoSub(a: f32, b: f32) -> fp64 {
    let s = (a - b);
    let v = (s * 1.0 - a) * 1.0;
    let err = (a - (s - v) * 1.0) * 1.0 - (b + v);
    return fp64(s, err);
}

fn twoProd(a: f32, b: f32) -> fp64 {
    let x = a * b;
    let a2 = split64(a);
    let b2 = split64(b);
    let err1 = x - (a2.high * b2.high * 1.0) * 1.0;
    let err2 = err1 - (a2.low * b2.high * 1.0) * 1.0;
    let err3 = err2 - (a2.high * b2.low * 1.0) * 1.0;
    let y = a2.low * b2.low - err3;
    return fp64(x, y);
}

fn sum64(a: fp64, b: fp64) -> fp64 {
    var s = twoSum(a.high, b.high);
    var t = twoSum(a.low, b.low);
    s.low += t.high;
    s = quickTwoSum(s.high, s.low);
    s.low += t.low;
    s = quickTwoSum(s.high, s.low);
    return s;
}

fn sub64(a: fp64, b: fp64) -> fp64 {
    var s = twoSub(a.high, b.high);
    var t = twoSub(a.low, b.low);
    s.low += t.high;
    s = quickTwoSum(s.high, s.low);
    s.low += t.low;
    s = quickTwoSum(s.high, s.low);
    return fp64(s.high, s.low);
}

fn mul64(a: fp64, b: fp64) -> fp64 {
    var p = twoProd(a.high, b.high);
    p.low += a.high * b.low;
    p.low += a.low * b.high;
    p = quickTwoSum(p.high, p.low);
    return p;
}

fn pow64(a: fp64, p: u32) -> fp64 {
    var v = a;
    var pow = p;
    var result = fp64(1.0, 0.0);

    while pow > 0u {
        if (pow & 1u) != 0u {
            result = mul64(result, v);
        }
        v = mul64(v, v);
        pow >>= 1u;
    }
    return result;
}

fn lt(a: fp64, b: fp64) -> bool {
    return a.high < b.high || (a.high == b.high && a.low < b.low);
}

struct Uniforms {
    time: u32,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

const MAX_ITERS: u32 = 2000;
const DIVERGENCE_BOUND: f32 = 1.e5;

const POINT: vec2<f32> = vec2<f32>(0.743643887037151, 0.131825904205330);

struct Complex {
    x: fp64,
    y: fp64,
}

fn complex_add(a: Complex, b: Complex) -> Complex {
    return Complex(sum64(a.x, b.x), sum64(a.y, b.y));
}

fn complex_sq(z: Complex) -> Complex {
    return Complex(sub64(mul64(z.x, z.x), mul64(z.y, z.y)),
        mul64(split64(2.), mul64(z.x, z.y)));
}

fn complex_mag2(z: Complex) -> fp64 {
    return sum64(mul64(z.x, z.x), mul64(z.y, z.y));
}

fn divergence(c: Complex) -> f32 {
    var z: Complex = Complex(split64(0.), split64(0.));
    let limit2 = split64(DIVERGENCE_BOUND * DIVERGENCE_BOUND);

    for (var i = 0u; i < MAX_ITERS; i++) {
        z = complex_add(complex_sq(z), c);
        let mag = complex_mag2(z);
        if lt(limit2, mag) {
            return f32(i) - log(log(fp64_to_f32(mag)) / log(DIVERGENCE_BOUND)) / log(2);
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

fn to_mandel_coords(location: vec2<f32>) -> array<fp64, 2> {
    let f = pow64(split64(0.99), uniforms.time);
    return array<fp64, 2>(sub64(mul64(split64(2. * location.x / 1280. - 1.), f), split64(POINT.x)),
        sub64(mul64(split64(2. * location.y / 720. - 1.), f), split64(POINT.y)));
}

fn mandelbrot_color(pos: vec2<f32>) -> vec4<f32> {
    let mandel_pos = to_mandel_coords(pos);
    let c: Complex = Complex(mandel_pos[0], mandel_pos[1]);
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
