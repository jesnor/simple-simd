#![allow(clippy::too_many_arguments)]

mod simd_f32;
mod simd_f64;
mod simd_i32;
mod simd_i64;

pub use simd_f32::f32x8;
pub use simd_f64::f64x4;
pub use simd_i32::i32x4;
pub use simd_i32::i32x8;
pub use simd_i64::i64x4;
