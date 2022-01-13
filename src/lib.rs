#![allow(clippy::too_many_arguments, clippy::from_over_into, clippy::ptr_offset_with_cast)]

mod simd_f32;
mod simd_f64;
mod simd_i32;

pub use simd_f32::f32x8;
pub use simd_f64::f64x4;
pub use simd_i32::i32x8;
