#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{f64x4, i32x8};
use std::ops::{Add, AddAssign, Mul, MulAssign};

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
pub struct f32x8 {
    v: __m256,
}

impl f32x8 {
    #[inline]
    pub fn new(v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32, v8: f32) -> Self {
        Self {
            v: unsafe { _mm256_set_ps(v8, v7, v6, v5, v4, v3, v2, v1) },
        }
    }

    #[inline]
    pub fn splat(v: f32) -> Self { unsafe { _mm256_set1_ps(v) }.into() }

    #[inline]
    pub fn permute<const IMM: i32>(self) -> Self { unsafe { _mm256_permute_ps::<IMM>(self.v) }.into() }

    #[inline]
    pub fn permute4x64<const IMM: i32>(self) -> Self { self.to_raw_f64().permute4x64::<IMM>().to_raw_f32() }

    #[inline]
    pub fn permute_var(self, idx: i32x8) -> Self { unsafe { _mm256_permutevar8x32_ps(self.v, idx.into()) }.into() }

    #[inline]
    pub unsafe fn from_slice_unchecked(a: &[f32]) -> Self { Self::from_ptr(a.as_ptr()) }

    #[inline]
    pub unsafe fn from_ptr(a: *const f32) -> Self { _mm256_loadu_ps(a).into() }

    #[inline]
    pub unsafe fn gather<const SCALE: i32>(a: *const f32, idx: i32x8) -> Self {
        _mm256_i32gather_ps::<SCALE>(a, idx.into()).into()
    }

    #[inline]
    pub fn to_raw_i32(self) -> i32x8 { unsafe { _mm256_castps_si256(self.v) }.into() }

    #[inline]
    pub fn to_raw_f64(self) -> f64x4 { unsafe { _mm256_castps_pd(self.v) }.into() }

    #[inline]
    pub fn blend<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm256_blend_ps::<IMM>(self.v, other.v) }.into()
    }

    #[inline]
    pub unsafe fn store_unchecked(self, a: &mut [f32]) { self.store_ptr(a.as_mut_ptr()) }

    #[inline]
    pub unsafe fn store_ptr(self, a: *mut f32) { _mm256_storeu_ps(a, self.v) }

    #[inline]
    pub fn hadd(self, other: Self) -> Self { unsafe { _mm256_hadd_ps(self.v, other.v) }.into() }

    #[inline]
    pub fn mul_add(self, mul: Self, add: Self) -> Self { unsafe { _mm256_fmadd_ps(self.v, mul.v, add.v) }.into() }
}

impl From<__m256> for f32x8 {
    #[inline]
    fn from(v: __m256) -> Self { Self { v } }
}

impl From<f32x8> for __m256 {
    #[inline]
    fn from(v: f32x8) -> Self { v.v }
}

impl Mul<f32x8> for f32x8 {
    type Output = f32x8;

    #[inline]
    fn mul(self, rhs: f32x8) -> Self::Output { unsafe { _mm256_mul_ps(self.v, rhs.v) }.into() }
}

impl MulAssign<f32x8> for f32x8 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32x8) { self.v = (*self * rhs).v }
}

impl Add<f32x8> for f32x8 {
    type Output = f32x8;

    #[inline]
    fn add(self, rhs: f32x8) -> Self::Output { unsafe { _mm256_add_ps(self.v, rhs.v) }.into() }
}

impl AddAssign for f32x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { self.v = (*self + rhs).v }
}
