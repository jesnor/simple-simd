#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{f32x8, i32x8, i64x4};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
pub struct f64x4 {
    v: __m256d,
}

impl f64x4 {
    #[inline]
    pub fn new(v1: f64, v2: f64, v3: f64, v4: f64) -> Self {
        Self {
            v: unsafe { _mm256_set_pd(v4, v3, v2, v1) },
        }
    }

    #[inline]
    pub fn splat(v: f64) -> Self { unsafe { _mm256_set1_pd(v) }.into() }

    #[inline]
    pub fn permute<const IMM: i32>(self) -> Self { unsafe { _mm256_permute_pd::<IMM>(self.v) }.into() }

    #[inline]
    pub fn permute4x64<const IMM: i32>(self) -> Self { unsafe { _mm256_permute4x64_pd::<IMM>(self.v) }.into() }

    #[inline]
    pub fn permute_var(self, idx: i32x8) -> Self { unsafe { _mm256_permutevar_pd(self.v, idx.into()) }.into() }

    #[inline]
    pub fn from_slice(a: &[f64]) -> Self {
        if a.len() < 4 {
            panic!("Slice too small!")
        }

        unsafe { Self::from_slice_unchecked(a) }
    }

    #[inline]
    pub unsafe fn from_slice_unchecked(a: &[f64]) -> Self { Self::from_ptr(a.as_ptr()) }

    #[inline]
    pub unsafe fn from_ptr(a: *const f64) -> Self { _mm256_loadu_pd(a).into() }

    #[inline]
    pub fn to_raw_i32(self) -> i32x8 { unsafe { _mm256_castpd_si256(self.v) }.into() }

    #[inline]
    pub fn to_raw_i64(self) -> i64x4 { unsafe { _mm256_castpd_si256(self.v) }.into() }

    #[inline]
    pub fn to_raw_f32(self) -> f32x8 { unsafe { _mm256_castpd_ps(self.v) }.into() }

    #[inline]
    pub fn blend<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm256_blend_pd::<IMM>(self.v, other.v) }.into()
    }

    #[inline]
    pub fn store(self, a: &mut [f64]) {
        if a.len() < 4 {
            panic!("Slice too small!")
        }

        unsafe { self.store_unchecked(a) }
    }

    #[inline]
    pub unsafe fn store_unchecked(self, a: &mut [f64]) { self.store_ptr(a.as_mut_ptr()) }

    #[inline]
    pub unsafe fn store_ptr(self, a: *mut f64) { _mm256_storeu_pd(a, self.v) }

    #[inline]
    pub fn hadd(self, other: Self) -> Self { unsafe { _mm256_hadd_pd(self.v, other.v) }.into() }

    #[inline]
    pub fn min(self, other: Self) -> Self { unsafe { _mm256_min_pd(self.v, other.v) }.into() }

    #[inline]
    pub fn max(self, other: Self) -> Self { unsafe { _mm256_max_pd(self.v, other.v) }.into() }

    #[inline]
    pub fn extract<const INDEX: i32>(self) -> f64 { f64::from_bits(self.to_raw_i64().extract::<INDEX>() as u64) }
}

impl From<__m256d> for f64x4 {
    #[inline]
    fn from(v: __m256d) -> Self { Self { v } }
}

impl From<f64x4> for __m256d {
    #[inline]
    fn from(v: f64x4) -> Self { v.v }
}

impl Mul<f64x4> for f64x4 {
    type Output = f64x4;

    #[inline]
    fn mul(self, rhs: f64x4) -> Self::Output { unsafe { _mm256_mul_pd(self.v, rhs.v) }.into() }
}

impl MulAssign<f64x4> for f64x4 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64x4) { self.v = (*self * rhs).v }
}

impl Add<f64x4> for f64x4 {
    type Output = f64x4;

    #[inline]
    fn add(self, rhs: f64x4) -> Self::Output { unsafe { _mm256_add_pd(self.v, rhs.v) }.into() }
}

impl AddAssign for f64x4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { self.v = (*self + rhs).v }
}

impl Sub<f64x4> for f64x4 {
    type Output = f64x4;

    #[inline]
    fn sub(self, rhs: f64x4) -> Self::Output { unsafe { _mm256_sub_pd(self.v, rhs.v) }.into() }
}

impl SubAssign for f64x4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { self.v = (*self - rhs).v }
}
