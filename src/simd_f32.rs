#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{f64x4, i32x8};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

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
    pub fn unpacklo(self, b: f32x8) -> Self { unsafe { _mm256_unpacklo_ps(self.v, b.v) }.into() }

    #[inline]
    pub fn unpackhi(self, b: f32x8) -> Self { unsafe { _mm256_unpackhi_ps(self.v, b.v) }.into() }

    #[inline]
    pub fn from_slice(a: &[f32]) -> Self {
        if a.len() < 8 {
            panic!("Slice too small!")
        }

        unsafe { Self::from_slice_unchecked(a) }
    }

    #[inline]
    pub unsafe fn from_slice_unchecked(a: &[f32]) -> Self { Self::from_ptr(a.as_ptr()) }

    #[inline]
    pub unsafe fn from_ptr(a: *const f32) -> Self { _mm256_loadu_ps(a).into() }

    #[inline]
    pub unsafe fn gather_ptr<const SCALE: i32>(a: *const f32, idx: i32x8) -> Self {
        _mm256_i32gather_ps::<SCALE>(a, idx.into()).into()
    }

    #[inline]
    pub unsafe fn gather_unchecked<const SCALE: i32>(a: &[f32], idx: i32x8) -> Self {
        Self::gather_ptr::<SCALE>(a.as_ptr(), idx)
    }

    #[inline]
    pub fn to_raw_i32(self) -> i32x8 { unsafe { _mm256_castps_si256(self.v) }.into() }

    #[inline]
    pub fn trunc(self) -> i32x8 { unsafe { _mm256_cvttps_epi32(self.v) }.into() }

    #[inline]
    pub fn floor(self) -> f32x8 { unsafe { _mm256_floor_ps(self.v) }.into() }

    #[inline]
    pub fn to_raw_f64(self) -> f64x4 { unsafe { _mm256_castps_pd(self.v) }.into() }

    #[inline]
    pub fn blend<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm256_blend_ps::<IMM>(self.v, other.v) }.into()
    }

    #[inline]
    pub fn store(self, a: &mut [f32]) {
        if a.len() < 8 {
            panic!("Slice too small!")
        }

        unsafe { self.store_unchecked(a) }
    }

    #[inline]
    pub unsafe fn store_unchecked(self, a: &mut [f32]) { self.store_ptr(a.as_mut_ptr()) }

    #[inline]
    pub unsafe fn store_ptr(self, a: *mut f32) { _mm256_storeu_ps(a, self.v) }

    #[inline]
    pub unsafe fn storeu2_ptr(self, hi: *mut f32, lo: *mut f32) { _mm256_storeu2_m128(hi, lo, self.v) }

    #[inline]
    pub fn hadd(self, other: Self) -> Self { unsafe { _mm256_hadd_ps(self.v, other.v) }.into() }

    #[inline]
    pub fn mul_add(self, mul: Self, add: Self) -> Self { unsafe { _mm256_fmadd_ps(self.v, mul.v, add.v) }.into() }

    #[inline]
    pub fn min(self, other: Self) -> Self { unsafe { _mm256_min_ps(self.v, other.v) }.into() }

    #[inline]
    pub fn max(self, other: Self) -> Self { unsafe { _mm256_max_ps(self.v, other.v) }.into() }

    #[inline]
    pub fn sum(self) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps::<1>(self.v); // (4, 5, 6, 7)
            let lo = _mm256_castps256_ps128(self.v); // (1, 2, 3, 4)
            let sum = _mm_add_ps(lo, hi); // (1 + 4, 2 + 5, 3 + 6, 4 + 7)
            let hid = _mm_movehl_ps(sum, sum); // (3 + 6, 4 + 7, -, -)
            let sum2 = _mm_add_ps(sum, hid); // (1 + 4 + 3 + 6, 2 + 5 + 4 + 7, -, -)
            let hi = _mm_shuffle_ps::<1>(sum2, sum2); // (2 + 5 + 4 + 7, -, -, -)
            let sum = _mm_add_ss(sum2, hi); // (1 + 2 + 3 + 4 + 5 + 6 + 7, -, -, -)
            _mm_cvtss_f32(sum)
        }
    }

    #[inline]
    pub fn extract<const INDEX: i32>(self) -> f32 { f32::from_bits(self.to_raw_i32().extract::<INDEX>() as u32) }
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

impl Sub<f32x8> for f32x8 {
    type Output = f32x8;

    #[inline]
    fn sub(self, rhs: f32x8) -> Self::Output { unsafe { _mm256_sub_ps(self.v, rhs.v) }.into() }
}

impl SubAssign for f32x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { self.v = (*self - rhs).v }
}
