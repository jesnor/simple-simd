#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::f32x8;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
pub struct i32x8 {
    v: __m256i,
}

impl i32x8 {
    #[inline]
    pub fn new(v1: i32, v2: i32, v3: i32, v4: i32, v5: i32, v6: i32, v7: i32, v8: i32) -> Self {
        Self {
            v: unsafe { _mm256_set_epi32(v8, v7, v6, v5, v4, v3, v2, v1) },
        }
    }

    #[inline]
    pub fn get<const IDX: i32>(self) -> i32 { unsafe { _mm256_extract_epi32::<IDX>(self.v) } }

    #[inline]
    pub fn set<const IDX: i32>(self, v: i32) -> Self { unsafe { _mm256_insert_epi32::<IDX>(self.v, v) }.into() }

    #[inline]
    pub fn to_raw_f32(self) -> f32x8 { unsafe { _mm256_castsi256_ps(self.v) }.into() }

    #[inline]
    pub fn to_f32(self) -> f32x8 { unsafe { _mm256_cvtepi32_ps(self.v) }.into() }
}

impl From<__m256i> for i32x8 {
    #[inline]
    fn from(v: __m256i) -> Self { Self { v } }
}

impl From<i32x8> for __m256i {
    #[inline]
    fn from(v: i32x8) -> Self { v.v }
}

impl Mul<i32x8> for i32x8 {
    type Output = i32x8;

    #[inline]
    fn mul(self, rhs: i32x8) -> Self::Output { unsafe { _mm256_mul_epi32(self.v, rhs.v) }.into() }
}

impl MulAssign<i32x8> for i32x8 {
    #[inline]
    fn mul_assign(&mut self, rhs: i32x8) { self.v = (*self * rhs).v }
}

impl Add<i32x8> for i32x8 {
    type Output = i32x8;

    #[inline]
    fn add(self, rhs: i32x8) -> Self::Output { unsafe { _mm256_add_epi32(self.v, rhs.v) }.into() }
}

impl AddAssign for i32x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { self.v = (*self + rhs).v }
}

impl Sub<i32x8> for i32x8 {
    type Output = i32x8;

    #[inline]
    fn sub(self, rhs: i32x8) -> Self::Output { unsafe { _mm256_sub_epi32(self.v, rhs.v) }.into() }
}

impl SubAssign for i32x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { self.v = (*self - rhs).v }
}
