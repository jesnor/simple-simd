#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{f32x8, i16x16, i64x4};
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
    pub fn splat(v: i32) -> i32x8 { unsafe { _mm256_set1_epi32(v) }.into() }

    #[inline]
    pub fn from_slice(a: &[i32]) -> Self {
        if a.len() < 8 {
            panic!("Slice too small!")
        }

        unsafe { Self::from_slice_unchecked(a) }
    }

    #[inline]
    pub unsafe fn from_slice_unchecked(a: &[i32]) -> Self { Self::from_ptr(a.as_ptr()) }

    #[inline]
    pub unsafe fn from_ptr(a: *const i32) -> Self { _mm256_loadu_si256(a as *const __m256i).into() }

    #[inline]
    pub fn blend<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm256_blend_epi32::<IMM>(self.v, other.v) }.into()
    }

    #[inline]
    pub fn get<const IDX: i32>(self) -> i32 { unsafe { _mm256_extract_epi32::<IDX>(self.v) } }

    #[inline]
    pub fn set<const IDX: i32>(self, v: i32) -> Self { unsafe { _mm256_insert_epi32::<IDX>(self.v, v) }.into() }

    #[inline]
    pub fn packs(self, other: i32x8) -> i16x16 { unsafe { _mm256_packs_epi32(self.v, other.v) }.into() }

    #[inline]
    pub fn to_raw_f32(self) -> f32x8 { unsafe { _mm256_castsi256_ps(self.v) }.into() }

    #[inline]
    pub fn to_raw_i16(self) -> i16x16 { self.v.into() }

    #[inline]
    pub fn to_raw_i64(self) -> i64x4 { self.v.into() }

    #[inline]
    pub fn to_f32(self) -> f32x8 { unsafe { _mm256_cvtepi32_ps(self.v) }.into() }

    #[inline]
    pub fn store(self, a: &mut [i32]) {
        if a.len() < 8 {
            panic!("Slice too small!")
        }

        unsafe { self.store_unchecked(a) }
    }

    #[inline]
    pub unsafe fn store_unchecked(self, a: &mut [i32]) { self.store_ptr(a.as_mut_ptr()) }

    #[inline]
    pub unsafe fn store_ptr(self, a: *mut i32) { _mm256_storeu_si256(a as *mut __m256i, self.v) }
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
    fn mul(self, rhs: i32x8) -> Self::Output { unsafe { _mm256_mullo_epi32(self.v, rhs.v) }.into() }
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

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
pub struct i32x4 {
    v: __m128i,
}

impl i32x4 {
    #[inline]
    pub fn new(v1: i32, v2: i32, v3: i32, v4: i32) -> Self {
        Self {
            v: unsafe { _mm_set_epi32(v4, v3, v2, v1) },
        }
    }

    #[inline]
    pub fn get<const IDX: i32>(self) -> i32 { unsafe { _mm_extract_epi32::<IDX>(self.v) } }

    #[inline]
    pub fn set<const IDX: i32>(self, v: i32) -> Self { unsafe { _mm_insert_epi32::<IDX>(self.v, v) }.into() }
}

impl From<__m128i> for i32x4 {
    #[inline]
    fn from(v: __m128i) -> Self { Self { v } }
}

impl From<i32x4> for __m128i {
    #[inline]
    fn from(v: i32x4) -> Self { v.v }
}

impl Mul<i32x4> for i32x4 {
    type Output = i32x4;

    #[inline]
    fn mul(self, rhs: i32x4) -> Self::Output { unsafe { _mm_mullo_epi32(self.v, rhs.v) }.into() }
}

impl MulAssign<i32x4> for i32x4 {
    #[inline]
    fn mul_assign(&mut self, rhs: i32x4) { self.v = (*self * rhs).v }
}

impl Add<i32x4> for i32x4 {
    type Output = i32x4;

    #[inline]
    fn add(self, rhs: i32x4) -> Self::Output { unsafe { _mm_add_epi32(self.v, rhs.v) }.into() }
}

impl AddAssign for i32x4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { self.v = (*self + rhs).v }
}

impl Sub<i32x4> for i32x4 {
    type Output = i32x4;

    #[inline]
    fn sub(self, rhs: i32x4) -> Self::Output { unsafe { _mm_sub_epi32(self.v, rhs.v) }.into() }
}

impl SubAssign for i32x4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { self.v = (*self - rhs).v }
}
