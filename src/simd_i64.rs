#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{f32x8, i16x16};
use std::ops::{Add, AddAssign, Sub, SubAssign};

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
pub struct i64x4 {
    v: __m256i,
}

impl i64x4 {
    #[inline]
    pub fn new(v1: i64, v2: i64, v3: i64, v4: i64) -> Self {
        Self {
            v: unsafe { _mm256_set_epi64x(v4, v3, v2, v1) },
        }
    }

    #[inline]
    pub fn get<const IDX: i32>(self) -> i64 { unsafe { _mm256_extract_epi64::<IDX>(self.v) } }

    #[inline]
    pub fn set<const IDX: i32>(self, v: i64) -> Self { unsafe { _mm256_insert_epi64::<IDX>(self.v, v) }.into() }

    #[inline]
    pub fn permute<const IMM: i32>(self) -> Self { unsafe { _mm256_permute4x64_epi64::<IMM>(self.v) }.into() }

    #[inline]
    pub unsafe fn gather_ptr<const SCALE: i32>(a: *const i64, idx: i64x4) -> Self {
        _mm256_i64gather_epi64::<SCALE>(a, idx.into()).into()
    }

    #[inline]
    pub unsafe fn gather_unchecked<const SCALE: i32>(a: &[i64], idx: i64x4) -> Self {
        Self::gather_ptr::<SCALE>(a.as_ptr(), idx)
    }

    #[inline]
    pub fn to_raw_i16(self) -> i16x16 { self.v.into() }

    #[inline]
    pub fn to_raw_f32(self) -> f32x8 { unsafe { _mm256_castsi256_ps(self.v) }.into() }

    #[inline]
    pub fn store(self, a: &mut [i64]) {
        if a.len() < 4 {
            panic!("Slice too small!")
        }

        unsafe { self.store_unchecked(a) }
    }

    #[inline]
    pub unsafe fn store_unchecked(self, a: &mut [i64]) { self.store_ptr(a.as_mut_ptr()) }

    #[inline]
    pub unsafe fn store_ptr(self, a: *mut i64) { _mm256_storeu_si256(a as *mut __m256i, self.v) }
}

impl From<__m256i> for i64x4 {
    #[inline]
    fn from(v: __m256i) -> Self { Self { v } }
}

impl From<i64x4> for __m256i {
    #[inline]
    fn from(v: i64x4) -> Self { v.v }
}

impl Add<i64x4> for i64x4 {
    type Output = i64x4;

    #[inline]
    fn add(self, rhs: i64x4) -> Self::Output { unsafe { _mm256_add_epi64(self.v, rhs.v) }.into() }
}

impl AddAssign for i64x4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { self.v = (*self + rhs).v }
}

impl Sub<i64x4> for i64x4 {
    type Output = i64x4;

    #[inline]
    fn sub(self, rhs: i64x4) -> Self::Output { unsafe { _mm256_sub_epi64(self.v, rhs.v) }.into() }
}

impl SubAssign for i64x4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { self.v = (*self - rhs).v }
}
