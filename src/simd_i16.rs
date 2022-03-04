#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{i32x8, i64x4};
use std::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Mul, MulAssign, Sub, SubAssign};

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
pub struct i16x16 {
    v: __m256i,
}

impl i16x16 {
    #[inline]
    pub fn new(
        v1: i16,
        v2: i16,
        v3: i16,
        v4: i16,
        v5: i16,
        v6: i16,
        v7: i16,
        v8: i16,
        v9: i16,
        v10: i16,
        v11: i16,
        v12: i16,
        v13: i16,
        v14: i16,
        v15: i16,
        v16: i16,
    ) -> Self {
        Self {
            v: unsafe { _mm256_set_epi16(v16, v15, v14, v13, v12, v11, v10, v9, v8, v7, v6, v5, v4, v3, v2, v1) },
        }
    }

    #[inline]
    pub fn splat(v: i16) -> i16x16 { unsafe { _mm256_set1_epi16(v) }.into() }

    #[inline]
    pub fn from_slice(a: &[i16]) -> Self {
        if a.len() < 8 {
            panic!("Slice too small!")
        }

        unsafe { Self::from_slice_unchecked(a) }
    }

    #[inline]
    pub unsafe fn from_slice_unchecked(a: &[i16]) -> Self { Self::from_ptr(a.as_ptr()) }

    #[inline]
    pub unsafe fn from_ptr(a: *const i16) -> Self { _mm256_loadu_si256(a as *const __m256i).into() }

    #[inline]
    pub fn blend<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm256_blend_epi16::<IMM>(self.v, other.v) }.into()
    }

    #[inline]
    pub fn get<const IDX: i32>(self) -> i16 { unsafe { _mm256_extract_epi16::<IDX>(self.v) as i16 } }

    #[inline]
    pub fn set<const IDX: i32>(self, v: i16) -> Self { unsafe { _mm256_insert_epi16::<IDX>(self.v, v) }.into() }

    #[inline]
    pub fn to_raw_i32(self) -> i32x8 { self.v.into() }

    #[inline]
    pub fn to_raw_i64(self) -> i64x4 { self.v.into() }

    #[inline]
    pub fn store(self, a: &mut [i16]) {
        if a.len() < 16 {
            panic!("Slice too small!")
        }

        unsafe { self.store_unchecked(a) }
    }

    #[inline]
    pub unsafe fn store_unchecked(self, a: &mut [i16]) { self.store_ptr(a.as_mut_ptr()) }

    #[inline]
    pub unsafe fn store_ptr(self, a: *mut i16) { _mm256_storeu_si256(a as *mut __m256i, self.v) }

    #[inline]
    pub fn extract<const INDEX: i32>(self) -> i16 { unsafe { _mm256_extract_epi16(self.v, INDEX) as i16 } }
}

impl From<__m256i> for i16x16 {
    #[inline]
    fn from(v: __m256i) -> Self { Self { v } }
}

impl From<i16x16> for __m256i {
    #[inline]
    fn from(v: i16x16) -> Self { v.v }
}

impl Mul<i16x16> for i16x16 {
    type Output = i16x16;

    #[inline]
    fn mul(self, rhs: i16x16) -> Self::Output { unsafe { _mm256_mullo_epi16(self.v, rhs.v) }.into() }
}

impl MulAssign<i16x16> for i16x16 {
    #[inline]
    fn mul_assign(&mut self, rhs: i16x16) { self.v = (*self * rhs).v }
}

impl Add<i16x16> for i16x16 {
    type Output = i16x16;

    #[inline]
    fn add(self, rhs: i16x16) -> Self::Output { unsafe { _mm256_add_epi16(self.v, rhs.v) }.into() }
}

impl AddAssign for i16x16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { self.v = (*self + rhs).v }
}

impl Sub<i16x16> for i16x16 {
    type Output = i16x16;

    #[inline]
    fn sub(self, rhs: i16x16) -> Self::Output { unsafe { _mm256_sub_epi16(self.v, rhs.v) }.into() }
}

impl SubAssign for i16x16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { self.v = (*self - rhs).v }
}

impl BitAnd<i16x16> for i16x16 {
    type Output = i16x16;

    #[inline]
    fn bitand(self, rhs: i16x16) -> Self::Output { unsafe { _mm256_and_si256(self.v, rhs.v) }.into() }
}

impl BitAndAssign for i16x16 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) { self.v = (*self & rhs).v }
}

impl BitOr<i16x16> for i16x16 {
    type Output = i16x16;

    #[inline]
    fn bitor(self, rhs: i16x16) -> Self::Output { unsafe { _mm256_or_si256(self.v, rhs.v) }.into() }
}

impl BitOrAssign for i16x16 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) { self.v = (*self | rhs).v }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
pub struct i16x8 {
    v: __m128i,
}

impl i16x8 {
    #[inline]
    pub fn new(v1: i16, v2: i16, v3: i16, v4: i16, v5: i16, v6: i16, v7: i16, v8: i16) -> Self {
        Self {
            v: unsafe { _mm_set_epi16(v8, v7, v6, v5, v4, v3, v2, v1) },
        }
    }

    #[inline]
    pub fn get<const IDX: i32>(self) -> i16 { unsafe { _mm_extract_epi16::<IDX>(self.v) as i16 } }

    #[inline]
    pub fn set<const IDX: i32>(self, v: i16) -> Self { unsafe { _mm_insert_epi16::<IDX>(self.v, v as i32) }.into() }
}

impl From<__m128i> for i16x8 {
    #[inline]
    fn from(v: __m128i) -> Self { Self { v } }
}

impl From<i16x8> for __m128i {
    #[inline]
    fn from(v: i16x8) -> Self { v.v }
}

impl Mul<i16x8> for i16x8 {
    type Output = i16x8;

    #[inline]
    fn mul(self, rhs: i16x8) -> Self::Output { unsafe { _mm_mullo_epi16(self.v, rhs.v) }.into() }
}

impl MulAssign<i16x8> for i16x8 {
    #[inline]
    fn mul_assign(&mut self, rhs: i16x8) { self.v = (*self * rhs).v }
}

impl Add<i16x8> for i16x8 {
    type Output = i16x8;

    #[inline]
    fn add(self, rhs: i16x8) -> Self::Output { unsafe { _mm_add_epi16(self.v, rhs.v) }.into() }
}

impl AddAssign for i16x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { self.v = (*self + rhs).v }
}

impl Sub<i16x8> for i16x8 {
    type Output = i16x8;

    #[inline]
    fn sub(self, rhs: i16x8) -> Self::Output { unsafe { _mm_sub_epi16(self.v, rhs.v) }.into() }
}

impl SubAssign for i16x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { self.v = (*self - rhs).v }
}
