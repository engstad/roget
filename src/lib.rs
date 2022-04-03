#![feature(portable_simd)]
#![allow(clippy::type_complexity)]
#![allow(clippy::blocks_in_if_conditions)]

use std::{collections::HashSet, num::NonZeroU8};
use core::simd::*;

mod solver;
pub use solver::{Rank, Solver};

include!(concat!(env!("OUT_DIR"), "/dictionary.rs"));

/// Store the word as a SIMD `u8x8`, which has the same layout as a
/// `[u8; 8]`, but with the same alignment restrictions as a `u64`.
/// In order to reduce cost for converting it into the "n-th" character,
/// we also normalize it so that 'a' = 0, 'b' = 1, ..., 'z' = 25.
#[derive(PartialEq, Eq, Copy, Clone, Default, Hash, Debug)]
pub struct Word(u8x8);

impl Word {
    /// Return the letter from an ASCII byte.
    /// E.g.: `letter(b'a') == 0` and `letter(b'b') == 1`.
    const fn letter(c: u8) -> u8 {
        c - b'a'
    }

    /// Construct a `Word` from a string-slice `s`.
    pub fn new(s: &str) -> Self {
        assert!(s.len() == 5);
        let mut w: [u8; 8] = [0u8; 8];
        for (w, c) in w.iter_mut().zip(s.bytes()) {
            *w = Word::letter(c);
        }
        Self(u8x8::from_array(w))
    }

    // Note: const functions can't appear in the From trait.
    pub const fn from(s: &[u8; 5]) -> Self {
        // Work-around for constexpr problems.
        let l = Word::letter;
        Self(u8x8::from_array([l(s[0]), l(s[1]), l(s[2]), l(s[3]),
                               l(s[4]), 0, 0, 0]))
    }

    /// Return the letters in the `Word` where 'a' is 0, etc.
    fn letters(&self) -> [u8; 5] {
        let a: [u8; 8] = self.0.into();
        [a[0], a[1], a[2], a[3], a[4]]
    }

    /// Return the letters in ASCII in the `Word`.
    fn ascii(&self) -> [u8; 5] {
        let a: [u8; 8] = self.0.into();
        [a[0] + b'a', a[1] + b'a', a[2] + b'a', a[3] + b'a', a[4] + b'a']
    }

    fn simd(&self) -> u8x8 { self.0 }
}

impl std::fmt::Display for Word {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", std::str::from_utf8(&self.ascii()).unwrap())
    }
}

pub struct Wordle {
    dictionary: HashSet<Word>,
}

impl Default for Wordle {
    fn default() -> Self {
        Self::new()
    }
}

impl Wordle {
    pub fn new() -> Self {
        Self {
            dictionary: HashSet::from_iter(DICTIONARY.iter().map(|(word, _)| Word::new(word))),
        }
    }

    pub fn play<G: Guesser>(&self, answer: Word, mut guesser: G) -> Option<usize> {

        let mut history = Vec::new();
        // Wordle only allows six guesses.
        // We allow more to avoid chopping off the score distribution for stats purposes.
        for i in 1..=32 {
            let guess = guesser.guess(&history);
            if guess == answer {
                guesser.finish(i);
                return Some(i);
            }
            assert!(
                self.dictionary.contains(&guess),
                "guess '{}' is not in the dictionary",
                guess
            );
            let pattern = Pattern::compute(answer, guess);
            history.push(Guess {
                word: guess,
                patt: pattern
            });
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Correctness {
    /// Gray
    Wrong = 0,
    /// Yellow
    Misplaced = 1,
    /// Green
    Correct = 2,
}

/// A `Pattern` is the result of matching two `Word` values.
///
/// If a letter is in the correct position, the corresponding `correct` bit is set to 1.
/// If a letter is not correct, but occurs somewhere in the word, the `misplaced` bit is set to 1.
///
/// Note that comparing the guess AAAXX to the answer YYYAA results in MMwww, not MMMww, because
/// the answer has two A's, not three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pattern {
    correct: u8,
    misplaced: u8
}

impl Pattern {
    pub fn new(e: [Correctness; 5]) -> Self {
        let mut c = 0u8;
        let mut m = 0u8;
        for (i, e) in e.iter().enumerate() {
            match e {
                Correctness::Correct => { c |= 1 << i; }
                Correctness::Misplaced => { m |= 1 << i; }
                _ => {}
            }
        }
        Self { correct: c, misplaced: m }
    }

    fn get(&self, i: usize) -> Correctness {
        assert!(i < 5);
        if (self.correct >> i) & 1 != 0 {
            Correctness::Correct
        } else if (self.misplaced >> i) & 1 != 0 {
            Correctness::Misplaced
        } else {
            Correctness::Wrong
        }
    }

    fn is_misplaced(letter: u8, answer: Word, used: &mut [bool; 5]) -> bool {
        answer.letters().iter().enumerate().any(|(i, a)| {
            if *a == letter && !used[i] {
                used[i] = true;
                return true;
            }
            false
        })
    }

    //
    // This compiles down to about 50 or so SSE instructions on Intel.
    //
    pub fn compute(answer: Word, guess: Word) -> Self {
        use Which::*;

        // Array indexed by lowercase ascii letters
        let a = answer.simd();
        let g = guess.simd();

        // So, we can get P(i), the number of places this letter can go to.
        // Note, that we also need to know the number of occurances, because
        // if there are two B's in the guess, and one other B to place it,
        // then only one of the B's should be yellow.
        //
        // We calculate the number of neighbors to the left, using parallel compares
        // and shuffles. E.g., assume that we guess ABBAS. We shift in the neighbors
        // to our lanes, and compare it:
        //
        //                 ABBAS
        //          _ABBA: _0100 (neighbor to the left)
        //          __ABB: __000 (neighbor 2 to the left)
        //          ___AB: ___10 (neighbor 3 to the left)
        //          ____A: ____0 (neighbor 4 to the left)
        //                 00110 (total left neighbors)
        //
        // If a letter can go to one place, but it has a neighbor to the left,
        // then it is not a yellow. So, let O(i) = 1 + N(i), then it is yellow
        // if the number of places the letter can go P(i) is equal to or greater
        // than O(i):
        //
        //     Y(i) = P(i) >= O(i).

        //
        // Calculate the number of neighbors to the left. Note that we must
        // exclude characters that were guessed correct, because we can not
        // move into an already correct position.
        //
        let cor = a.lanes_eq(g);
        let lhs = Simd::splat(27);
        let rhs = Simd::splat(28);

        let lg = cor.select(lhs, g);

        let g0 = cor.select(rhs, g);
        let g1 = simd_swizzle!(g0, rhs, [Second(0), First(0),  First(1),  First(2),  First(3), Second(0), Second(0), Second(0)]);
        let g2 = simd_swizzle!(g0, rhs, [Second(0), Second(0), First(0),  First(1),  First(2), Second(0), Second(0), Second(0)]);
        let g3 = simd_swizzle!(g0, rhs, [Second(0), Second(0), Second(0), First(0),  First(1), Second(0), Second(0), Second(0)]);
        let g4 = simd_swizzle!(g0, rhs, [Second(0), Second(0), Second(0), Second(0), First(0), Second(0), Second(0), Second(0)]);

        let no = Simd::splat(1) - (lg.lanes_eq(g1).to_int() +
                                   lg.lanes_eq(g2).to_int() +
                                   lg.lanes_eq(g3).to_int() +
                                   lg.lanes_eq(g4).to_int());

        // Calculate the number of places the guess can go to be correct.
        // We make sure that correct places are excluded.
        let a0 = cor.select(lhs, a);
        let a1 = simd_swizzle!(a0, [1, 2, 3, 4, 0, 7, 7, 7]);
        let a2 = simd_swizzle!(a0, [2, 3, 4, 0, 1, 7, 7, 7]);
        let a3 = simd_swizzle!(a0, [3, 4, 0, 1, 2, 7, 7, 7]);
        let a4 = simd_swizzle!(a0, [4, 0, 1, 2, 3, 7, 7, 7]);

        let np = -(a1.lanes_eq(g0).to_int() +
                   a2.lanes_eq(g0).to_int() +
                   a3.lanes_eq(g0).to_int() +
                   a4.lanes_eq(g0).to_int());

        let mp = np.lanes_ge(no);

        let correct = cor.to_bitmask() & 0x1f;
        let misplaced = mp.to_bitmask() & 0x1f & !correct;

        Self { correct, misplaced }
    }
}

pub const MAX_MASK_ENUM: usize = 3 * 3 * 3 * 3 * 3;

/// A wrapper type for `[Correctness; 5]` packed into a single byte with a niche.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
// The NonZeroU8 here lets the compiler know that we're not using the value `0`, and that `0` can
// therefore be used to represent `None` for `Option<PackedPattern>`.
pub struct PackedPattern(NonZeroU8);

impl From<Pattern> for PackedPattern {
    fn from(p: Pattern) -> Self {
        let c = p.correct;
        let m = p.misplaced;

        // Converting two 5-bit numbers to a trit, i.e. calculating
        //  t = (c_4 * 3^4 + c_3 * 3^3 + c_2 * 3^2 + c_1 * 3^1 + c_0)
        // but instead of calculating c_k = (c >> k) & 1, etc., use
        //  t = (c_4 * 2^4 + c_3 * 2^3 + c_2 * 2^2 + c_1 * 2^1 + c_0) +
        //      (c_4 * 2^3 + c_3 * 2^2 + c_2 * 2^1 + c_1 * 2^0)       +
        //      (c_4 * 2^2 + c_3 * 2^1 + c_2 * 2^0) * 3               +
        //      (c_4 * 2^1 + c_3 * 2^0) * 9                           +
        //      (c_4 * 2^0) * 27,
        //
        let ct = c + (c>>1) + 3*(c>>2) + 9*(c>>3) + 27*(c>>4);
        let mt = m + (m>>1) + 3*(m>>2) + 9*(m>>3) + 27*(m>>4);

        let packed = 2*ct + mt;

        Self(NonZeroU8::new(packed + 1).unwrap())
    }
}

impl From<PackedPattern> for u8 {
    fn from(this: PackedPattern) -> Self {
        this.0.get() - 1
    }
}

pub struct Guess {
    pub word: Word,
    pub patt: Pattern
}

impl Guess {
    pub fn matches(&self, word: Word) -> bool {
        // Check if the guess would be possible to observe when `word` is the correct answer.
        // This is equivalent to
        //     Pattern::compute(word, &self.word) == self.patt
        // without _necessarily_ computing the full mask for the tested word
        let mut used = [false; 5];

        // Check Correct letters
        for (i, (a, g)) in word.letters().into_iter().zip(self.word.letters().into_iter()).enumerate() {
            if a == g {
                if self.patt.get(i) != Correctness::Correct {
                    return false;
                }
                used[i] = true;
            } else if self.patt.get(i) == Correctness::Correct {
                return false;
            }
        }

        // Check Misplaced letters
        for (i, g) in self.word.letters().into_iter().enumerate() {
            let e = self.patt.get(i);
            if e == Correctness::Correct {
                continue;
            }
            if Pattern::is_misplaced(g, word, &mut used) != (e == Correctness::Misplaced) {
                return false;
            }
        }

        // The rest will be all correctly Wrong letters
        true
    }
}

pub trait Guesser {
    fn guess(&mut self, history: &[Guess]) -> Word;
    fn finish(&self, _guesses: usize) {}
}

impl Guesser for fn(history: &[Guess]) -> Word {
    fn guess(&mut self, history: &[Guess]) -> Word {
        (*self)(history)
    }
}

#[cfg(test)]
macro_rules! guesser {
    (|$history:ident| $impl:block) => {{
        struct G;
        impl $crate::Guesser for G {
            fn guess(&mut self, $history: &[Guess]) -> crate::Word {
                $impl
            }
        }
        G
    }};
}

#[cfg(test)]
macro_rules! mask {
    (C) => {$crate::Correctness::Correct};
    (M) => {$crate::Correctness::Misplaced};
    (W) => {$crate::Correctness::Wrong};
    ($($c:tt)+) => { crate::Pattern::new([
        $(mask!($c)),+
    ])}
}

#[cfg(test)]
mod tests {
    mod guess_matcher {
        use crate::Guess;

        macro_rules! check {
            ($prev:literal + [$($mask:tt)+] allows $next:literal) => {
                assert!(Guess {
                    word: crate::Word::new($prev),
                    patt: mask![$($mask )+]
                }
                .matches(crate::Word::new($next)));
                assert_eq!($crate::Pattern::compute(crate::Word::new($next), crate::Word::new($prev)), mask![$($mask )+]);
            };
            ($prev:literal + [$($mask:tt)+] disallows $next:literal) => {
                assert!(!Guess {
                    word: crate::Word::new($prev),
                    patt: mask![$($mask )+]
                }
                .matches(crate::Word::new($next)));
                assert_ne!($crate::Pattern::compute(crate::Word::new($next), crate::Word::new($prev)), mask![$($mask )+]);
            }
        }

        #[test]
        fn from_jon() {
            check!("abcde" + [C C C C C] allows "abcde");
            check!("abcdf" + [C C C C C] disallows "abcde");
            check!("abcde" + [W W W W W] allows "fghij");
            check!("abcde" + [M M M M M] allows "eabcd");
            check!("baaaa" + [W C M W W] allows "aaccc");
            check!("baaaa" + [W C M W W] disallows "caacc");
        }

        #[test]
        fn from_crash() {
            check!("tares" + [W M M W W] disallows "brink");
        }

        #[test]
        fn from_yukosgiti() {
            check!("aaaab" + [C C C W M] allows "aaabc");
            check!("aaabc" + [C C C M W] allows "aaaab");
        }

        #[test]
        fn from_chat() {
            // flocular
            check!("aaabb" + [C M W W W] disallows "accaa");
            // ritoban
            check!("abcde" + [W W W W W] disallows "bcdea");
        }
    }
    mod game {
        use crate::{Guess, Word, Wordle};

        #[test]
        fn genius() {
            let w = Wordle::new();
            let guesser = guesser!(|_history| { Word::new("right") });
            assert_eq!(w.play(Word::new("right"), guesser), Some(1));
        }

        #[test]
        fn magnificent() {
            let w = Wordle::new();
            let guesser = guesser!(|history| {
                if history.len() == 1 {
                    return Word::new("right");
                }
                Word::new("wrong")
            });
            assert_eq!(w.play(Word::new("right"), guesser), Some(2));
        }

        #[test]
        fn impressive() {
            let w = Wordle::new();
            let guesser = guesser!(|history| {
                if history.len() == 2 {
                    return Word::new("right");
                }
                Word::new("wrong")
            });
            assert_eq!(w.play(Word::new("right"), guesser), Some(3));
        }

        #[test]
        fn splendid() {
            let w = Wordle::new();
            let guesser = guesser!(|history| {
                if history.len() == 3 {
                    return Word::new("right");
                }
                Word::new("wrong")
            });
            assert_eq!(w.play(Word::new("right"), guesser), Some(4));
        }

        #[test]
        fn great() {
            let w = Wordle::new();
            let guesser = guesser!(|history| {
                if history.len() == 4 {
                    return Word::new("right");
                }
                Word::new("wrong")
            });
            assert_eq!(w.play(Word::new("right"), guesser), Some(5));
        }

        #[test]
        fn phew() {
            let w = Wordle::new();
            let guesser = guesser!(|history| {
                if history.len() == 5 {
                    return Word::new("right");
                }
                Word::new("wrong")
            });
            assert_eq!(w.play(Word::new("right"), guesser), Some(6));
        }

        #[test]
        fn oops() {
            let w = Wordle::new();
            let guesser = guesser!(|_history| { Word::new("wrong") });
            assert_eq!(w.play(Word::new("right"), guesser), None);
        }
    }

    mod compute {
        use crate::{Pattern, Word};

        #[test]
        fn all_green() {
            assert_eq!(Pattern::compute(Word::new("abcde"), Word::new("abcde")), mask![C C C C C]);
        }

        #[test]
        fn all_gray() {
            assert_eq!(Pattern::compute(Word::new("abcde"), Word::new("fghij")), mask![W W W W W]);
        }

        #[test]
        fn all_yellow() {
            assert_eq!(Pattern::compute(Word::new("abcde"), Word::new("eabcd")), mask![M M M M M]);
        }

        #[test]
        fn repeat_green() {
            assert_eq!(Pattern::compute(Word::new("aabbb"), Word::new("aaccc")), mask![C C W W W]);
        }

        #[test]
        fn repeat_yellow() {
            assert_eq!(Pattern::compute(Word::new("aabbb"), Word::new("ccaac")), mask![W W M M W]);
        }

        #[test]
        fn repeat_some_green() {
            assert_eq!(Pattern::compute(Word::new("aabbb"), Word::new("caacc")), mask![W C M W W]);
        }

        #[test]
        fn dremann_from_chat() {
            assert_eq!(Pattern::compute(Word::new("azzaz"), Word::new("aaabb")), mask![C M W W W]);
        }

        #[test]
        fn itsapoque_from_chat() {
            assert_eq!(Pattern::compute(Word::new("baccc"), Word::new("aaddd")), mask![W C W W W]);
        }

        #[test]
        fn ricoello_from_chat() {
            assert_eq!(Pattern::compute(Word::new("abcde"), Word::new("aacde")), mask![C W C C C]);
        }
    }
}
