use roget::{Pattern, Guess, Correctness, Word};

const TESTS: &str = include_str!("../wordle-tests/data/tests.txt");

fn to_pattern(result: &str) -> Pattern {
    assert_eq!(result.len(), 5);
    let mut out = [Correctness::Wrong; 5];
    for (c, out) in result.bytes().zip(out.iter_mut()) {
        *out = match c {
            b'c' => Correctness::Correct,
            b'm' => Correctness::Misplaced,
            b'w' => Correctness::Wrong,
            _ => {
                unreachable!("unknown pattern character '{}'", c);
            }
        };
    }
    Pattern::new(out)
}

#[test]
fn all() {
    for line in TESTS.lines() {
        let mut fields = line.split(',');
        let answer = fields.next().expect("word1");
        let guess = fields.next().expect("word2");
        let result = fields.next().expect("result");
        assert_eq!(fields.count(), 0);
        let result = to_pattern(result);
        assert_eq!(
            Pattern::compute(Word::new(answer), Word::new(guess)),
            result,
            "guess {} against {}",
            guess,
            answer
        );
        assert!(Guess {
            word: Word::new(guess),
            patt: result,
        }
        .matches(Word::new(answer)));
    }
}
