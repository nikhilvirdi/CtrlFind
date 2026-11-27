"""Independent counterexamples for verification-depth adjudication on training statements.

A matching witness establishes only compatibility on that case. Only an observed
semantic contradiction or deterministic program error can establish a false pass.
"""

# Each entry gives valid stdin, the unique semantic answer and its derivation.
CASES = {
    "q1605": [
        ("2 2\n..\n..", "2", "The empty square permits RD and DR."),
        ("2 2\n..\n.R", "0", "Entering the occupied destination would push its rock through a wall."),
    ],
    "q2075": [
        ("2 1\n1 11", "10", "Either permutation has the single absolute difference 10."),
        ("3 2\n5 8 15", "3", "Only positions one and three contribute; the closest pair is 5 and 8."),
    ],
    "q1606": [
        ("2 3", "33", "For remainders 1 and 2 and multipliers 1 and 2, x is 4, 7, 8 or 14. Their sum is 33."),
    ],
    "q2060": [
        ("2\n13 9", "0", "Sign choices give the sorted pair -13,9."),
        ("3\n1 4 1", "1", "The middle 4 is either larger than the final entry or smaller than the first. All-positive has one inversion."),
    ],
    "q2095": [
        ("7\nzeroone", "1 0", "The letters spell one zero and one one. Put the larger digit first."),
        ("6\noneone", "1 1", "Two copies of one supply two binary ones."),
    ],
    "q2127": [
        ("2\n3 11", "33", "There is exactly one pair and its least common multiple is 33."),
        ("3\n6 10 15", "30", "Each pair has least common multiple 30."),
    ],
    "q2155": [
        ("1 3\n1\n500\n1000", "1 1 1", "With only one type, every threshold is first achieved on day one."),
        ("2 2\n500\n1000", "2 2", "The probability of both types is zero on day one and one half on day two."),
    ],
    "q2169": [
        ("2\n0 5\n7 0\n1 2", "12 0", "The two directed shortest paths cost 5 and 7. A single vertex has no distinct pairs."),
    ],
    "q2006": [
        ("1111", "3", "Windows [1,3], [2,4] and [1,4] contain three equally spaced ones."),
        ("1", "0", "A length-one interval cannot contain a triple."),
    ],
    "q2134": [
        ("3\n1000000\n1000000\n1000000", "3", "All three groups meet in slot one."),
        ("3\n1000000\n0100000\n0010000", "1", "The three classes occupy disjoint slots."),
    ],
    "q2066": [
        ("3 1\n3 1 2 3", "0", "The only chain is already the target."),
        ("2 2\n1 1\n1 2", "1", "One nesting operation joins the isolated dolls."),
    ],
    "q2091": [
        ("3 2\n1 2\n2 3\n1 3 1\n1 2 1", "-1", "The unique 1-to-3 path has two edges, exceeding the budget of one."),
    ],
    "q2119": [
        ("1\n1", "NO", "Sam takes the sole stone and Jon has no move."),
        ("2\n1\n1", "YES", "With two singleton piles Sam and then Jon each take the only stone from one pile."),
    ],
    "q2153": [
        ("1\n1", "-1", "One indivisible piece cannot make every box count divisible by a common integer greater than one."),
        ("3\n1 0 1", "2", "Both pieces must end in the same box. Their initial separation is two."),
    ],
    "q2196": [
        ("3\n7 7 7", "0", "Equal elements cannot increase any position."),
        ("4\n1 2 3 4", "3", "The cyclic permutation 2,3,4,1 improves three positions. The maximum cannot increase."),
    ],
    "q2201": [
        ("2\n7 12", "19", "The XOR contains only one pair sum, 19."),
    ],
    "q20": [
        ("2\n3 3 3\n1 2 3\n2 1 2\n1 1", "3 1", "The first full window has three different shows and the second has one."),
    ],
    "q2007": [
        ("2\n1 2", "Bob", "The only difference is already in the set, so Alice loses."),
        ("2\n1 3", "Alice", "Alice adds 2 and leaves the closed set {1,2,3}."),
    ],
    "q2058": [
        ("ab", "1", "One replacement gives bba, which contains no ab."),
        ("aab", "3", "The only possible sequence is aab, abba, bbaba, bbbbaa."),
    ],
    "q2064": [
        ("3 2\n0\n0\n0", "3", "All three employees need to learn at least one language; teaching one common language suffices."),
    ],
    "q2067": [
        ("2\n5 61", "56", "With just two positions there is no choice of the final pair. Their distance is 56."),
    ],
    "q2073": [
        ("1 0\n1 1 5 4", "12", "The one required magnet has dimensions 4 by 3."),
    ],
    "q2089": [
        ("3\n1 2 1", "1", "The entire nonempty row is a palindrome."),
        ("3\n1 2 3", "3", "All colors are distinct, so each removed palindrome is a singleton."),
    ],
    "q2098": [
        ("2\n1 2 17\n1\n1", "17", "Assign each node the other node. Every path then uses the only edge of weight 17."),
    ],
    "q2131": [
        ("2\n2\n2", "3", "The last ball must be color 2; the other color-2 ball may occupy any of the first three positions."),
    ],
    "q2139": [
        ("0\n1", "NO", "Leading deletions and appending even parity preserve the absence of ones."),
        ("1\n11", "YES", "Appending the odd parity of 1 gives 11."),
    ],
    "q2192": [
        ("2\n5 9", "4", "The largest allowed remainder is 9 modulo 5."),
    ],
    "q2193": [
        ("2\n2 1\n100 1", "0", "Both original beams are too short to reach another beacon. An added distant short beam can avoid both."),
    ],
    "q2194": [
        ("3\n0 0 0 0\n1 0 0 0\n1 0 0 0", "3", "Both other students have higher totals than student 1."),
    ],
    "q2216": [
        ("2 1 2\n1 2 5\n2 8\n2 9", "2", "The road is shorter than both trains, so both trains can close."),
    ],
    "q2218": [
        ("2 1\n1 0\n1 2", "-1", "Keeping the only edge gives both vertices odd degree; removing it gives both even."),
    ],
    "q2253": [
        ("3\na\na\na", "3", "All three unordered index pairs concatenate to aa."),
    ],
    "q2256": [
        ("3\n5 5 5", "2", "Only the two length-two intervals have even length, and both halves agree."),
    ],
    "q2279": [
        ("2\n0 0\n1 1", "1", "One pair of distinct x coordinates defines one parabola; both given points lie on it."),
    ],
    "q2355": [
        ("ab\n1\ncc", "NO", "A sequence of copies of cc never contains ab."),
        ("ab\n2\nca\nbd", "YES", "The concatenation cabd contains ab."),
    ],
    "q2376": [
        ("2\n1\n9\n4\n2 2 2 2", "1 4", "Each array consists of a repeated single value and the entire array is a three-block palindrome."),
    ],
    "q8": [
        ("2\n1\n0 8\n2\n1 4\n1 7", "0 4", "A zero requirement votes freely. In the second election one voter must be bought; the cheaper one costs 4 and unlocks the other."),
    ],
}
