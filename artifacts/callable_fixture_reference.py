"""Hand-transcribed training examples and independent, small-input references."""
from collections import Counter, deque
from fractions import Fraction
from itertools import combinations, permutations
from math import comb, gcd
import re


# Each row is (positional argument tuple, published expected return).
PUBLISHED = {
    "q3410": [(("11112222",), 3), (("111111111111",), 3)],
    "q4170": [(("x = 12t + 18", "y = 8t + 7"), "2x - 3y = 15"),
              (("x = -12t - 18", "y = 8t + 7"), "2x + 3y = -15"),
              (("x = -t + 12", "y = 12t - 1"), "12x + y = 143")],
    "q3930": [(("Hello World",), "World Hello"), (("Hi There.",), "There. Hi")],
    "q2946": [((100000, 1, 2000, 15, 1), True), ((100000, 1, 10000, 10, 1), True),
              ((100000, 1, 9185, 12, 1), False)],
    "q4230": [(([1, 5, 12],), "odds win"), (([7, -3, 20],), "evens win"),
              (([7, -3, -2, 6],), "tie")],
    "q3798": [((0, [(0, 1), (2, 1), (0, 1)]), 2)],
    "q3611": [(("ab3f", "ac"), "07"), (("aadf", "bce2"), "163d"),
              (("1c0111001f010100061a024b53535009181c", "686974207468652062756c6c277320657965"),
               "746865206b696420646f6e277420706c6179")],
    "q4857": [((3, 0), 0), ((3, 1), 3), ((3, -2), 0), ((3, -4), -3),
              ((3, 3), 3), ((3, 6), 6), ((3, 7), 9)],
    "q3016": [((4111111111111111,), "VISA"), ((4012888888881881,), "VISA"),
              ((378282246310005,), "AMEX"), ((6011111111111117,), "Discover"),
              ((5105105105105100,), "Mastercard"), ((5105105105105106,), "Mastercard"),
              ((9111111111111111,), "Unknown")],
    "q3097": [(([1, 2, 3, 4, 5, 6, 7], 3), [3, 6, 2, 7, 5, 1, 4])],
    "q3229": [(("abab", "truck car truck car"), True), (("aaaa", "dog dog dog dog"), True),
              (("abab", "apple banana banana apple"), False), (("aaaa", "cat cat dog cat"), False)],
    "q3246": [((6, 4), 2), ((4, 4), 1), ((4, 2), -1), ((35, 7), 3), ((36, 7), -1)],
    "q2926": [((3,), 15), ((10,), 250), ((200,), 25000), ((0,), 0), ((-3,), -15)],
    "q2526": [((1, 4), 2)],
    "q1956": [(("dcab", [[0, 3], [1, 2]]), "bacd"), (("dcab", [[0, 3], [1, 2], [0, 2]]), "abcd"),
              (("cba", [[0, 1], [1, 2]]), "abc")],
    "q1888": [(([[1, 1, 0], [1, 1, 0], [0, 0, 1]],), 2), (([[1, 1, 0], [1, 1, 1], [0, 1, 1]],), 1)],
    "q373": [(("aa", "a"), False), (("aa", "a*"), True), (("ab", ".*"), True),
             (("aab", "c*a*b"), True), (("mississippi", "mis*is*p*."), False)],
    "q193": [(([2, 4, 1, 2, 7, 8],), 9), (([2, 4, 5],), 4), (([9, 8, 7, 6, 5, 1, 2, 3, 4],), 18)],
    "q138": [((0,), 0), ((3,), 2), ((6,), 4), ((9,), 14), ((333,), 393)],
    "q123": [(([1, 2, 3, 4, 5, 6, 1], 3), 12), (([2, 2, 2], 2), 4), (([9, 7, 7, 9, 7, 7, 9], 7), 55),
             (([1, 1000, 1], 1), 1), (([1, 79, 80, 1, 1, 1, 200, 1], 3), 202)],
    "q2498": [(([2, 6, 4, 1],), False), (([1, 2, 34, 3, 4, 5, 7, 23, 12],), True)],
    "q279": [(([8, 1, 9],), "981"), (([8, 6, 7, 1, 0],), "8760"),
             (([1],), ""), (([0, 0, 0, 0, 0, 0],), "0")],
    "q368": [(([1, 3, 4, 2, 2],), 2), (([3, 1, 3, 4, 2],), 3)],
    "q268": [((["2", "1", "+", "3", "*"],), 9), ((["4", "13", "5", "/", "+"],), 6),
             ((["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"],), 22)],
    "q452": [(("0.(52)", "0.5(25)"), True), (("0.1666(6)", "0.166(66)"), True), (("0.9(9)", "1."), True)],
    "q272": [(([2, 3, 1, 1, 4],), True), (([3, 2, 1, 0, 4],), False)],
    "q1991": [(([[1, 2], [2, 3], [3, 4]],), 2)],
}

ABSTAIN = {
    "q3498": "Any valid perfect-power factorization may be returned.",
    "q4828": "Floating-point distribution validation and return tolerance are outside the exact typed comparator.",
    "q4555": "The statement permits any ordering of the remaining resources in its returned tuple.",
    "q1659": "Floating-point input and mixed fraction expressions/list notation are outside the typed protocol.",
    "q4076": "The recurrence is missing in the available statement. Its sample results cannot be checked independently.",
    "q383": "Multiple peak indices can be legitimate; the second output explicitly says '1 or 5'.",
    "q1909": "The input is a drawn TreeNode graph requiring a specified platform object adapter.",
    "q1841": "The list is a serialized TreeNode graph, not the declared argument type.",
    "q423": "Either longest palindrome may be valid; the first published output has an explicitly valid alternative.",
}


def fish(shoal):
    eaten = 0
    for food in sorted(map(int, shoal)):
        size = 1
        while eaten >= 2 * size * (size + 1):
            size += 1
        if food > size:
            return size
        eaten += food
    return next(size for size in range(1, len(shoal) + 2) if eaten < 2 * size * (size + 1))


def rectangular(left, right):
    def parse(line):
        match = re.fullmatch(r"[xy] = ([-\d]*)t ([+-]) (\d+)", line)
        slope, sign, intercept = match.groups()
        return int(slope + "1" if slope in {"", "-"} else slope), int(sign + intercept)

    a, b = parse(left)
    c, d = parse(right)
    factor = gcd(a, c) * (1 if c > 0 else -1)
    x, y = c // factor, -a // factor
    return f"{'' if x == 1 else x}x {'+' if y > 0 else '-'} {'' if abs(y) == 1 else abs(y)}y = {x * b + y * d}"


def fortune(balance, rate, cost, years, inflation):
    for _ in range(years - 1):
        balance = int(balance * (1 + Fraction(rate, 100)) - cost)
        cost = int(cost * (1 + Fraction(inflation, 100)))
    return balance >= 0


def bit_war(numbers):
    def points(parity):
        return sum((-1 if n < 0 else 1) * bin(abs(n)).count("1") for n in numbers if n % 2 == parity)
    odd, even = points(1), points(0)
    return "odds win" if odd > even else "evens win" if even > odd else "tie"


def shell_game(start, swaps):
    # Trace possible source position back from each final cup.
    for end in {start, *(cup for swap in swaps for cup in swap)}:
        origin = end
        for a, b in reversed(swaps):
            origin = b if origin == a else a if origin == b else origin
        if origin == start:
            return end


def issuer(number):
    digits = str(number)
    for lengths, prefixes, company in (
        ({13, 16}, ("4",), "VISA"), ({15}, ("34", "37"), "AMEX"),
        ({16}, ("6011",), "Discover"), ({16}, ("51", "52", "53", "54", "55"), "Mastercard"),
    ):
        if len(digits) in lengths and digits.startswith(prefixes):
            return company
    return "Unknown"


def josephus(items, k):
    ring = deque(items)
    while ring:
        ring.rotate(1 - k)
        yield ring.popleft()


def word_pattern(pattern, text):
    words = text.split()
    return len(pattern) == len(words) and all(
        (pattern[i] == pattern[j]) == (words[i] == words[j])
        for i, j in combinations(range(len(words)), 2)
    )


def swaps(text, pairs):
    seen, todo = {text}, [text]
    for item in todo:
        for i, j in pairs:
            chars = list(item)
            chars[i], chars[j] = chars[j], chars[i]
            other = "".join(chars)
            if other not in seen:
                todo.append(other)
                seen.add(other)
    return min(seen)


def friend_circles(matrix):
    groups = [{i} for i in range(len(matrix))]
    for i, j in combinations(range(len(matrix)), 2):
        if matrix[i][j]:
            a, b = (next(group for group in groups if n in group) for n in (i, j))
            if a is not b:
                a.update(b)
                groups.remove(b)
    return len(groups)


def coins(piles):
    if not piles:
        return 0
    return max(sorted((piles[0], piles[i], piles[j]))[1] +
               coins([pile for k, pile in enumerate(piles) if k not in {0, i, j}])
               for i, j in combinations(range(1, len(piles)), 2))


def bit_steps(start):
    distance, todo = {start: 0}, [start]
    for n in todo:
        if n == 0:
            return distance[0]
        neighbors = [n ^ 1]
        for bit in range(1, start.bit_length()):
            if n % (1 << bit) == 1 << (bit - 1):
                neighbors.append(n ^ (1 << bit))
        for nxt in neighbors:
            if nxt not in distance:
                distance[nxt] = distance[n] + 1
                todo.append(nxt)


def cards(piles, k):
    if k == 0:
        return 0
    return max(piles[0] + cards(piles[1:], k - 1), piles[-1] + cards(piles[:-1], k - 1))


def multiple_three(digits):
    valid = [0]
    for length in range(1, len(digits) + 1):
        for seq in permutations(digits, length):
            if sum(seq) % 3 == 0:
                valid.append(int("".join(map(str, seq))))
    maximum = max(valid)
    return str(maximum) if maximum or 0 in digits else ""


def reverse_polish(tokens):
    def expression():
        item = reverse.pop()
        if item not in {"+", "-", "*", "/"}:
            return int(item)
        b, a = expression(), expression()
        if item == "/":
            return int(Fraction(a, b))
        return {"+": a + b, "-": a - b, "*": a * b}[item]

    reverse = tokens[:]
    return expression()


def decimal(value):
    whole, _, fraction = value.partition(".")
    prefix, _, cycle = fraction.partition("(")
    cycle = cycle.removesuffix(")")
    units = 10 ** len(prefix)
    return (int(whole) + Fraction(int(prefix or "0"), units) +
            Fraction(int(cycle or "0"), units * max(1, 10 ** len(cycle) - 1)))


def can_jump(nums):
    reached = {0}
    for i, step in enumerate(nums):
        if i in reached:
            reached.update(range(i + 1, min(len(nums), i + step + 1)))
    return len(nums) - 1 in reached


def longest_chain(pairs):
    return max(len(seq) for n in range(len(pairs) + 1) for seq in combinations(sorted(pairs), n)
               if all(a[1] < b[0] for a, b in zip(seq, seq[1:])))


REFERENCES = {
    "q3410": fish, "q4170": rectangular, "q3930": lambda s: " ".join(s.split()[::-1]),
    "q2946": fortune, "q4230": bit_war, "q3798": shell_game,
    "q3611": lambda a, b: "".join(format(int(x, 16) ^ int(y, 16), "x") for x, y in zip(a, b)),
    "q4857": lambda coin, price: next(i for i in range(price, price + coin) if i % coin == 0),
    "q3016": issuer, "q3097": lambda items, k: list(josephus(items, k)), "q3229": word_pattern,
    "q3246": lambda m, n: next((k for k in range(n + 1) if comb(n, k) == m), -1),
    "q2926": lambda n: n * 5 ** len(str(abs(n))), "q2526": lambda x, y: sum(
        a != b for a, b in zip(f"{x:032b}", f"{y:032b}")),
    "q1956": swaps, "q1888": friend_circles, "q373": lambda s, p: re.fullmatch(p, s) is not None,
    "q193": coins, "q138": bit_steps, "q123": cards,
    "q2498": lambda values: any(a % 2 and b % 2 and c % 2 for a, b, c in zip(values, values[1:], values[2:])),
    "q279": multiple_three, "q368": lambda ns: next(x for x, count in Counter(ns).items() if count > 1),
    "q268": reverse_polish, "q452": lambda s, t: decimal(s) == decimal(t),
    "q272": can_jump, "q1991": longest_chain,
}


def max_mex_subset(values, target):
    subsets = (subset for n in range(len(values) + 1) for subset in combinations(values, n))
    return max((len(subset) for subset in subsets if target not in subset and
                all(i in subset for i in range(1, target))), default=-1)
