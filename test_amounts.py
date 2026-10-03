"""
Tests for amount parsing.

The contract that matters most: anything ambiguous returns None. A blank cell is
recoverable in a spreadsheet; a wrong rent quietly summed into a portfolio total
is not.
"""

import pytest

from amounts import format_amount, parse_amount


# --- accepts the ways these leases actually write money --------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Rs. 65,000", 65000),
        ("Rs.65,000", 65000),
        ("RS 65,000", 65000),
        ("INR 65,000", 65000),
        ("₹65,000", 65000),
        ("Rs. 65,000/-", 65000),
        ("65,000", 65000),
        ("65000", 65000),
        ("65,000.50", 65000.50),
    ],
)
def test_plain_amounts(text, expected):
    assert parse_amount(text) == expected


# --- Indian lakh/crore grouping --------------------------------------------
# Indian grouping puts a comma every two digits after the last three, so
# "3,25,000" is 325000, not 3.25.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Rs. 3,25,000", 325000),
        ("Rs. 4,20,000", 420000),
        ("Rs. 1,20,00,000", 12000000),
        ("Rs. 42,00,000", 4200000),
    ],
)
def test_indian_digit_grouping(text, expected):
    assert parse_amount(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 lakh", 100000),
        ("2 lakhs", 200000),
        ("1.5 lakh", 150000),
        ("1.5 lakhs per month", 150000),
        ("1 crore", 10000000),
        ("2.5 crores", 25000000),
    ],
)
def test_lakh_and_crore_words(text, expected):
    assert parse_amount(text) == expected


# --- takes the first amount, never the sum ---------------------------------


def test_digits_win_over_the_same_amount_spelled_out():
    """Leases write both. Summing them would double the rent."""
    text = "Rs. 4,20,000 (Rupees Four Lakh Twenty Thousand)"
    assert parse_amount(text) == 420000


def test_takes_first_when_two_amounts_appear():
    assert parse_amount("Deposit Rs. 3,25,000 and rent Rs. 65,000") == 325000


def test_deposit_written_first_is_still_taken():
    assert parse_amount("monthly rent of Rs. 65,000 against deposit Rs. 3,25,000") == 65000


# --- refuses ambiguity -----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        "to be discussed separately",
        "as mutually agreed",
        "11 months",
        "4500 sq ft",
        "valid for 36 months",
        "10% annually",
        "Rent to be finalised",
        "N/A",
        "-",
    ],
)
def test_ambiguous_text_returns_none(text):
    """The important cases. None is correct; a number here would be a bug."""
    assert parse_amount(text) is None


def test_duration_is_never_read_as_money():
    """lease_term is a string like '11 months'. Reading it as 11 would put a
    term length into a rent column."""
    assert parse_amount("eleven (11) months") is None
    assert parse_amount("36 months") is None


def test_non_string_input_is_handled():
    assert parse_amount(65000) is None
    assert parse_amount(["Rs. 65,000"]) is None


def test_zero_is_a_real_amount():
    """A rent of zero is unusual but it is a number, not a failure."""
    assert parse_amount("Rs. 0") == 0


# --- formatting ------------------------------------------------------------


def test_format_uses_indian_grouping():
    assert format_amount(325000) == "Rs. 3,25,000"


def test_format_keeps_decimals():
    assert format_amount(65000.5) == "Rs. 65,000.50"


def test_format_of_none_is_empty():
    assert format_amount(None) == ""


def test_round_trip_through_format():
    assert parse_amount(format_amount(420000)) == 420000
