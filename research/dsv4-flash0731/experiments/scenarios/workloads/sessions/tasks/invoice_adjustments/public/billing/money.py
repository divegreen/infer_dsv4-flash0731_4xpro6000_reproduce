def discounted_line(unit_cents, quantity, discount_bps):
    for value in (unit_cents, quantity, discount_bps):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("Amounts must be integers")
    if unit_cents < 0 or quantity <= 0 or discount_bps < 0 or discount_bps > 10000:
        raise ValueError("Invalid money input")
    numerator = unit_cents * quantity * (10000 - discount_bps)
    return numerator // 10000
