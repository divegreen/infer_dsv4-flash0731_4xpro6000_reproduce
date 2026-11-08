from billing.money import discounted_line


def invoice_total(lines):
    total = 0
    for line in lines:
        total += discounted_line(line["unit_cents"], line["quantity"], line.get("discount_bps", 0))
    return total
