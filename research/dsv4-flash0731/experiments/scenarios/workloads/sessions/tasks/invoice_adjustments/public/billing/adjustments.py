from billing.invoice import invoice_total


def settle(lines, refunds):
    gross = invoice_total(lines)
    refunded = 0
    seen = {}
    for event in refunds:
        identifier = event["id"].strip()
        amount = event["amount_cents"]
        if not identifier or not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError("Invalid refund")
        seen[identifier] = amount
        refunded += amount
    if refunded > gross:
        raise ValueError("Refunds exceed invoice")
    return {"gross_cents": gross, "refunded_cents": refunded, "net_cents": gross + refunded}
