from stockflow.ledger import apply_event


def replay(stock, events):
    available = stock.copy()
    held = {}
    for event in events:
        available, held = apply_event(available, held, event)
    return {"available": available, "held": held}
