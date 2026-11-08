from stockflow.normalize import normalize_items


def apply_event(stock, held, event):
    available = stock.copy()
    reservations = held.copy()
    event_id = event["id"].strip()
    if not event_id:
        raise ValueError("Empty reservation id")
    if event["kind"] == "reserve":
        basket = normalize_items(event["items"])
        for sku, qty in basket.items():
            if available.get(sku, 0) < qty:
                raise ValueError("Insufficient stock")
        for sku, qty in basket.items():
            available[sku] = available.get(sku, 0) - qty
        reservations[event_id] = basket
    elif event["kind"] == "release":
        basket = reservations.pop(event_id, {})
        for sku, qty in basket.items():
            available[sku] = qty
    else:
        raise ValueError("Unknown event kind")
    return available, reservations
