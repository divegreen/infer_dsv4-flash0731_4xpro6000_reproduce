def normalize_items(items):
    result = {}
    for item in items:
        sku = item["sku"].strip()
        qty = item["qty"]
        if not sku or not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            raise ValueError("Invalid item")
        result[sku] = qty
    return result
