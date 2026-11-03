def point_in_time_join(events: list, feature_rows: list, ttl: int) -> list:
    history = {}
    for row in feature_rows:
        history.setdefault(row["entity_id"], []).append((row["timestamp"], row["value"]))
    for rows in history.values():
        rows.sort(key=lambda pair: pair[0])

    joined = []
    for event in events:
        entity_id = event["entity_id"]
        event_time = event["timestamp"]
        chosen = None
        for feature_time, value in history.get(entity_id, []):
            if feature_time > event_time:
                break
            if event_time - feature_time <= ttl:
                chosen = value
        joined.append({"entity_id": entity_id, "timestamp": event_time, "value": chosen})
    return joined
