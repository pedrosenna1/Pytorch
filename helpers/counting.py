class UniqueCounter:
    def __init__(self, multipliers: dict):
        """
        multipliers: dict {classe: peso}
        """
        self.multipliers = multipliers
        self.seen_ids = {}     # {classe: set(ids)}
        self.raw_counts = {}  # {classe: int}

    def _ensure(self, cls_name):
        if cls_name not in self.seen_ids:
            self.seen_ids[cls_name] = set()
        if cls_name not in self.raw_counts:
            self.raw_counts[cls_name] = 0

    def observe(self, cls_name, track_id):
        self._ensure(cls_name)
        if track_id not in self.seen_ids[cls_name]:
            self.seen_ids[cls_name].add(track_id)
            self.raw_counts[cls_name] += 1

    def weighted_total(self):
        total = 0.0
        for cls, count in self.raw_counts.items():
            total += count * float(self.multipliers.get(cls, 0))
        return total
