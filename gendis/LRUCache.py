from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def reset_stats(self):
        self.hits = 0
        self.misses = 0

    def get(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
        except KeyError:
            self.misses += 1
            return None

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value
