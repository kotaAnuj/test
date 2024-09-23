# memory.py
from collections import deque

class Memory:
    def __init__(self, short_term_size=10):
        self.short_term = deque(maxlen=short_term_size)
        self.long_term = {}

    def add_to_short_term(self, key, value):
        self.short_term.appendleft((key, value))

    def add_to_long_term(self, key, value):
        self.long_term[key] = value

    def get_relevant_info(self, keywords):
        relevant_info = []
        for key, value in self.short_term:
            if any(keyword in key for keyword in keywords):
                relevant_info.append((key, value))
        for key, value in self.long_term.items():
            if any(keyword in key for keyword in keywords):
                relevant_info.append((key, value))
        return relevant_info