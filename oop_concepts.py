"""
OOP concepts helpers:
- Decorators (log_decorator, time_decorator) -> used by GUI/model methods
- OOPExplanation base class + OOPConcepts subclass -> demonstrates method overriding
- Multiple inheritance will be used by the GUI (AIGUI inherits from BaseGUI and OOPConcepts)
"""

import time
from functools import wraps

def log_decorator(func):
    """Logs function entry/exit to console (and returns original result)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[LOG] Entering {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[LOG] Exiting {func.__name__}")
        return result
    return wrapper

def time_decorator(func):
    """Measures elapsed time (notifies via print)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        t1 = time.time()
        print(f"[TIME] {func.__name__} took {t1-t0:.2f}s")
        return res
    return wrapper

class OOPExplanation:
    """Base class whose method will be overridden by a subclass to show
    method overriding in action."""
    def explanation(self):
        return "OOPExplanation: base explanation (should be overridden)."

class OOPConcepts(OOPExplanation):
    """Subclass that overrides explanation() and applies multiple decorators."""
    @log_decorator
    @time_decorator
    def explanation(self):
        # Method overriding: replaces base behavior with a richer string.
        return (
            "OOP Concepts used in this project:\n"
            "- Multiple inheritance (GUI inherits from BaseGUI and OOPConcepts)\n"
            "- Encapsulation (model internals are _protected attributes)\n"
            "- Polymorphism (BaseModel.run overridden by subclasses)\n"
            "- Method overriding (OOPConcepts.explanation overrides base)\n"
            "- Multiple decorators (time_decorator, log_decorator) applied here\n"
            "- Decorators also used to log and time model runs\n"
        )
