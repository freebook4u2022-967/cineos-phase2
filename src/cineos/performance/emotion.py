from dataclasses import dataclass, field

CONTRADICTIONS = {
    frozenset(("joy", "grief")),
    frozenset(("calm", "panic")),
    frozenset(("trust", "disgust")),
}


@dataclass(slots=True)
class EmotionalState:
    time: float
    emotion: str
    intensity: float = 0.5
    source: str = "nova"
    locked: bool = False


@dataclass(slots=True)
class EmotionalArc:
    states: list[EmotionalState] = field(default_factory=list)
    cinedna_constraints: list[str] = field(default_factory=list)
    continuity_source: str = ""

    def contradictions(self):
        by_time = {}
        for state in self.states:
            by_time.setdefault(state.time, set()).add(state.emotion)
        return [
            sorted(pair)
            for values in by_time.values()
            for pair in CONTRADICTIONS
            if pair <= values
        ]
