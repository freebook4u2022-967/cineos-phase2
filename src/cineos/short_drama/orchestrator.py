"""Short-drama planning orchestrator."""

from .agents import (
    ContinuitySupervisor,
    DirectorAgent,
    ScreenwriterAgent,
    ShotPlanner,
    StoryArchitect,
)
from .models import DramaBrief, DramaPlan


class ShortDramaOrchestrator:
    """Turn one creative brief into a renderer-independent drama plan."""

    def __init__(self) -> None:
        self.story_architect = StoryArchitect()
        self.screenwriter = ScreenwriterAgent()
        self.director = DirectorAgent()
        self.shot_planner = ShotPlanner()
        self.continuity_supervisor = ContinuitySupervisor()

    def plan(self, brief: DramaBrief) -> DramaPlan:
        story = self.story_architect.run(brief)
        screenplay = self.screenwriter.run(story)
        direction = self.director.run(screenplay, brief.tone)
        shots = self.shot_planner.run(screenplay, brief.duration_seconds)
        continuity = self.continuity_supervisor.run(shots)
        return DramaPlan(
            brief=brief,
            story=story,
            screenplay=screenplay,
            direction=direction,
            shots=shots,
            continuity=continuity,
        )
