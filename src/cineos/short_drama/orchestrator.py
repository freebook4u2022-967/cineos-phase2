"""Short-drama planning orchestrator."""

from .agents import ContinuitySupervisor, ScreenwriterAgent, ShotPlanner
from .brains import CharacterBrain, DramaBrain
from .directing import DirectorDecisionEngine
from .models import DramaBrief, DramaPlan
from .state import SceneStateEngine


class ShortDramaOrchestrator:
    """Turn one creative brief into a renderer-independent drama plan."""

    def __init__(self) -> None:
        self.drama_brain = DramaBrain()
        self.character_brain = CharacterBrain()
        self.screenwriter = ScreenwriterAgent()
        self.director = DirectorDecisionEngine()
        self.shot_planner = ShotPlanner()
        self.state_engine = SceneStateEngine()
        self.continuity_supervisor = ContinuitySupervisor()

    def plan(self, brief: DramaBrief) -> DramaPlan:
        story = self.drama_brain.run(brief)
        characters = self.character_brain.run(brief, story)
        screenplay = self.screenwriter.run(story)
        direction = self.director.run(screenplay, brief.tone)
        shots = self.shot_planner.run(
            screenplay, brief.duration_seconds, direction
        )
        scene_states = self.state_engine.build_timeline(
            characters, screenplay["scenes"]
        )
        continuity = self.continuity_supervisor.run(shots, scene_states)
        return DramaPlan(
            brief=brief,
            story=story,
            characters=characters,
            screenplay=screenplay,
            direction=direction,
            shots=shots,
            continuity=continuity,
            scene_states=scene_states,
        )
