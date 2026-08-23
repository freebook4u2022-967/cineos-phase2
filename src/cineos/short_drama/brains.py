"""Provider-neutral creative intelligence for CINEOS short drama planning.

Sprint 2 deliberately keeps these brains deterministic and local. The decision
contracts are the product surface; future learned models may implement the same
interfaces without changing the orchestrator.
"""

from __future__ import annotations

import re

from .models import CharacterProfile, DramaBrief


class DramaBrain:
    """Expand a one-line premise into a short-drama story bible."""

    STRUCTURE = ("hook", "escalation", "reversal", "climax", "resolution")

    def run(self, brief: DramaBrief) -> dict:
        premise = brief.premise.strip()
        hook = f"Open on the most emotionally disruptive consequence of: {premise}"
        stakes = "The protagonist must act before the situation becomes irreversible."
        reversal = "New evidence changes the meaning of what the protagonist believed."
        climax = "The protagonist makes a costly choice that resolves the central question."
        resolution = "End on a visual consequence that answers the premise but leaves emotional residue."
        return {
            "premise": premise,
            "genre": brief.genre,
            "tone": brief.tone,
            "target_duration_seconds": brief.duration_seconds,
            "theme": self._theme_for(brief.genre),
            "hook": hook,
            "stakes": stakes,
            "twist": reversal,
            "climax": climax,
            "resolution": resolution,
            "structure": list(self.STRUCTURE),
            "emotional_curve": ["curiosity", "unease", "shock", "pressure", "aftershock"],
        }

    @staticmethod
    def _theme_for(genre: str) -> str:
        themes = {
            "mystery": "truth has a personal cost",
            "thriller": "control collapses under pressure",
            "romance": "intimacy requires vulnerability",
            "horror": "the feared truth cannot stay buried",
            "comedy": "certainty is often the setup for embarrassment",
            "drama": "a difficult choice reveals character",
        }
        return themes.get(genre.lower(), "choices reveal character")


class CharacterBrain:
    """Create persistent dramatic character state from the premise."""

    def run(self, brief: DramaBrief, story: dict) -> list[CharacterProfile]:
        names = self._extract_names(brief.premise)
        protagonist_name = names[0] if names else "Protagonist"
        profiles = [
            CharacterProfile(
                character_id="char-protagonist",
                name=protagonist_name,
                role="protagonist",
                motivation="discover what is really happening and regain agency",
                fear="losing control of reality",
                secret="withholds one emotionally important fact",
                relationships={},
                knowledge=["the ordinary world before the inciting incident"],
                emotion="guarded",
                physical_state="uninjured",
                wardrobe="continuity-default",
                props=[],
            )
        ]
        if self._implies_second_character(brief.premise):
            profiles.append(
                CharacterProfile(
                    character_id="char-counterpart",
                    name=names[1] if len(names) > 1 else "Counterpart",
                    role="counterpart",
                    motivation="force the protagonist to confront the hidden truth",
                    fear="being misunderstood or erased",
                    secret=story["twist"],
                    relationships={"char-protagonist": "emotionally consequential"},
                    knowledge=["information withheld from the protagonist"],
                    emotion="unreadable",
                    physical_state="story-dependent",
                    wardrobe="continuity-default",
                    props=[],
                )
            )
        return profiles

    @staticmethod
    def _extract_names(text: str) -> list[str]:
        # Conservative extraction: preserve capitalized multi-letter words while
        # filtering sentence-openers and common dramatic nouns.
        blocked = {"A", "An", "The", "His", "Her", "Man", "Woman", "Wife", "Husband"}
        return [
            token
            for token in re.findall(r"\b[A-Z][a-z]{2,}\b", text)
            if token not in blocked
        ][:4]

    @staticmethod
    def _implies_second_character(text: str) -> bool:
        lowered = text.lower()
        markers = ("wife", "husband", "friend", "daughter", "son", "mother", "father", "stranger", "message from")
        return any(marker in lowered for marker in markers)
