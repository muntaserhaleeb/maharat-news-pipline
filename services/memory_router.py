"""
Memory router — decides which Qdrant collection(s) to query.

Routes:
  "news"      → maharat_content_live   (events, announcements, MoUs, graduations)
  "knowledge" → maharat_knowledge_live  (mission, programs, accreditation, campus)
  "both"      → dual retrieval for drafting tasks or mixed-intent queries
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RouteResult:
    route: str                              # "news" | "knowledge" | "both"
    intent: str                             # same or caller-supplied override
    reasoning: str                          # short explanation for logging
    news_filters: Dict[str, Any] = field(default_factory=dict)
    knowledge_filters: Dict[str, Any] = field(default_factory=dict)


class MemoryRouter:
    """
    Keyword-signal router. No LLM call required.

    Scoring:
      - Count token matches against three signal lists: DRAFT, NEWS, KNOWLEDGE
      - DRAFT hit wins unconditionally → "both"
      - Otherwise: whichever of NEWS / KNOWLEDGE scores higher wins
      - Tie or both-zero → "both" (safest default for unknown queries)
    """

    _DRAFT_SIGNALS: List[str] = [
        "draft", "write", "generate", "article", "linkedin", "post",
        "magazine", "press release", "news story", "feature story",
        "create content", "create article",
    ]

    _NEWS_SIGNALS: List[str] = [
        "event", "ceremony", "graduation", "mou", "agreement", "signed",
        "collaboration", "drill", "competition", "award", "visit",
        "workshop", "conference", "seminar", "celebration",
        "achievement", "activity", "highlights", "recap", "announcement",
        "hosted", "welcomed", "completed", "launched", "held", "attended",
        "sinopec", "samsung", "nesma", "saudi aramco", "sabic",
        "ojt", "on-the-job", "intake", "job fair",
        "recent", "latest", "this year", "2024", "2025", "2026",
    ]

    _KNOWLEDGE_SIGNALS: List[str] = [
        "what is maharat", "who is maharat", "about maharat",
        "mission", "vision", "history", "established", "founded",
        "accreditation", "certification", "iso", "abet", "ncaaa",
        "program", "course", "curriculum", "short course",
        "methodology", "training approach", "campus", "facility", "facilities",
        "faq", "frequently asked",
        "describe maharat", "explain maharat", "overview",
        "institutional", "governance", "strategy", "strategic",
        "brand", "editorial style", "credential", "qualification",
        "who are", "tell me about maharat",
    ]

    def route_query(
        self,
        query: str,
        intent: Optional[str] = None,
    ) -> RouteResult:
        """
        Classify the query and return a RouteResult.

        intent — explicit override: "news" | "knowledge" | "both"
                 Bypasses keyword scoring when supplied.
        """
        if intent in ("news", "knowledge", "both"):
            return RouteResult(
                route=intent,
                intent=intent,
                reasoning=f"Explicit intent override: {intent}",
            )

        q               = query.lower()
        draft_score     = sum(1 for kw in self._DRAFT_SIGNALS     if kw in q)
        news_score      = sum(1 for kw in self._NEWS_SIGNALS      if kw in q)
        knowledge_score = sum(1 for kw in self._KNOWLEDGE_SIGNALS if kw in q)

        if draft_score > 0:
            route     = "both"
            reasoning = (
                f"Drafting/authoring task detected "
                f"(draft={draft_score}, news={news_score}, knowledge={knowledge_score})"
            )
        elif news_score > 0 and knowledge_score > 0:
            route     = "both"
            reasoning = (
                f"Mixed signals — querying both collections "
                f"(news={news_score}, knowledge={knowledge_score})"
            )
        elif knowledge_score > news_score:
            route     = "knowledge"
            reasoning = (
                f"Institutional knowledge query "
                f"(knowledge={knowledge_score}, news={news_score})"
            )
        elif news_score > 0:
            route     = "news"
            reasoning = (
                f"News/event query "
                f"(news={news_score}, knowledge={knowledge_score})"
            )
        else:
            route     = "both"
            reasoning = "No strong signal — defaulting to both collections"

        return RouteResult(
            route=route,
            intent=intent or route,
            reasoning=reasoning,
        )
