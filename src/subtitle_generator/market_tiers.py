"""Shared market-tier definitions for source labels and jacket tone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MarketTier = Literal["pop", "mainstream", "niche"]


@dataclass(frozen=True)
class MarketTierDefinition:
    tier: MarketTier
    title: str
    short_label: str
    source_label_definition: str
    jacket_definition: str


MARKET_TIER_DEFINITIONS: dict[MarketTier, MarketTierDefinition] = {
    "pop": MarketTierDefinition(
        tier="pop",
        title="POP / mass-market commercial",
        short_label="broad airport-bookstore / mass nonfiction appeal",
        source_label_definition=(
            "Broad airport-bookstore, big-box, bestseller-list, BookTok/Bookstagram, "
            "or mass nonfiction appeal. The title/subtitle should be instantly legible "
            "to casual readers and promise a high-concept hook, transformation, suspense, "
            "celebrity/pop-culture relevance, self-help payoff, pop science/history, or "
            "similarly wide demand."
        ),
        jacket_definition=(
            "High-concept, instantly legible, and built for casual readers, gift buyers, "
            "BookTok/Bookstagram discovery, airport tables, Target/Costco displays, and "
            "library hold lists. Think celebrity memoir, self-help, pop science/history, "
            "or Malcolm Gladwell / Mary Roach / Atomic Habits-style nonfiction. Research "
            "recent bestseller lists, publisher pages, retailer copy, BookTok-friendly "
            "comps, podcasts, magazine features, and pop-culture flashpoints from the last "
            "18-24 months. The hook should land in five seconds and promise surprise, "
            "suspense, empowerment, escape, or transformation."
        ),
    ),
    "mainstream": MarketTierDefinition(
        tier="mainstream",
        title="MAINSTREAM / broad trade, book-club, literary-commercial",
        short_label="general trade or indie-bookstore appeal",
        source_label_definition=(
            "General trade, literary-commercial, narrative nonfiction, book-club, indie "
            "bookstore, NPR/NYT Book Review, LibraryReads, or public-library new-book-shelf "
            "appeal. The title/subtitle should feel accessible to a broad reading public "
            "but more substantial, literary, reported, historical, or essayistic than pure "
            "mass-market pop."
        ),
        jacket_definition=(
            "Accessible but substantial general-readership trade work for indie bookstore "
            "staff picks, NPR listeners, LibraryReads, book clubs, NYT Book Review coverage, "
            "and public-library new-book shelves. Think Ann Patchett, Erik Larson, Tara "
            "Westover, Rebecca Solnit, Patrick Radden Keefe, or narrative nonfiction from "
            "Knopf, Riverhead, FSG, Scribner, Norton, or Ecco. Research trade publisher copy, "
            "newspaper/book-section reviews, longform journalism, author interviews, and "
            "accessible scholarship. Promise emotional involvement plus something to think "
            "about, never pure hype or academic dryness."
        ),
    ),
    "niche": MarketTierDefinition(
        tier="niche",
        title="NICHE / scholarly, specialty, small-press, or deep-genre",
        short_label="specialist, academic, local, technical, or narrow audience",
        source_label_definition=(
            "Specialist, academic, local-interest, technical, professional, small-press, "
            "deep-genre, course-adoption, hobbyist, practitioner, or narrow-audience appeal. "
            "The title/subtitle may still be good, but its natural buyers are a defined "
            "field, community, genre, region, or institutional/library audience rather than "
            "the broad trade market."
        ),
        jacket_definition=(
            "For a clearly defined audience: specialists, students, practitioners, hobbyists, "
            "genre devotees, course adopters, or acquiring librarians. Think university-press "
            "monographs and academic trade crossovers from Princeton, Yale, Chicago, Duke, "
            "MIT, Verso, or Oxford; specialty nonfiction; translated/small-press literary "
            "work; technical books; or deep-genre titles. Research publisher catalog pages, "
            "Choice/ACRL-style reviews, field-specific journals, author bios, scholarly "
            "debates, specialist blogs, and prior books in the same series or subfield. "
            "The hook is authority and contribution."
        ),
    ),
}


def jacket_tone_text(tier: MarketTier) -> str:
    """Return the jacket prompt tone block for a market tier."""

    definition = MARKET_TIER_DEFINITIONS[tier]
    return f"BOOK TYPE: {definition.title}.\n{definition.jacket_definition}"


def source_label_tier_definitions() -> str:
    """Return sorter prompt definitions for source-title tier labels."""

    lines = ["Tiers:"]
    for tier in ("pop", "mainstream", "niche"):
        definition = MARKET_TIER_DEFINITIONS[tier]
        lines.append(
            f"- {tier}: {definition.source_label_definition}"
        )
    return "\n".join(lines)
