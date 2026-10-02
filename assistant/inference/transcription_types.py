"""Lightweight transcription data types (no heavy dependencies)."""

from dataclasses import dataclass


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    text: str
    words: list[WordTimestamp]
