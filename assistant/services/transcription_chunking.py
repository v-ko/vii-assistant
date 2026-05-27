"""Client-side audio chunking and stitching utilities for ASR.

The inference server should run chunk-level inference only.
These helpers are intended for frontend/client orchestration.
"""

from __future__ import annotations

import numpy as np

from assistant.transcription import TranscriptionResult, WordTimestamp

CHUNK_DURATION_S = 30
OVERLAP_DURATION_S = 7


def split_audio_into_chunks(
    audio: np.ndarray,
    sample_rate: int,
    *,
    chunk_duration_s: int,
    overlap_duration_s: int,
) -> list[tuple[float, np.ndarray]]:
    """Split audio into overlapping chunks.

    Returns tuples of (chunk_start_time_seconds, chunk_audio).
    """
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be > 0")
    if overlap_duration_s < 0:
        raise ValueError("overlap_duration_s must be >= 0")

    chunk_samples = int(chunk_duration_s * sample_rate)
    overlap_samples = int(overlap_duration_s * sample_rate)
    step = chunk_samples - overlap_samples
    if step <= 0:
        raise ValueError("overlap_duration_s must be smaller than chunk_duration_s")

    chunks: list[tuple[float, np.ndarray]] = []
    offset = 0
    while offset < len(audio):
        end = min(offset + chunk_samples, len(audio))
        chunks.append((offset / sample_rate, audio[offset:end]))
        if end >= len(audio):
            break
        offset += step
    return chunks


def stitch_chunk_results(
    chunk_results: list[tuple[float, TranscriptionResult]],
    *,
    chunk_duration_s: int,
) -> TranscriptionResult:
    """Stitch chunk-level ASR results into one timeline.

    `chunk_results` items are (chunk_start_time_seconds, result), where each
    result has timestamps relative to chunk start.
    """
    if not chunk_results:
        return TranscriptionResult(text="", words=[])
    if len(chunk_results) == 1:
        _, result = chunk_results[0]
        return result

    # Shift each chunk result to absolute timeline first.
    shifted: list[tuple[float, TranscriptionResult]] = []
    for chunk_start, result in chunk_results:
        words = [
            WordTimestamp(
                word=w.word, start=w.start + chunk_start, end=w.end + chunk_start
            )
            for w in result.words
        ]
        shifted.append(
            (chunk_start, TranscriptionResult(text=result.text, words=words))
        )

    stitched: list[WordTimestamp] = list(shifted[0][1].words)

    for i in range(1, len(shifted)):
        prev_start = shifted[i - 1][0]
        curr_start = shifted[i][0]
        curr_words = shifted[i][1].words

        overlap_begin = curr_start
        overlap_end = prev_start + chunk_duration_s

        left_overlap = [w for w in stitched if w.start >= overlap_begin]
        right_overlap = [w for w in curr_words if w.start <= overlap_end]

        left_cut, right_cut = _find_overlap_alignment(left_overlap, right_overlap)

        if left_cut is not None and right_cut is not None:
            trim_count = len(left_overlap) - left_cut
            if trim_count > 0:
                stitched = stitched[:-trim_count]
            stitched.extend(curr_words[right_cut:])
        else:
            last_time = stitched[-1].end if stitched else 0.0
            stitched.extend(w for w in curr_words if w.start > last_time)

    text = " ".join(w.word for w in stitched)
    return TranscriptionResult(text=text, words=stitched)


def _find_overlap_alignment(
    left_words: list[WordTimestamp],
    right_words: list[WordTimestamp],
) -> tuple[int | None, int | None]:
    """Find best contiguous token match and return midpoint cut indices."""
    left_texts = [w.word.lower() for w in left_words]
    right_texts = [w.word.lower() for w in right_words]

    if not left_texts or not right_texts:
        return None, None

    best_match_len = 0
    best_left_start = 0
    best_right_start = 0

    for shift in range(-len(right_texts) + 1, len(left_texts)):
        match_len = 0
        li = max(0, shift)
        ri = max(0, -shift)
        current_run = 0
        run_start_left = li
        run_start_right = ri

        while li < len(left_texts) and ri < len(right_texts):
            if left_texts[li] == right_texts[ri]:
                current_run += 1
                if current_run > match_len:
                    match_len = current_run
                    run_start_left = li - current_run + 1
                    run_start_right = ri - current_run + 1
            else:
                current_run = 0
            li += 1
            ri += 1

        if match_len > best_match_len:
            best_match_len = match_len
            best_left_start = run_start_left
            best_right_start = run_start_right

    if best_match_len < 2:
        return None, None

    mid_offset = best_match_len // 2
    return best_left_start + mid_offset, best_right_start + mid_offset
