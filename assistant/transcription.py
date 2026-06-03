"""Parakeet TDT speech-to-text via ONNX Runtime.

Uses the pre-exported ONNX model files (encoder, decoder_joint, preprocessor)
from the Parakeet TDT 0.6B v3 model.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    text: str
    words: list[WordTimestamp]


class ParakeetTranscriber:
    """Runs Parakeet TDT inference via ONNX Runtime on GPU or CPU."""

    VOCAB_SIZE = 8193
    NUM_DURATIONS = 5  # TDT durations 0-4
    BLANK_ID = 8192  # <blk> is the last vocab token
    SPECIAL_TOKEN_IDS = set(range(17))  # <unk>, <|nospeech|>, <pad>, etc.
    SUBSAMPLING_FACTOR = 8
    SAMPLE_RATE = 16000
    HOP_LENGTH = 160  # 10ms at 16kHz
    PREPEND_SILENCE_S = 1.0  # Silence to prepend for conv subsampling context. Without that often the first words were not transcribed.
    MIN_AUDIO_DURATION_S = 3.0  # Pad short recordings with silence up to this duration. Arguably improves accuracy on short clips.

    def __init__(self, model_dir: str | Path, device: str = "cuda"):
        model_dir = Path(model_dir)

        providers = self._get_providers(device)

        self.preprocessor = ort.InferenceSession(
            str(model_dir / "nemo128.onnx"), providers=providers
        )
        self.encoder = ort.InferenceSession(
            str(model_dir / "encoder-model.int8.onnx"), providers=providers
        )
        self.decoder_joint = ort.InferenceSession(
            str(model_dir / "decoder_joint-model.int8.onnx"), providers=providers
        )

        self.vocab = self._load_vocab(model_dir / "vocab.txt")

    def _get_providers(self, device: str) -> list[str]:
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_vocab(self, path: Path) -> list[str]:
        vocab = []
        with open(path) as f:
            for line in f:
                token = line.split()[0] if line.strip() else ""
                vocab.append(token)
        return vocab

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> TranscriptionResult:
        """Transcribe audio samples to text with word timestamps.

        Args:
            audio: float32 array of audio samples, normalized to [-1, 1].
            sample_rate: Sample rate of audio (will resample if != 16000).
        """
        if sample_rate != self.SAMPLE_RATE:
            # Simple resample via linear interpolation
            ratio = self.SAMPLE_RATE / sample_rate
            new_len = int(len(audio) * ratio)
            indices = np.arange(new_len) / ratio
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        # Prepend silence to give conv subsampling layers left-side context
        pre_pad_samples = int(self.PREPEND_SILENCE_S * self.SAMPLE_RATE)
        # Append silence to pad short recordings up to MIN_AUDIO_DURATION_S
        min_samples = int(self.MIN_AUDIO_DURATION_S * self.SAMPLE_RATE)
        post_pad_samples = max(0, min_samples - len(audio))
        audio = np.concatenate(
            [
                np.zeros(pre_pad_samples, dtype=np.float32),
                audio,
                np.zeros(post_pad_samples, dtype=np.float32),
            ]
        )

        # Run preprocessor
        waveforms = audio.reshape(1, -1).astype(np.float32)
        waveforms_lens = np.array([audio.shape[0]], dtype=np.int64)

        features, features_lens = self.preprocessor.run(
            None, {"waveforms": waveforms, "waveforms_lens": waveforms_lens}
        )

        # Run encoder
        encoder_out, encoded_lengths = self.encoder.run(
            None, {"audio_signal": features, "length": features_lens}
        )

        # TDT greedy decode
        tokens, frame_indices = self._tdt_greedy_decode(encoder_out, encoded_lengths)

        # Build result with timestamps
        words = self._tokens_to_words_with_timestamps(tokens, frame_indices)
        text = " ".join(w.word for w in words)

        return TranscriptionResult(text=text, words=words)

    def _tdt_greedy_decode(
        self, encoder_out: np.ndarray, encoded_lengths: np.ndarray
    ) -> tuple[list[int], list[int]]:
        """TDT greedy decoding loop.

        Returns list of token IDs and their corresponding encoder frame indices.
        """
        batch_size = encoder_out.shape[0]
        assert batch_size == 1, "Only batch_size=1 supported"

        num_frames = int(encoded_lengths[0])

        # Initial decoder state
        targets = np.zeros((1, 1), dtype=np.int32)
        target_length = np.array([1], dtype=np.int32)
        states_1 = np.zeros((2, 1, 640), dtype=np.float32)
        states_2 = np.zeros((2, 1, 640), dtype=np.float32)

        tokens = []
        frame_indices = []
        frame_idx = 0
        max_tokens_per_frame = 10  # safety limit

        while frame_idx < num_frames:
            tokens_this_frame = 0

            while tokens_this_frame < max_tokens_per_frame:
                # Get encoder output for current frame
                enc_frame = encoder_out[:, :, frame_idx : frame_idx + 1]

                outputs, _, out_states_1, out_states_2 = self.decoder_joint.run(
                    None,
                    {
                        "encoder_outputs": enc_frame,
                        "targets": targets,
                        "target_length": target_length,
                        "input_states_1": states_1,
                        "input_states_2": states_2,
                    },
                )

                # outputs shape: [1, 1, 1, 8198]
                logits = outputs[0, 0, 0]  # [8198]
                token_logits = logits[: self.VOCAB_SIZE]
                duration_logits = logits[self.VOCAB_SIZE :]

                token_id = int(np.argmax(token_logits))
                duration = int(np.argmax(duration_logits))

                if token_id == self.BLANK_ID:
                    # Blank: advance by duration (minimum 1)
                    frame_idx += max(1, duration)
                    break
                else:
                    # Non-blank: emit token
                    tokens.append(token_id)
                    frame_indices.append(frame_idx)
                    tokens_this_frame += 1

                    # Update decoder state
                    targets = np.array([[token_id]], dtype=np.int32)
                    states_1 = out_states_1
                    states_2 = out_states_2

                    # TDT: advance by duration after emitting
                    if duration > 0:
                        frame_idx += duration
                        break

        return tokens, frame_indices

    def _tokens_to_words_with_timestamps(
        self, tokens: list[int], frame_indices: list[int]
    ) -> list[WordTimestamp]:
        """Convert token IDs and frame indices to words with timestamps."""
        if not tokens:
            return []

        # Decode tokens to text pieces, filtering special tokens
        filtered = [
            (self.vocab[t], fi)
            for t, fi in zip(tokens, frame_indices)
            if t not in self.SPECIAL_TOKEN_IDS
        ]
        if not filtered:
            return []
        pieces, frame_indices = zip(*filtered)
        pieces = list(pieces)
        frame_indices = list(frame_indices)

        # Group into words (SentencePiece uses ▁ as word boundary)
        words = []
        current_word = ""
        word_start_frame = frame_indices[0] if frame_indices else 0
        word_end_frame = word_start_frame

        for i, piece in enumerate(pieces):
            if piece.startswith("▁") and current_word:
                # End previous word
                words.append(
                    WordTimestamp(
                        word=current_word,
                        start=self._frame_to_time(word_start_frame),
                        end=self._frame_to_time(word_end_frame),
                    )
                )
                current_word = piece.lstrip("▁")
                word_start_frame = frame_indices[i]
            else:
                current_word += piece.lstrip("▁")

            word_end_frame = frame_indices[i]

        # Last word
        if current_word:
            words.append(
                WordTimestamp(
                    word=current_word,
                    start=self._frame_to_time(word_start_frame),
                    end=self._frame_to_time(word_end_frame),
                )
            )

        return words

    def _frame_to_time(self, frame_idx: int) -> float:
        """Convert encoder frame index to time in seconds."""
        # Each encoder frame corresponds to SUBSAMPLING_FACTOR * HOP_LENGTH samples
        sample_idx = frame_idx * self.SUBSAMPLING_FACTOR * self.HOP_LENGTH
        time_s = sample_idx / self.SAMPLE_RATE - self.PREPEND_SILENCE_S
        return max(0.0, time_s)
