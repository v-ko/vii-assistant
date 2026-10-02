from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from assistant.utils.capture_utils import take_screenshot
from assistant.utils.misc import RectShape, Shape

log = logging.getLogger(__name__)


@dataclass
class Sample:
    """A single experiment sample."""

    image: Image.Image
    prompt: str
    ground_truth_shapes: list[Shape] = field(default_factory=list)


class DataLoader:
    """Base class for experiment data loaders.

    Subclasses must implement __len__ and get_sample.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def get_sample(self, index: int) -> Sample:
        raise NotImplementedError


class ScreenGrabDataLoader(DataLoader):
    """Grabs a screenshot from the watched screen as a single sample.

    Used for testing the experiment pipeline.
    """

    def __init__(self, prompt: str):
        self._prompt = prompt

    def __len__(self) -> int:
        return 1

    def get_sample(self, index: int) -> Sample:
        if index != 0:
            raise IndexError(f"ScreenGrabDataLoader has 1 sample, got index={index}")

        from assistant.facade import vii
        from assistant.utils.image_utils import qpixmap_to_pil

        capture = vii.app.view_state.capture_screen_info
        if capture is None:
            raise RuntimeError("No capture screen configured")
        from assistant.utils.misc import get_screen_by_name

        screen = get_screen_by_name(capture.name)
        if screen is None:
            raise RuntimeError(f"Screen '{capture.name}' not found")
        pixmap = take_screenshot(screen)
        if pixmap is None:
            raise RuntimeError("Failed to capture screenshot")

        pil_image = qpixmap_to_pil(pixmap)
        return Sample(
            image=pil_image,
            prompt=self._prompt,
        )


class ScreenSpotProDataLoader(DataLoader):
    """Loads samples from the ScreenSpot-Pro dataset.

    Expects a dataset_path pointing to the root of the cloned repo
    (containing 'annotations/' and 'images/' folders).
    """

    def __init__(
        self,
        dataset_path: str,
        prompt_template: str = "Locate the element: {instruction}",
    ):
        self._dataset_path = Path(dataset_path)
        self._prompt_template = prompt_template
        self._samples: list[dict] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        annotations_dir = self._dataset_path / "annotations"
        if not annotations_dir.exists():
            raise FileNotFoundError(f"Annotations dir not found: {annotations_dir}")

        for json_file in sorted(annotations_dir.glob("*.json")):
            with open(json_file, encoding="utf-8") as f:
                entries = json.load(f)
            self._samples.extend(entries)

        log.info(
            f"ScreenSpotPro: loaded {len(self._samples)} samples from {annotations_dir}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def get_sample(self, index: int) -> Sample:
        if index < 0 or index >= len(self._samples):
            raise IndexError(f"Index {index} out of range (0-{len(self._samples)-1})")

        entry = self._samples[index]
        log.info(f"ScreenSpotPro sample {index}: {entry}")
        img_path = self._dataset_path / "images" / entry["img_filename"]
        image = Image.open(img_path).convert("RGB")

        # Convert bbox [x1, y1, x2, y2] pixels → RectShape (x, y, w, h) in 0-1000 grid
        bbox = entry["bbox"]
        img_w, img_h = entry["img_size"]
        x1_norm = int(bbox[0] / img_w * 1000)
        y1_norm = int(bbox[1] / img_h * 1000)
        x2_norm = int(bbox[2] / img_w * 1000)
        y2_norm = int(bbox[3] / img_h * 1000)
        gt_shape: RectShape = {
            "type": "rect",
            "geometry": (x1_norm, y1_norm, x2_norm - x1_norm, y2_norm - y1_norm),
            "color": "#00ff00",
        }

        prompt = self._prompt_template.format(instruction=entry["instruction"])

        return Sample(
            image=image,
            prompt=prompt,
            ground_truth_shapes=[gt_shape],
        )


class I2EBenchDataLoader(DataLoader):
    """Loads samples from the I2E-Bench dataset.

    Expects a dataset_path pointing to the root folder containing
    'annotations.jsonl' and 'screenshot/' subfolders.
    """

    def __init__(
        self,
        dataset_path: str,
        prompt_template: str = "Locate the element: {instruction}",
    ):
        self._dataset_path = Path(dataset_path)
        self._prompt_template = prompt_template
        self._samples: list[dict] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        annotations_file = self._dataset_path / "annotations.jsonl"
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

        with open(annotations_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._samples.append(json.loads(line))

        log.info(
            f"I2EBench: loaded {len(self._samples)} samples from {annotations_file}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def get_sample(self, index: int) -> Sample:
        if index < 0 or index >= len(self._samples):
            raise IndexError(f"Index {index} out of range (0-{len(self._samples)-1})")

        entry = self._samples[index]
        img_path = self._dataset_path / entry["image"]
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # bbox is [x1, y1, x2, y2] in pixel coords → normalize to 0-1000 grid
        bbox = entry["bounding_box"]
        x1_norm = int(bbox[0] / img_w * 1000)
        y1_norm = int(bbox[1] / img_h * 1000)
        x2_norm = int(bbox[2] / img_w * 1000)
        y2_norm = int(bbox[3] / img_h * 1000)
        gt_shape: RectShape = {
            "type": "rect",
            "geometry": (x1_norm, y1_norm, x2_norm - x1_norm, y2_norm - y1_norm),
            "color": "#00ff00",
        }

        prompt = self._prompt_template.format(instruction=entry["instruction"])

        return Sample(
            image=image,
            prompt=prompt,
            ground_truth_shapes=[gt_shape],
        )


class SyntheticRectangleDataLoader(DataLoader):
    """Generates images with one random black rectangle on white background.

    Used for calibration — testing pure localization capability.
    """

    def __init__(
        self,
        num_samples: int = 50,
        min_size: float = 0.05,
        max_size: float = 0.4,
        image_width: int = 1920,
        image_height: int = 1080,
        seed: int = 42,
    ):
        self._num_samples = num_samples
        self._min_size = min_size  # fraction of image dimension
        self._max_size = max_size
        self._image_width = image_width
        self._image_height = image_height
        self._rng = random.Random(seed)
        self._samples = self._generate_samples()

    def _generate_samples(self) -> list[dict]:
        samples = []
        for _ in range(self._num_samples):
            # Random width and height as fraction of image
            w_frac = self._rng.uniform(self._min_size, self._max_size)
            h_frac = self._rng.uniform(self._min_size, self._max_size)
            # Random position (ensuring rectangle fits within image)
            max_x = 1.0 - w_frac
            max_y = 1.0 - h_frac
            x_frac = self._rng.uniform(0, max_x)
            y_frac = self._rng.uniform(0, max_y)
            samples.append(
                {
                    "x": x_frac,
                    "y": y_frac,
                    "w": w_frac,
                    "h": h_frac,
                }
            )
        return samples

    def __len__(self) -> int:
        return self._num_samples

    def get_sample(self, index: int) -> Sample:
        if index < 0 or index >= self._num_samples:
            raise IndexError(f"Index {index} out of range (0-{self._num_samples - 1})")

        s = self._samples[index]
        img = Image.new("RGB", (self._image_width, self._image_height), "white")
        draw = ImageDraw.Draw(img)

        # Pixel coordinates
        x1 = int(s["x"] * self._image_width)
        y1 = int(s["y"] * self._image_height)
        x2 = int((s["x"] + s["w"]) * self._image_width)
        y2 = int((s["y"] + s["h"]) * self._image_height)
        draw.rectangle([x1, y1, x2, y2], fill="black")

        # Ground truth in 0-1000 grid (xyxy → xywh for RectShape)
        x1_norm = int(s["x"] * 1000)
        y1_norm = int(s["y"] * 1000)
        w_norm = int(s["w"] * 1000)
        h_norm = int(s["h"] * 1000)
        gt_shape: RectShape = {
            "type": "rect",
            "geometry": (x1_norm, y1_norm, w_norm, h_norm),
            "color": "#00ff00",
        }

        return Sample(
            image=img,
            prompt="Locate the black rectangle. Return the bounding box as JSON.",
            ground_truth_shapes=[gt_shape],
        )


# Registry: data_loader name in config → class
DATA_LOADER_REGISTRY: dict[str, type[DataLoader]] = {
    "screen_grab": ScreenGrabDataLoader,
    "screenspot_pro": ScreenSpotProDataLoader,
    "i2e_bench": I2EBenchDataLoader,
    "synthetic_rectangle": SyntheticRectangleDataLoader,
}
