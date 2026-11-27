"""Frame alignments saved by the live stack, so the stack command does not need to match stars again."""
import json
from pathlib import Path

import numpy as np
from skimage.transform import SimilarityTransform

FILENAME = "alignment.json"


def save(directory, reference, matrices):
    """Save, for each frame name, the matrix that maps its binned pixels onto those of the reference frame."""
    data = {
        "reference": reference,
        "frames": {name: np.asarray(matrix, dtype=float).tolist() for name, matrix in matrices.items()},
    }
    path = Path(directory) / FILENAME
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=1))
    # Replaced in one step, so a reader never sees half a file
    temporary.replace(path)
    return path


class Alignment:
    def __init__(self, reference, matrices):
        self.reference = reference
        self.matrices = {name: np.array(matrix, dtype=float) for name, matrix in matrices.items()}

    def transform(self, source, target):
        """Return the transform from one frame's binned pixels to another's, or None unless both were aligned."""
        if source not in self.matrices or target not in self.matrices:
            return None
        # Through the shared reference: source to reference, then reference to target
        return SimilarityTransform(matrix=np.linalg.inv(self.matrices[target]) @ self.matrices[source])


def load(directory):
    """Return the saved alignment of a directory's frames, or None when there is none or it cannot be read."""
    path = Path(directory) / FILENAME
    try:
        data = json.loads(path.read_text())
        matrices = data["frames"]
        alignment = Alignment(str(data["reference"]), matrices)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if any(matrix.shape != (3, 3) or not np.isfinite(matrix).all() for matrix in alignment.matrices.values()):
        return None
    return alignment
