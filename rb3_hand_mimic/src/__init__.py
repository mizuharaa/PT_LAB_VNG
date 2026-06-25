"""rb3_hand_mimic -- real-time robotic hand mimic for the Qualcomm RB3 Gen 2.

The package is intentionally modular so individual stages of the pipeline
(camera capture, hand tracking, gesture mapping, transform, smoothing, safety,
hand control) can be developed, tested, and swapped independently.
"""

__version__ = "1.0.0"
