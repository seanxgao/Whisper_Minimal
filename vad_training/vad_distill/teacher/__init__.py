"""Teacher VAD modules."""

from vad_distill.teacher.frame_level_teacher import FrameLevelTeacher
from vad_distill.teacher.teacher_silero import get_vad_probs, load_silero

__all__ = ["FrameLevelTeacher", "load_silero", "get_vad_probs"]
