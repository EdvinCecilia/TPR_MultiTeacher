"""Model package for TPR (Task-Performance-based Routing for multi-teacher)."""

from .student import StudentModel
from .teachers import load_teacher_full_models, TeacherFullModel
from .tpr_routing import TPR, TaskPerformanceRouter

__all__ = [
    "StudentModel",
    "load_teacher_full_models",
    "TeacherFullModel",
    "TPR",
    "TaskPerformanceRouter",
]
