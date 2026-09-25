"""
Feature encoding and scaling for KNN Master's recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

SKILL_COLUMNS = [
    "Python",
    "Java",
    "SQL",
    "Machine Learning",
    "Web Development",
    "Cloud Computing",
    "Data Visualization",
    "Cybersecurity",
    "UI/UX Design",
    "Deep Learning",
    "C++",
    "DevOps",
]

CATEGORICAL_COLS = ["ug_branch", "area_of_interest", "career_goal"]
NUMERIC_COLS = ["cgpa", "projects_count", "internships_count"]


def skills_to_binary(skills: Iterable[str] | str | None) -> dict[str, int]:
    if skills is None:
        skill_set: set[str] = set()
    elif isinstance(skills, str):
        skill_set = {s.strip() for s in skills.split("|") if s.strip()}
    else:
        skill_set = {str(s).strip() for s in skills if str(s).strip()}
    return {col: int(col in skill_set) for col in SKILL_COLUMNS}


def expand_skills_frame(df: pd.DataFrame) -> pd.DataFrame:
    skill_rows = df["skills"].apply(skills_to_binary).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), skill_rows], axis=1)


@dataclass
class FeaturePipeline:
    """Fit on historical profiles; transform new student rows the same way."""

    encoder: OneHotEncoder | None = None
    scaler: MinMaxScaler | None = None
    feature_names_: list[str] = field(default_factory=list)
    profiles_: pd.DataFrame | None = None

    def fit(self, profiles: pd.DataFrame) -> "FeaturePipeline":
        data = expand_skills_frame(profiles.copy())
        self.profiles_ = data

        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_matrix = self.encoder.fit_transform(data[CATEGORICAL_COLS])
        cat_names = list(self.encoder.get_feature_names_out(CATEGORICAL_COLS))

        self.scaler = MinMaxScaler()
        num_matrix = self.scaler.fit_transform(data[NUMERIC_COLS])

        skill_matrix = data[SKILL_COLUMNS].to_numpy(dtype=float)
        X = np.hstack([num_matrix, cat_matrix, skill_matrix])
        self.feature_names_ = NUMERIC_COLS + cat_names + SKILL_COLUMNS
        self._X_train = X
        return self

    @property
    def X_train(self) -> np.ndarray:
        if not hasattr(self, "_X_train"):
            raise RuntimeError("Pipeline not fitted")
        return self._X_train

    def transform_row(self, student: dict) -> np.ndarray:
        if self.encoder is None or self.scaler is None:
            raise RuntimeError("Pipeline not fitted")

        row = {
            "cgpa": float(student["cgpa"]),
            "projects_count": int(student["projects_count"]),
            "internships_count": int(student["internships_count"]),
            "ug_branch": student["ug_branch"],
            "area_of_interest": student["area_of_interest"],
            "career_goal": student["career_goal"],
            "skills": student.get("skills", []),
        }
        frame = pd.DataFrame([row])
        if isinstance(row["skills"], list):
            frame["skills"] = "|".join(row["skills"])
        frame = expand_skills_frame(frame)

        num = self.scaler.transform(frame[NUMERIC_COLS])
        cat = self.encoder.transform(frame[CATEGORICAL_COLS])
        skills = frame[SKILL_COLUMNS].to_numpy(dtype=float)
        return np.hstack([num, cat, skills])

    def transform_profiles(self) -> np.ndarray:
        return self.X_train
