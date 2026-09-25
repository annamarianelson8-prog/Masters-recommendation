"""
KNN-based Master's program recommender with similarity-weighted voting
and human-readable contributing factors.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.preprocess import FeaturePipeline, skills_to_binary

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "student_profiles_clean.csv"


@dataclass
class RecommendationResult:
    recommendations: list[dict[str, Any]]
    neighbors: pd.DataFrame
    explanations: dict[str, list[str]]


class MastersRecommender:
    def __init__(self, profiles: pd.DataFrame | None = None):
        if profiles is None:
            if not DATA_PATH.exists():
                raise FileNotFoundError(
                    f"Clean dataset not found at {DATA_PATH}. "
                    "Run data/generate_dataset.py then etl.py first."
                )
            profiles = pd.read_csv(DATA_PATH)

        self.profiles = profiles.reset_index(drop=True)
        self.pipeline = FeaturePipeline().fit(self.profiles)
        self._nn: NearestNeighbors | None = None
        self._fitted_k: int | None = None

    def _fit_nn(self, k: int) -> NearestNeighbors:
        n_neighbors = min(k, len(self.profiles))
        if self._nn is None or self._fitted_k != n_neighbors:
            self._nn = NearestNeighbors(
                n_neighbors=n_neighbors, metric="euclidean", algorithm="auto"
            )
            self._nn.fit(self.pipeline.X_train)
            self._fitted_k = n_neighbors
        return self._nn

    def recommend(
        self,
        student: dict,
        k: int = 5,
        region_filter: str | None = None,
        category_filter: str | None = None,
        top_n: int = 3,
    ) -> RecommendationResult:
        query = self.pipeline.transform_row(student)
        nn = self._fit_nn(k)
        distances, indices = nn.kneighbors(query, n_neighbors=min(k, len(self.profiles)))
        distances = distances[0]
        indices = indices[0]

        neighbors = self.profiles.iloc[indices].copy()
        neighbors["distance"] = distances
        # Similarity: inverse distance (add epsilon to avoid div-by-zero)
        neighbors["similarity"] = 1.0 / (neighbors["distance"] + 1e-6)

        if region_filter and region_filter != "All":
            neighbors = neighbors[neighbors["region"] == region_filter]
        if category_filter and category_filter != "All":
            neighbors = neighbors[neighbors["category"] == category_filter]

        if neighbors.empty:
            # Fall back to unfiltered neighbors if filters wipe everything
            neighbors = self.profiles.iloc[indices].copy()
            neighbors["distance"] = distances
            neighbors["similarity"] = 1.0 / (neighbors["distance"] + 1e-6)

        vote_scores: dict[str, float] = defaultdict(float)
        program_meta: dict[str, dict] = {}
        for _, row in neighbors.iterrows():
            prog = row["chosen_masters_program"]
            vote_scores[prog] += float(row["similarity"])
            if prog not in program_meta:
                program_meta[prog] = {
                    "field": row.get("field", ""),
                    "duration_years": row.get("duration_years", ""),
                    "example_universities": row.get("example_universities", ""),
                    "region": row.get("region", ""),
                    "category": row.get("category", ""),
                }

        ranked = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)
        total = sum(vote_scores.values()) or 1.0

        recommendations = []
        for program, score in ranked[:top_n]:
            meta = program_meta.get(program, {})
            recommendations.append(
                {
                    "program": program,
                    "score": round(score, 4),
                    "confidence_pct": round(100.0 * score / total, 1),
                    **meta,
                }
            )

        explanations = {
            rec["program"]: self._explain(student, neighbors, rec["program"])
            for rec in recommendations
        }

        display_cols = [
            "student_id",
            "cgpa",
            "ug_branch",
            "projects_count",
            "internships_count",
            "skills",
            "area_of_interest",
            "career_goal",
            "chosen_masters_program",
            "distance",
            "similarity",
        ]
        neighbor_view = neighbors[display_cols].sort_values("distance").reset_index(drop=True)
        neighbor_view["similarity"] = neighbor_view["similarity"].round(4)
        neighbor_view["distance"] = neighbor_view["distance"].round(4)

        return RecommendationResult(
            recommendations=recommendations,
            neighbors=neighbor_view,
            explanations=explanations,
        )

    def _explain(
        self, student: dict, neighbors: pd.DataFrame, program: str
    ) -> list[str]:
        subset = neighbors[neighbors["chosen_masters_program"] == program]
        if subset.empty:
            return ["Recommended based on overall similarity to historical students."]

        factors: list[str] = []
        user_skills = set(skills_to_binary(student.get("skills", [])).keys())
        # Only skills the user actually has
        user_has = {
            s
            for s, v in skills_to_binary(student.get("skills", [])).items()
            if v == 1
        }

        avg_cgpa = subset["cgpa"].mean()
        user_cgpa = float(student["cgpa"])
        cgpa_diff = abs(avg_cgpa - user_cgpa)
        factors.append(
            f"Your CGPA ({user_cgpa:.2f}) is close to similar students who chose this "
            f"program (avg {avg_cgpa:.2f}, diff={cgpa_diff:.2f})."
        )

        shared_skill_counts: dict[str, int] = defaultdict(int)
        for skills_str in subset["skills"]:
            their = {
                s
                for s, v in skills_to_binary(str(skills_str)).items()
                if v == 1
            }
            for s in user_has & their:
                shared_skill_counts[s] += 1
        if shared_skill_counts:
            top_shared = sorted(
                shared_skill_counts.items(), key=lambda x: x[1], reverse=True
            )[:4]
            names = ", ".join(s for s, _ in top_shared)
            factors.append(f"Overlapping technical skills with similar alumni: {names}.")

        same_interest = (
            subset["area_of_interest"] == student["area_of_interest"]
        ).sum()
        if same_interest:
            factors.append(
                f"{int(same_interest)} of {len(subset)} similar student(s) share your "
                f"area of interest ({student['area_of_interest']})."
            )

        same_goal = (subset["career_goal"] == student["career_goal"]).sum()
        if same_goal:
            factors.append(
                f"{int(same_goal)} of {len(subset)} similar student(s) share your "
                f"career goal ({student['career_goal']})."
            )

        avg_proj = subset["projects_count"].mean()
        avg_intern = subset["internships_count"].mean()
        factors.append(
            f"Project/internship profile is similar "
            f"(you: {student['projects_count']} projects, "
            f"{student['internships_count']} internships; "
            f"neighbors avg: {avg_proj:.1f} projects, {avg_intern:.1f} internships)."
        )

        # Silence unused variable warning for user_skills if needed
        _ = user_skills
        return factors
