"""
Generate synthetic star-schema data warehouse for the Master's Recommendation System.
Creates: dim_program.csv, dim_skill.csv, dim_interest.csv, fact_student.csv
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

RANDOM_SEED = 42
N_STUDENTS = 140

PROGRAMS = [
    {
        "program_id": "P01",
        "program_name": "M.Tech Computer Science & Engineering",
        "field": "Computer Science",
        "duration_years": 2,
        "example_universities": "IIT Bombay; NIT Trichy; IIIT Hyderabad",
        "region": "India",
        "category": "Technical",
    },
    {
        "program_id": "P02",
        "program_name": "M.Tech Artificial Intelligence",
        "field": "Artificial Intelligence",
        "duration_years": 2,
        "example_universities": "IIIT Hyderabad; IIT Madras; IISc Bangalore",
        "region": "India",
        "category": "Technical",
    },
    {
        "program_id": "P03",
        "program_name": "M.Sc Data Science",
        "field": "Data Science",
        "duration_years": 2,
        "example_universities": "IIT Madras; CMI; University of Hyderabad",
        "region": "India",
        "category": "Technical",
    },
    {
        "program_id": "P04",
        "program_name": "MS Computer Science",
        "field": "Computer Science",
        "duration_years": 2,
        "example_universities": "Stanford; Carnegie Mellon; Georgia Tech",
        "region": "Abroad",
        "category": "Technical",
    },
    {
        "program_id": "P05",
        "program_name": "MS Data Science",
        "field": "Data Science",
        "duration_years": 1.5,
        "example_universities": "Columbia; NYU; University of Michigan",
        "region": "Abroad",
        "category": "Technical",
    },
    {
        "program_id": "P06",
        "program_name": "MBA Business Analytics",
        "field": "Analytics / Management",
        "duration_years": 2,
        "example_universities": "ISB; IIM Bangalore; SPJIMR",
        "region": "India",
        "category": "Management",
    },
    {
        "program_id": "P07",
        "program_name": "M.Tech Cybersecurity",
        "field": "Cybersecurity",
        "duration_years": 2,
        "example_universities": "NIT Surathkal; IIIT Delhi; Amrita",
        "region": "India",
        "category": "Technical",
    },
    {
        "program_id": "P08",
        "program_name": "M.Des Human-Computer Interaction",
        "field": "HCI / Design",
        "duration_years": 2,
        "example_universities": "IIIT Delhi; IIT Guwahati; IDC IIT Bombay",
        "region": "India",
        "category": "Design",
    },
    {
        "program_id": "P09",
        "program_name": "M.Tech Software Engineering",
        "field": "Software Engineering",
        "duration_years": 2,
        "example_universities": "BITS Pilani; VIT; IIIT Bangalore",
        "region": "India",
        "category": "Technical",
    },
    {
        "program_id": "P10",
        "program_name": "MS Cybersecurity",
        "field": "Cybersecurity",
        "duration_years": 1.5,
        "example_universities": "Northeastern; NYU Tandon; Purdue",
        "region": "Abroad",
        "category": "Technical",
    },
]

SKILLS = [
    {"skill_id": "S01", "skill_name": "Python"},
    {"skill_id": "S02", "skill_name": "Java"},
    {"skill_id": "S03", "skill_name": "SQL"},
    {"skill_id": "S04", "skill_name": "Machine Learning"},
    {"skill_id": "S05", "skill_name": "Web Development"},
    {"skill_id": "S06", "skill_name": "Cloud Computing"},
    {"skill_id": "S07", "skill_name": "Data Visualization"},
    {"skill_id": "S08", "skill_name": "Cybersecurity"},
    {"skill_id": "S09", "skill_name": "UI/UX Design"},
    {"skill_id": "S10", "skill_name": "Deep Learning"},
    {"skill_id": "S11", "skill_name": "C++"},
    {"skill_id": "S12", "skill_name": "DevOps"},
]

INTERESTS = [
    {"interest_id": "I01", "interest_name": "Artificial Intelligence"},
    {"interest_id": "I02", "interest_name": "Data Science & Analytics"},
    {"interest_id": "I03", "interest_name": "Software Development"},
    {"interest_id": "I04", "interest_name": "Cybersecurity"},
    {"interest_id": "I05", "interest_name": "Product & UX Design"},
    {"interest_id": "I06", "interest_name": "Cloud & DevOps"},
    {"interest_id": "I07", "interest_name": "Business & Strategy"},
]

CAREER_GOALS = [
    "Research Scientist",
    "Data Scientist",
    "Software Engineer",
    "ML Engineer",
    "Security Analyst",
    "Product Manager",
    "UX Designer",
    "Cloud Architect",
    "Business Analyst",
]

UG_BRANCHES = [
    "CSE",
    "IT",
    "ECE",
    "EEE",
    "Mechanical",
    "Mathematics",
    "Statistics",
]

# Program affinity: preferred interest, career goals, skill pools, cgpa bias
PROGRAM_PROFILES = {
    "P01": {
        "interests": ["I03", "I01", "I06"],
        "goals": ["Software Engineer", "Research Scientist", "Cloud Architect"],
        "skills": ["Python", "Java", "C++", "SQL", "Cloud Computing", "Web Development"],
        "cgpa_mean": 8.2,
        "projects_mean": 4,
        "internships_mean": 2,
        "branches": ["CSE", "IT", "ECE"],
    },
    "P02": {
        "interests": ["I01", "I02"],
        "goals": ["ML Engineer", "Research Scientist", "Data Scientist"],
        "skills": ["Python", "Machine Learning", "Deep Learning", "SQL", "Data Visualization"],
        "cgpa_mean": 8.5,
        "projects_mean": 5,
        "internships_mean": 2,
        "branches": ["CSE", "IT", "ECE", "Mathematics"],
    },
    "P03": {
        "interests": ["I02", "I01"],
        "goals": ["Data Scientist", "Business Analyst", "ML Engineer"],
        "skills": ["Python", "SQL", "Machine Learning", "Data Visualization"],
        "cgpa_mean": 8.0,
        "projects_mean": 4,
        "internships_mean": 2,
        "branches": ["CSE", "IT", "Statistics", "Mathematics"],
    },
    "P04": {
        "interests": ["I03", "I01", "I06"],
        "goals": ["Software Engineer", "Research Scientist", "ML Engineer"],
        "skills": ["Python", "Java", "C++", "Machine Learning", "Cloud Computing", "Web Development"],
        "cgpa_mean": 8.8,
        "projects_mean": 5,
        "internships_mean": 2,
        "branches": ["CSE", "IT", "ECE"],
    },
    "P05": {
        "interests": ["I02", "I01"],
        "goals": ["Data Scientist", "ML Engineer"],
        "skills": ["Python", "SQL", "Machine Learning", "Deep Learning", "Data Visualization"],
        "cgpa_mean": 8.6,
        "projects_mean": 5,
        "internships_mean": 2,
        "branches": ["CSE", "IT", "Statistics", "Mathematics"],
    },
    "P06": {
        "interests": ["I07", "I02"],
        "goals": ["Product Manager", "Business Analyst"],
        "skills": ["SQL", "Data Visualization", "Python", "Web Development"],
        "cgpa_mean": 7.8,
        "projects_mean": 3,
        "internships_mean": 2,
        "branches": ["CSE", "IT", "ECE", "Mechanical", "EEE"],
    },
    "P07": {
        "interests": ["I04"],
        "goals": ["Security Analyst", "Software Engineer"],
        "skills": ["Cybersecurity", "Python", "C++", "Java", "Cloud Computing"],
        "cgpa_mean": 7.9,
        "projects_mean": 4,
        "internships_mean": 1,
        "branches": ["CSE", "IT", "ECE"],
    },
    "P08": {
        "interests": ["I05", "I03"],
        "goals": ["UX Designer", "Product Manager"],
        "skills": ["UI/UX Design", "Web Development", "Python", "Java"],
        "cgpa_mean": 7.6,
        "projects_mean": 4,
        "internships_mean": 2,
        "branches": ["CSE", "IT", "ECE"],
    },
    "P09": {
        "interests": ["I03", "I06"],
        "goals": ["Software Engineer", "Cloud Architect"],
        "skills": ["Java", "Web Development", "DevOps", "Cloud Computing", "SQL", "Python"],
        "cgpa_mean": 7.7,
        "projects_mean": 4,
        "internships_mean": 2,
        "branches": ["CSE", "IT"],
    },
    "P10": {
        "interests": ["I04", "I06"],
        "goals": ["Security Analyst", "Cloud Architect"],
        "skills": ["Cybersecurity", "Python", "Cloud Computing", "DevOps", "C++"],
        "cgpa_mean": 8.3,
        "projects_mean": 4,
        "internships_mean": 2,
        "branches": ["CSE", "IT", "ECE"],
    },
}

ALL_SKILL_NAMES = [s["skill_name"] for s in SKILLS]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pick_skills(preferred: list[str], rng: random.Random) -> list[str]:
    n = rng.randint(3, 6)
    core = rng.sample(preferred, k=min(len(preferred), max(2, n - 1)))
    extras = [s for s in ALL_SKILL_NAMES if s not in core]
    while len(core) < n and extras:
        core.append(extras.pop(rng.randrange(len(extras))))
    return sorted(set(core))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate() -> None:
    rng = random.Random(RANDOM_SEED)
    out_dir = Path(__file__).resolve().parent

    write_csv(
        out_dir / "dim_program.csv",
        PROGRAMS,
        [
            "program_id",
            "program_name",
            "field",
            "duration_years",
            "example_universities",
            "region",
            "category",
        ],
    )
    write_csv(out_dir / "dim_skill.csv", SKILLS, ["skill_id", "skill_name"])
    write_csv(out_dir / "dim_interest.csv", INTERESTS, ["interest_id", "interest_name"])

    interest_lookup = {i["interest_id"]: i["interest_name"] for i in INTERESTS}
    program_ids = list(PROGRAM_PROFILES.keys())
    # Slightly uneven distribution so popular programs appear more often
    weights = [14, 16, 15, 10, 10, 12, 11, 10, 14, 8]

    fact_rows = []
    for i in range(1, N_STUDENTS + 1):
        program_id = rng.choices(program_ids, weights=weights, k=1)[0]
        profile = PROGRAM_PROFILES[program_id]

        cgpa = clamp(rng.gauss(profile["cgpa_mean"], 0.45), 6.0, 10.0)
        projects = int(clamp(round(rng.gauss(profile["projects_mean"], 1.2)), 0, 10))
        internships = int(clamp(round(rng.gauss(profile["internships_mean"], 0.8)), 0, 5))
        branch = rng.choice(profile["branches"])
        interest_id = rng.choice(profile["interests"])
        career_goal = rng.choice(profile["goals"])
        skills = pick_skills(profile["skills"], rng)

        fact_rows.append(
            {
                "student_id": f"STU{i:03d}",
                "cgpa": round(cgpa, 2),
                "ug_branch": branch,
                "projects_count": projects,
                "internships_count": internships,
                "skills": "|".join(skills),
                "interest_id": interest_id,
                "area_of_interest": interest_lookup[interest_id],
                "career_goal": career_goal,
                "program_id": program_id,
            }
        )

    write_csv(
        out_dir / "fact_student.csv",
        fact_rows,
        [
            "student_id",
            "cgpa",
            "ug_branch",
            "projects_count",
            "internships_count",
            "skills",
            "interest_id",
            "area_of_interest",
            "career_goal",
            "program_id",
        ],
    )
    print(f"Generated {N_STUDENTS} students and dimension tables in {out_dir}")


if __name__ == "__main__":
    generate()
