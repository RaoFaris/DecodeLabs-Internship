# Tech Stack Recommender — Project 3: AI Recommendation Logic

## Overview
A content-based recommendation system that suggests the most relevant job roles based on a user's skills. Built as Project 3 of the DecodeLabs AI Industrial Training Kit (Batch 2026).

## Objective
Design and implement a recommendation engine that:
- Accepts a user's skill set as input
- Matches those skills against a dataset of job roles using similarity logic
- Returns the Top-3 most relevant job roles, ranked by match score

This project demonstrates logic building, pattern matching, and content-based recommendation concepts — without using any machine learning libraries.

## Technologies
- Python 3 (standard library only)
- `csv` — reading the skills dataset
- `math` — TF-IDF and cosine similarity calculations

No external/ML libraries (no scikit-learn, no numpy, no pandas) are used. All vector math is implemented from scratch.

## Recommendation Logic
The system uses **Content-Based Filtering**: it compares the user's profile directly against each job role's attributes (skills), rather than comparing the user to other users.

The pipeline follows 4 stages:

1. **Ingestion** — Capture the user's skills (minimum 3 required).
2. **Scoring** — Convert both the user profile and every job role into **TF-IDF weighted vectors** over a shared skill vocabulary, then calculate **Cosine Similarity** between the user vector and each job role vector.
   - **TF (Term Frequency)** — how prominent a skill is within a given profile.
   - **IDF (Inverse Document Frequency)** — down-weights skills that appear across many roles (e.g. "python", "git") and up-weights rarer, more specific skills (e.g. "kubernetes").
   - **Cosine Similarity** — measures the angle between the user's vector and each role's vector, so the *orientation* of interests matters more than the raw count of skills.
3. **Sorting** — Job roles are ranked in descending order of similarity score.
4. **Filtering** — Only the Top-3 highest-scoring roles are returned to the user.

## How to Run
1. Make sure `tech_stack_recommender.py` and `raw_skills.csv` are in the same folder.
2. Run:
   ```
   python3 tech_stack_recommender.py
   ```
3. Enter at least 3 skills, separated by commas, when prompted.
4. View your Top-3 recommended job roles with match percentages.
5. Choose `y` to try another skill set, or `n` to exit.

## Example Interaction
```
Enter at least 3 of your skills, separated by commas
(e.g. Python, Cloud Computing, Automation): Python, Cloud Computing, Automation

Top recommended career paths for you:
  1. Cloud Architect  (match score: 62.0%)
  2. Systems Administrator  (match score: 18.5%)
  3. Data Scientist  (match score: 14.6%)

Try another set of skills? (y/n): n
Goodbye!
```

If none of the entered skills are found in the dataset (e.g. typos or unrelated terms), the system falls back to a friendly message instead of showing meaningless 0% matches.

## Project Structure
```
.
├── tech_stack_recommender.py   # Main program: TF-IDF + Cosine Similarity engine
├── raw_skills.csv              # Dataset mapping job roles to their associated skills
└── README.md                   # This file
```

## Learning Outcomes
- Understood the difference between collaborative filtering and content-based filtering
- Implemented TF-IDF weighting from first principles (no ML libraries)
- Implemented cosine similarity as a distance/similarity metric and understood why it's preferred over Euclidean distance for this kind of data
- Built a full Input → Process → Output (IPO) recommendation pipeline: Ingestion, Scoring, Sorting, Filtering
- Handled edge cases like unmatched/unknown user input (a simplified cold-start scenario)
