import csv
import math

DATA_FILE = "raw_skills.csv"


# STAGE 1: LOAD THE DATA
def load_job_roles(filepath):
    """
    Reads raw_skills.csv and turns it into a dictionary like:
        {
            "Data Scientist": ["python", "sql", "machine learning", ...],
            "DevOps Engineer": ["aws", "docker", ...],
            ...
        }
    We lowercase everything so "Python" and "python" are treated
    as the same skill (this is the "normalization" step).
    """
    roles = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role_name = row["role"].strip()
            skills = [s.strip().lower() for s in row["skills"].split(";")]
            roles[role_name] = skills
    return roles


# STAGE 2: TAKE USER INPUT (INGESTION)
def get_user_skills():
    """
    Official requirement: minimum 3 user inputs.
    We ask for a comma-separated list and normalize it the same
    way we normalized the dataset (lowercase, no extra spaces).
    """
    while True:
        raw = input(
            "\nEnter at least 3 of your skills, separated by commas\n"
            "(e.g. Python, Cloud Computing, Automation): "
        )
        skills = [s.strip().lower() for s in raw.split(",") if s.strip()]
        if len(skills) >= 3:
            return skills
        print(f"That's only {len(skills)} skill(s). Please enter at least 3.")


# STAGE 3: BUILD THE SHARED VOCABULARY
def build_vocabulary(roles, user_skills):
    """
    TF-IDF needs every 'document' (each job role, AND the user
    profile) expressed as a vector over the SAME list of skills.
    This function collects every unique skill mentioned anywhere
    (in the dataset or in the user's input) into one master list.
    This is the "vector mapping" step from the slides -- without
    a shared vocabulary, the cosine math would break.
    """
    vocab = set()
    for skill_list in roles.values():
        vocab.update(skill_list)
    vocab.update(user_skills)
    return sorted(vocab)


# STAGE 4: TF-IDF WEIGHTING
def compute_tf(document_skills, vocabulary):
    """
    Term Frequency (TF): how often each vocabulary skill appears
    in this particular document, divided by the document length.
    For our data, each skill either appears once or not at all,
    so TF is just 1/len or 0.
    """
    tf_vector = []
    total_terms = len(document_skills)
    for term in vocabulary:
        count = document_skills.count(term)
        tf_vector.append(count / total_terms if total_terms > 0 else 0)
    return tf_vector


def compute_idf(all_documents, vocabulary):
    """
    Inverse Document Frequency (IDF): penalizes skills that show
    up in almost every job role (e.g. "git", "linux") and rewards
    skills that are rare/specific (e.g. "kubernetes").

    Formula from the slides:
        IDF = log( total_documents / documents_containing_term )
    """
    total_docs = len(all_documents)
    idf_vector = []
    for term in vocabulary:
        docs_with_term = sum(1 for doc in all_documents if term in doc)
        # +1 avoids division-by-zero if a term somehow appears nowhere
        idf = math.log(total_docs / (docs_with_term + 1))
        idf_vector.append(idf)
    return idf_vector


def compute_tfidf(tf_vector, idf_vector):
    """TF-IDF weight = TF * IDF, applied element-wise."""
    return [tf * idf for tf, idf in zip(tf_vector, idf_vector)]


# STAGE 5: COSINE SIMILARITY (THE SCORING ENGINE)
def cosine_similarity(vector_a, vector_b):
    """
    cos(theta) = (A . B) / (||A|| * ||B||)

    - Dot product (A . B): how much the two vectors point in the
      same direction.
    - ||A|| and ||B||: the magnitude (length) of each vector.

    Dividing by the magnitudes is what makes this "distance"
    metric ignore vector size and focus purely on orientation --
    which is exactly why the slides prefer it over Euclidean
    distance for this kind of text-like data.
    """
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # this is the "Cold Start" case from the slides
    return dot_product / (magnitude_a * magnitude_b)


# STAGE 6: THE 4-STEP RANKING PIPELINE (Ingestion->Scoring->Sorting->Filtering)
def recommend(user_skills, roles, top_n=3):
    all_documents = list(roles.values())
    vocabulary = build_vocabulary(roles, user_skills)

    # IDF is computed once, across every job-role document
    idf_vector = compute_idf(all_documents, vocabulary)

    # Build the user's own TF-IDF vector
    user_tf = compute_tf(user_skills, vocabulary)
    user_tfidf = compute_tfidf(user_tf, idf_vector)

    # SCORING: compare the user vector against every job role
    scores = []
    for role_name, skill_list in roles.items():
        role_tf = compute_tf(skill_list, vocabulary)
        role_tfidf = compute_tfidf(role_tf, idf_vector)
        similarity = cosine_similarity(user_tfidf, role_tfidf)
        scores.append((role_name, similarity))

    # SORTING: highest similarity first
    scores.sort(key=lambda pair: pair[1], reverse=True)

    # FILTERING: keep only the Top-N
    return scores[:top_n]


# STAGE 7: DISPLAY + REPEAT LOOP
def display_recommendations(results):
    print("\nTop recommended career paths for you:")
    if all(score == 0 for _, score in results):
        # Fallback for unmatched / unknown skills (Cold Start handling)
        print("  No strong matches found for those skills.")
        print("  Try broader terms like 'Python', 'Networking', or 'Testing'.")
        return
    for rank, (role, score) in enumerate(results, start=1):
        match_percent = round(score * 100, 1)
        print(f"  {rank}. {role}  (match score: {match_percent}%)")


def main():
    roles = load_job_roles(DATA_FILE)

    while True:
        user_skills = get_user_skills()
        results = recommend(user_skills, roles, top_n=3)
        display_recommendations(results)

        again = input("\nTry another set of skills? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main(), 