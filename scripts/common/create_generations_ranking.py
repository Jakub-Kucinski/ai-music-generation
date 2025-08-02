import json
from pathlib import Path
from typing import Any

import pandas as pd

# Global path variables

# Directory containing inner_similarity.jsonl, reference_similarity.jsonl, conditional_prefix_similarity.jsonl
STRUCTURAL_METRICS_DIR = Path(
    "data/04_generated/music21_bach_512_context_augmented/conditioned_4_bars/metrics/structure"
)

# Directory containing aesthetics.jsonl and wav_paths.jsonl
AESTHETICS_DIR = Path(
    "data/04_generated/music21_bach_512_context_augmented/conditioned_4_bars/metrics/audiobox_aesthetics"
)

# Directory where results will be saved
RESULTS_DIR = Path("data/04_generated/music21_bach_512_context_augmented/conditioned_4_bars/metrics/ranking")

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(filepath: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries."""
    data: list[dict[str, Any]] = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_identifier_from_path(path: str) -> str:
    """Extract the identifier (e.g., 'bwv103.6') from a path."""
    # Get the filename without extension
    filename = Path(path).stem

    # Handle WAV paths: file_sample_bwv103.6 -> bwv103.6
    if "file_sample_" in filename:
        return filename.replace("file_sample_", "")

    # Handle MIDI paths: sample_bwv121.6 -> bwv121.6
    elif "sample_" in filename:
        return filename.replace("sample_", "")

    # If neither pattern matches, return the filename as is
    return filename


def extract_similarities_as_dict(data: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Extract melodic and rhythmic mean_best_similarities as a dictionary keyed by identifier."""
    similarities_dict: dict[str, tuple[float, float]] = {}

    for entry in data:
        path = entry["path"]
        identifier = extract_identifier_from_path(path)
        melodic_sim = entry["melody"]["mean_best_similarities"]
        rhythmic_sim = entry["rhythm"]["mean_best_similarities"]
        similarities_dict[identifier] = (melodic_sim, rhythmic_sim)

    return similarities_dict


def create_rankings(df: pd.DataFrame, ranking_columns: list[str]) -> pd.DataFrame:
    """Create rankings for each metric and calculate final ranking."""
    # Create ranking columns
    for col in ranking_columns:
        # rank() gives 1 to the smallest value by default, so we use ascending=False
        # to give 1 to the largest value
        df[f"{col}_rank"] = df[col].rank(ascending=False, method="min")

    # Calculate sum of rankings (lower sum is better)
    rank_columns: list[str] = [f"{col}_rank" for col in ranking_columns]
    df["sum_of_ranks"] = df[rank_columns].sum(axis=1)

    # Create final ranking based on sum of ranks (lower sum gets better final rank)
    df["final_rank"] = df["sum_of_ranks"].rank(ascending=True, method="min").astype(int)

    return df


def save_outputs(df: pd.DataFrame) -> None:
    """Save various output files with ranking results."""
    # Sort by final rank
    df_sorted: pd.DataFrame = df.sort_values("final_rank")

    # Save detailed ranking results
    detailed_path = RESULTS_DIR / "detailed_ranking.csv"
    df_sorted.to_csv(detailed_path, index=False)
    print(f"Saved detailed ranking to '{detailed_path}'")

    # Save simple ranking file (just identifier and final rank)
    ranking_output: pd.DataFrame = df_sorted[["identifier", "final_rank"]].copy()
    final_ranking_path = RESULTS_DIR / "final_ranking.csv"
    ranking_output.to_csv(final_ranking_path, index=False)
    print(f"Saved final ranking to '{final_ranking_path}'")

    # Also save a simple text file with ranked identifiers
    ranked_paths_path = RESULTS_DIR / "ranked_identifiers.txt"
    with open(ranked_paths_path, "w") as f:
        f.write("Rank\tIdentifier\n")
        for _, row in ranking_output.iterrows():
            f.write(f"{row['final_rank']}\t{row['identifier']}\n")
    print(f"Saved ranked identifiers to '{ranked_paths_path}'")


def print_summary(df: pd.DataFrame, ranking_columns: list[str]) -> None:
    """Print summary statistics and correlations."""
    df_sorted: pd.DataFrame = df.sort_values("final_rank")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total number of files: {len(df)}")
    print("\nTop 5 files by final ranking:")
    print(df_sorted[["identifier", "final_rank", "sum_of_ranks"]].head())

    # Print correlation between metrics
    print("\nCorrelation between metrics:")
    correlation_df: pd.DataFrame = df[ranking_columns].corr()
    print(correlation_df)


def main() -> None:
    """Main function to process similarity and aesthetics data and create rankings."""
    # Read all data files
    print("Reading data files...")
    inner_sim_data: list[dict[str, Any]] = read_jsonl(STRUCTURAL_METRICS_DIR / "inner_similarity.jsonl")
    reference_sim_data: list[dict[str, Any]] = read_jsonl(STRUCTURAL_METRICS_DIR / "reference_similarity.jsonl")
    prefix_sim_data: list[dict[str, Any]] = read_jsonl(STRUCTURAL_METRICS_DIR / "conditional_prefix_similarity.jsonl")
    aesthetics_data: list[dict[str, Any]] = read_jsonl(AESTHETICS_DIR / "aesthetics.jsonl")
    wav_paths_data: list[dict[str, Any]] = read_jsonl(AESTHETICS_DIR / "wav_paths.jsonl")

    # Extract similarities as dictionaries keyed by identifier
    print("\nExtracting similarities...")
    inner_sim_dict: dict[str, tuple[float, float]] = extract_similarities_as_dict(inner_sim_data)
    ref_sim_dict: dict[str, tuple[float, float]] = extract_similarities_as_dict(reference_sim_data)
    prefix_sim_dict: dict[str, tuple[float, float]] = extract_similarities_as_dict(prefix_sim_data)

    # Debug: Print sample entries
    print(f"\nSample inner similarity entries: {list(inner_sim_dict.items())[:3]}")
    print(f"Sample reference similarity entries: {list(ref_sim_dict.items())[:3]}")
    print(f"Sample prefix similarity entries: {list(prefix_sim_dict.items())[:3]}")

    # Extract wav paths and their identifiers
    wav_paths: list[str] = [d["path"] for d in wav_paths_data]
    wav_identifiers: list[str] = [extract_identifier_from_path(path) for path in wav_paths]

    # Debug: Print sample WAV identifiers
    print(f"\nSample WAV identifiers: {wav_identifiers[:5]}")

    # Extract aesthetics scores (aligned with wav_paths)
    ce_scores: list[float] = [d["CE"] for d in aesthetics_data]
    cu_scores: list[float] = [d["CU"] for d in aesthetics_data]
    pc_scores: list[float] = [d["PC"] for d in aesthetics_data]
    pq_scores: list[float] = [d["PQ"] for d in aesthetics_data]

    # Build data for DataFrame by matching identifiers
    data_rows: list[dict[str, Any]] = []

    for i, (wav_path, identifier) in enumerate(zip(wav_paths, wav_identifiers)):
        row: dict[str, Any] = {
            "identifier": identifier,
            "original_wav_path": wav_path,
            "CE": ce_scores[i],
            "CU": cu_scores[i],
            "PC": pc_scores[i],
            "PQ": pq_scores[i],
        }

        # Look up similarity values using the identifier
        if identifier in inner_sim_dict:
            melodic, rhythmic = inner_sim_dict[identifier]
            row["inner_melodic_sim"] = melodic
            row["inner_rhythmic_sim"] = rhythmic
        else:
            row["inner_melodic_sim"] = None
            row["inner_rhythmic_sim"] = None

        if identifier in ref_sim_dict:
            melodic, rhythmic = ref_sim_dict[identifier]
            row["reference_melodic_sim"] = melodic
            row["reference_rhythmic_sim"] = rhythmic
        else:
            row["reference_melodic_sim"] = None
            row["reference_rhythmic_sim"] = None

        if identifier in prefix_sim_dict:
            melodic, rhythmic = prefix_sim_dict[identifier]
            row["prefix_melodic_sim"] = melodic
            row["prefix_rhythmic_sim"] = rhythmic
        else:
            row["prefix_melodic_sim"] = None
            row["prefix_rhythmic_sim"] = None

        data_rows.append(row)

    # Create DataFrame
    print("\nCreating DataFrame...")
    df: pd.DataFrame = pd.DataFrame(data_rows)

    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        print("\nWarning: Missing values detected:")
        print(missing_counts[missing_counts > 0])

        # Drop rows with missing similarity values
        similarity_columns = [
            "inner_melodic_sim",
            "inner_rhythmic_sim",
            "reference_melodic_sim",
            "reference_rhythmic_sim",
            "prefix_melodic_sim",
            "prefix_rhythmic_sim",
        ]
        df = df.dropna(subset=similarity_columns)
        print(f"\nAfter dropping rows with missing similarities: {len(df)} rows remaining")

    # Save the complete dataframe
    all_metrics_path = RESULTS_DIR / "all_metrics.csv"
    df.to_csv(all_metrics_path, index=False)
    print(f"\nSaved all metrics to '{all_metrics_path}'")

    # Define ranking columns
    ranking_columns: list[str] = [
        "inner_melodic_sim",
        "inner_rhythmic_sim",
        "reference_melodic_sim",
        "reference_rhythmic_sim",
        "prefix_melodic_sim",
        "prefix_rhythmic_sim",
        "CE",
        "CU",
        "PC",
        "PQ",
    ]

    # Create rankings
    print("\nCreating rankings...")
    df = create_rankings(df, ranking_columns)

    # Save outputs
    save_outputs(df)

    # Print summary
    print_summary(df, ranking_columns)


if __name__ == "__main__":
    main()
