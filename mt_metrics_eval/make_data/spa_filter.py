import pandas as pd
from collections import defaultdict, Counter


def filter_spa_suitable_systems(
    csv_summary_file, min_common_segments=10, min_pairs_per_system=6
):
    df = pd.read_csv(csv_summary_file)

    print(f"Analyzing SPA suitability from: {csv_summary_file}")
    print(f"Criteria:")
    print(f"  - Min common segments per pair: {min_common_segments}")
    print(f"  - Min valid pairs per system: {min_pairs_per_system}")
    print("=" * 80)

    results = {}

    for lp in sorted(df["language_pair"].unique()):
        print(f"\nLanguage Pair: {lp}")
        print("-" * 50)

        lp_data = df[df["language_pair"] == lp]

        suitable_pairs = lp_data[lp_data["common_segments"] >= min_common_segments]

        print(f"Total pairs: {len(lp_data)}")
        print(
            f"Suitable pairs (≥{min_common_segments} common segments): {len(suitable_pairs)}"
        )

        if len(suitable_pairs) == 0:
            print(f"❌ No suitable pairs found for {lp}")
            results[lp] = {
                "suitable_systems": [],
                "all_systems": sorted(
                    set(list(lp_data["system1"]) + list(lp_data["system2"]))
                ),
                "suitable_pairs_count": 0,
                "total_pairs_count": len(lp_data),
            }
            continue

        system_pair_counts = defaultdict(int)
        system_common_segment_stats = defaultdict(list)

        all_systems = set(list(lp_data["system1"]) + list(lp_data["system2"]))

        for _, row in suitable_pairs.iterrows():
            sys1, sys2 = row["system1"], row["system2"]
            common_segs = row["common_segments"]

            system_pair_counts[sys1] += 1
            system_pair_counts[sys2] += 1

            system_common_segment_stats[sys1].append(common_segs)
            system_common_segment_stats[sys2].append(common_segs)

        suitable_systems = []
        for system in all_systems:
            valid_pairs = system_pair_counts[system]
            if valid_pairs >= min_pairs_per_system:
                suitable_systems.append(system)

        suitable_systems = sorted(suitable_systems)

        print(f"Systems analysis:")
        print(f"  Total systems: {len(all_systems)}")
        print(
            f"  Systems with ≥{min_pairs_per_system} valid pairs: {len(suitable_systems)}"
        )
        print(f"\nSystem breakdown:")
        for system in sorted(all_systems):
            valid_pairs = system_pair_counts[system]
            if system in suitable_systems:
                avg_common = sum(system_common_segment_stats[system]) / len(
                    system_common_segment_stats[system]
                )
                min_common = min(system_common_segment_stats[system])
                max_common = max(system_common_segment_stats[system])
                print(
                    f"  ✅ {system:25s}: {valid_pairs:2d} pairs, common segs: {min_common:2d}-{max_common:2d} (avg {avg_common:.1f})"
                )
            else:
                print(f"  ❌ {system:25s}: {valid_pairs:2d} pairs (insufficient)")

        print(f"\n✅ Suitable systems for {lp} SPA analysis: {len(suitable_systems)}")
        for sys in suitable_systems:
            print(f"    {sys}")

        results[lp] = {
            "suitable_systems": suitable_systems,
            "all_systems": sorted(all_systems),
            "suitable_pairs_count": len(suitable_pairs),
            "total_pairs_count": len(lp_data),
            "system_stats": dict(system_pair_counts),
        }

    return results


def create_spa_system_lists(results, output_file="spa_suitable_systems.txt"):
    """
    Create output files with SPA-suitable system lists
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SYSTEMS SUITABLE FOR SPA ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        for lp, data in results.items():
            f.write(f"Language Pair: {lp}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Suitable systems ({len(data['suitable_systems'])}):\n")

            if data["suitable_systems"]:
                for i, system in enumerate(data["suitable_systems"], 1):
                    f.write(f"  {i:2d}. {system}\n")
            else:
                f.write("  None - insufficient common segments\n")

            f.write(f"\nSummary:\n")
            f.write(f"  - Total systems: {len(data['all_systems'])}\n")
            f.write(f"  - SPA-suitable systems: {len(data['suitable_systems'])}\n")
            f.write(f"  - Suitable pairs: {data['suitable_pairs_count']}\n")
            f.write(f"  - Total pairs: {data['total_pairs_count']}\n")
            f.write("\n" + "=" * 50 + "\n\n")

    print(f"System lists written to: {output_file}")


def create_python_dict_format(results, output_file="spa_systems_dict.py"):
    """
    Create a Python file with the systems as a dictionary for easy import
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# SPA-suitable systems by language pair\n")
        f.write("# Generated automatically from common segments analysis\n\n")
        f.write("SPA_SUITABLE_SYSTEMS = {\n")

        for lp, data in results.items():
            f.write(f"    '{lp}': [\n")
            for system in data["suitable_systems"]:
                f.write(f"        '{system}',\n")
            f.write(f"    ],\n")

        f.write("}\n\n")

        # Add summary stats
        f.write("# Summary statistics\n")
        f.write("SPA_STATS = {\n")
        for lp, data in results.items():
            f.write(f"    '{lp}': {{\n")
            f.write(f"        'total_systems': {len(data['all_systems'])},\n")
            f.write(f"        'suitable_systems': {len(data['suitable_systems'])},\n")
            f.write(f"        'suitable_pairs': {data['suitable_pairs_count']},\n")
            f.write(f"        'total_pairs': {data['total_pairs_count']}\n")
            f.write(f"    }},\n")
        f.write("}\n")

    print(f"Python dictionary written to: {output_file}")


def analyze_spa_coverage(results):
    """
    Analyze SPA coverage across language pairs
    """
    print(f"\n{'='*80}")
    print("SPA COVERAGE ANALYSIS")
    print(f"{'='*80}")

    total_lps = len(results)
    viable_lps = sum(
        1 for data in results.values() if len(data["suitable_systems"]) >= 2
    )

    print(f"Language pairs analyzed: {total_lps}")
    print(f"Language pairs viable for SPA: {viable_lps}")
    print(f"SPA viability rate: {viable_lps/total_lps*100:.1f}%")

    if viable_lps > 0:
        print(f"\nViable language pairs:")
        for lp, data in results.items():
            if len(data["suitable_systems"]) >= 2:
                coverage = (
                    data["suitable_pairs_count"] / data["total_pairs_count"] * 100
                )
                print(
                    f"  {lp}: {len(data['suitable_systems'])} systems, {coverage:.1f}% pair coverage"
                )

    if viable_lps < total_lps:
        print(f"\nNon-viable language pairs:")
        for lp, data in results.items():
            if len(data["suitable_systems"]) < 2:
                print(f"  {lp}: Only {len(data['suitable_systems'])} suitable systems")


def main_spa_filter_analysis(csv_summary_file, min_common_segments=2):
    """
    Complete SPA system filtering analysis
    """
    print("Filtering systems suitable for SPA analysis...")

    results = filter_spa_suitable_systems(csv_summary_file, min_common_segments)

    create_spa_system_lists(results, "spa_suitable_systems.txt")
    create_python_dict_format(results, "spa_systems_dict.py")

    # Analyze coverage
    analyze_spa_coverage(results)

    # Try stricter criteria for comparison
    if min_common_segments == 2:
        print(f"\n{'='*80}")
        print("COMPARISON: STRICTER CRITERIA (≥10 common segments)")
        print(f"{'='*80}")

        strict_results = filter_spa_suitable_systems(
            csv_summary_file, min_common_segments=10
        )

        print(f"\nComparison summary:")
        for lp in results.keys():
            loose_count = len(results[lp]["suitable_systems"])
            strict_count = (
                len(strict_results[lp]["suitable_systems"])
                if lp in strict_results
                else 0
            )
            print(
                f"  {lp}: {loose_count} systems (≥2 segs) vs {strict_count} systems (≥10 segs)"
            )

    return results


# Usage example
if __name__ == "__main__":
    # Replace with your CSV summary file path
    csv_file = "common_segments_analysis_summary.csv"

    try:
        results = main_spa_filter_analysis(csv_file)

        print(f"\nFiles created:")
        print(f"  - spa_suitable_systems.txt (readable format)")
        print(f"  - spa_systems_dict.py (for importing into Python)")

    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Make sure you've run the common segments analysis first.")
    except Exception as e:
        print(f"Error: {e}")
