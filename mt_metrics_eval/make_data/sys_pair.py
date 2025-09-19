import pandas as pd
from itertools import combinations
from collections import defaultdict
from datetime import datetime


def write_to_file_and_console(text, output_file=None):
    """Helper function to write to both console and file"""
    print(text)
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def analyze_common_segments_per_system_pair(csv_file, output_file=None):
    """
    For each language pair, analyze common segments between every system pair

    Args:
        csv_file: Path to your CSV file
        output_file: Path to output text file (optional)

    Returns:
        Dictionary with analysis results
    """

    # Initialize output file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("COMMON SEGMENTS ANALYSIS BETWEEN SYSTEM PAIRS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {csv_file}\n")
            f.write("=" * 100 + "\n")

    # Print header to console
    print("=" * 100)
    print("COMMON SEGMENTS ANALYSIS BETWEEN SYSTEM PAIRS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input file: {csv_file}")
    print("=" * 100)

    # Read the CSV
    df = pd.read_csv(csv_file)
    results = {}

    # Group by language pair
    for lp in sorted(df["lp"].unique()):
        write_to_file_and_console(f"\n{'='*80}", output_file)
        write_to_file_and_console(f"LANGUAGE PAIR: {lp}", output_file)
        write_to_file_and_console(f"{'='*80}", output_file)

        lp_data = df[df["lp"] == lp]
        systems = sorted(lp_data["system"].unique())

        write_to_file_and_console(f"Total systems: {len(systems)}", output_file)
        write_to_file_and_console(f"Systems: {systems}", output_file)

        # Get segments for each system
        system_segments = {}
        for system in systems:
            system_data = lp_data[lp_data["system"] == system]
            segments = set(system_data["src"].tolist())
            system_segments[system] = segments
            write_to_file_and_console(
                f"{system}: {len(segments)} segments", output_file
            )

        write_to_file_and_console(f"\nPAIRWISE COMMON SEGMENTS:", output_file)
        write_to_file_and_console(f"{'-'*60}", output_file)

        # Calculate common segments for each system pair
        pairwise_results = []

        for sys1, sys2 in combinations(systems, 2):
            common_segs = system_segments[sys1] & system_segments[sys2]
            total_sys1 = len(system_segments[sys1])
            total_sys2 = len(system_segments[sys2])
            common_count = len(common_segs)

            pairwise_results.append(
                {
                    "system1": sys1,
                    "system2": sys2,
                    "common_segments": common_count,
                    "sys1_total": total_sys1,
                    "sys2_total": total_sys2,
                    "overlap_pct_sys1": (
                        (common_count / total_sys1) * 100 if total_sys1 > 0 else 0
                    ),
                    "overlap_pct_sys2": (
                        (common_count / total_sys2) * 100 if total_sys2 > 0 else 0
                    ),
                    "segments": common_segs,
                }
            )

            write_to_file_and_console(f"{sys1} vs {sys2}:", output_file)
            write_to_file_and_console(f"  Common segments: {common_count}", output_file)
            write_to_file_and_console(
                f"  {sys1} total: {total_sys1} ({common_count/total_sys1*100:.1f}% overlap)",
                output_file,
            )
            write_to_file_and_console(
                f"  {sys2} total: {total_sys2} ({common_count/total_sys2*100:.1f}% overlap)",
                output_file,
            )
            write_to_file_and_console("", output_file)

        # Summary statistics for this language pair
        common_counts = [pair["common_segments"] for pair in pairwise_results]
        if common_counts:
            write_to_file_and_console(f"SUMMARY FOR {lp}:", output_file)
            write_to_file_and_console(
                f"  Min common segments: {min(common_counts)}", output_file
            )
            write_to_file_and_console(
                f"  Max common segments: {max(common_counts)}", output_file
            )
            write_to_file_and_console(
                f"  Average common segments: {sum(common_counts)/len(common_counts):.1f}",
                output_file,
            )
            write_to_file_and_console(
                f"  Total system pairs: {len(pairwise_results)}", output_file
            )

            # Check SPA feasibility
            spa_feasible_pairs = [
                pair for pair in pairwise_results if pair["common_segments"] >= 2
            ]
            write_to_file_and_console(
                f"  Pairs suitable for SPA (≥2 common segments): {len(spa_feasible_pairs)}/{len(pairwise_results)}",
                output_file,
            )

        results[lp] = {
            "systems": systems,
            "system_segments": system_segments,
            "pairwise_results": pairwise_results,
            "total_pairs": len(pairwise_results),
        }

    return results


def check_spa_feasibility(results, output_file=None):
    """
    Check which language pairs and system pairs are suitable for SPA calculation
    """
    write_to_file_and_console(f"\n{'='*80}", output_file)
    write_to_file_and_console("SPA FEASIBILITY ANALYSIS", output_file)
    write_to_file_and_console(f"{'='*80}", output_file)

    for lp, lp_data in results.items():
        write_to_file_and_console(f"\nLanguage Pair: {lp}", output_file)

        pairs = lp_data["pairwise_results"]
        total_pairs = len(pairs)

        # Count pairs by common segment ranges
        ranges = [
            (0, 0),
            (1, 1),
            (2, 4),
            (5, 9),
            (10, 19),
            (20, 49),
            (50, float("inf")),
        ]
        range_counts = {
            f"{low}-{high if high != float('inf') else '50+'}": 0
            for low, high in ranges
        }

        for pair in pairs:
            common = pair["common_segments"]
            for low, high in ranges:
                if low <= common <= high:
                    range_key = f"{low}-{high if high != float('inf') else '50+'}"
                    range_counts[range_key] += 1
                    break

        write_to_file_and_console("  Common segments distribution:", output_file)
        for range_key, count in range_counts.items():
            if count > 0:
                pct = (count / total_pairs) * 100
                write_to_file_and_console(
                    f"    {range_key:8s}: {count:3d} pairs ({pct:5.1f}%)", output_file
                )

        # SPA recommendations
        spa_suitable = sum(1 for pair in pairs if pair["common_segments"] >= 2)
        strong_spa = sum(1 for pair in pairs if pair["common_segments"] >= 10)

        write_to_file_and_console(f"  SPA Analysis:", output_file)
        write_to_file_and_console(f"    Total pairs: {total_pairs}", output_file)
        write_to_file_and_console(
            f"    Suitable for SPA (≥2 segments): {spa_suitable} ({spa_suitable/total_pairs*100:.1f}%)",
            output_file,
        )
        write_to_file_and_console(
            f"    Strong SPA candidates (≥10 segments): {strong_spa} ({strong_spa/total_pairs*100:.1f}%)",
            output_file,
        )

        if spa_suitable < total_pairs:
            write_to_file_and_console(
                f"    ⚠️  {total_pairs - spa_suitable} pairs have <2 common segments (SPA not reliable)",
                output_file,
            )
        if strong_spa < total_pairs * 0.5:
            write_to_file_and_console(
                f"    ⚠️  <50% of pairs have ≥10 common segments (limited statistical power)",
                output_file,
            )


def print_detailed_segment_overlap(
    results, language_pair=None, min_common=None, output_file=None
):
    """
    Print detailed segment overlap information
    """
    lps_to_show = [language_pair] if language_pair else results.keys()

    for lp in lps_to_show:
        if lp not in results:
            write_to_file_and_console(f"Language pair {lp} not found", output_file)
            continue

        lp_results = results[lp]

        write_to_file_and_console(f"\n{'='*80}", output_file)
        write_to_file_and_console(f"DETAILED OVERLAP FOR {lp}", output_file)
        write_to_file_and_console(f"{'='*80}", output_file)

        # Sort by number of common segments (descending)
        pairs = sorted(
            lp_results["pairwise_results"],
            key=lambda x: x["common_segments"],
            reverse=True,
        )

        for pair in pairs:
            if min_common and pair["common_segments"] < min_common:
                continue

            write_to_file_and_console(
                f"\n{pair['system1']} vs {pair['system2']}:", output_file
            )
            write_to_file_and_console(
                f"  Common segments: {pair['common_segments']}", output_file
            )
            write_to_file_and_console(
                f"  Coverage: {pair['overlap_pct_sys1']:.1f}% of {pair['system1']}, "
                f"{pair['overlap_pct_sys2']:.1f}% of {pair['system2']}",
                output_file,
            )

            # Show first few common segments as examples
            if pair["segments"]:
                sample_segments = list(pair["segments"])[:3]
                write_to_file_and_console(f"  Example segments:", output_file)
                for i, seg in enumerate(sample_segments, 1):
                    write_to_file_and_console(f"    {i}. {seg[:100]}...", output_file)


def create_summary_csv(results, csv_output_file):
    """
    Create a CSV summary of all pairwise common segments
    """
    summary_data = []

    for lp, lp_data in results.items():
        for pair in lp_data["pairwise_results"]:
            summary_data.append(
                {
                    "language_pair": lp,
                    "system1": pair["system1"],
                    "system2": pair["system2"],
                    "common_segments": pair["common_segments"],
                    "sys1_total_segments": pair["sys1_total"],
                    "sys2_total_segments": pair["sys2_total"],
                    "overlap_pct_sys1": round(pair["overlap_pct_sys1"], 1),
                    "overlap_pct_sys2": round(pair["overlap_pct_sys2"], 1),
                    "spa_suitable": pair["common_segments"] >= 2,
                    "strong_spa_candidate": pair["common_segments"] >= 10,
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(
        ["language_pair", "common_segments"], ascending=[True, False]
    )

    summary_df.to_csv(csv_output_file, index=False)
    print(f"Summary CSV created: {csv_output_file}")


def main_analysis(csv_file, output_file="common_segments_analysis.txt"):
    """
    Complete analysis pipeline with file output
    """
    print(f"Loading data and analyzing common segments between system pairs...")
    print(f"Writing results to: {output_file}")

    # Main analysis - writes to file
    results = analyze_common_segments_per_system_pair(csv_file, output_file)

    # Check SPA feasibility - appends to file
    check_spa_feasibility(results, output_file)

    # Show some detailed examples - appends to file
    write_to_file_and_console(f"\n{'='*80}", output_file)
    write_to_file_and_console("EXAMPLES OF HIGH-OVERLAP PAIRS", output_file)
    write_to_file_and_console(f"{'='*80}", output_file)

    # Add detailed overlap for each language pair
    for lp in results.keys():
        print_detailed_segment_overlap(
            results, language_pair=lp, min_common=1, output_file=output_file
        )

    # Also create a summary CSV file
    summary_csv_file = output_file.replace(".txt", "_summary.csv")
    create_summary_csv(results, summary_csv_file)

    print(f"\nAnalysis complete! Check these files:")
    print(f"  - Full report: {output_file}")
    print(f"  - Summary CSV: {summary_csv_file}")

    return results


# Usage example
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "benchmark_predictions/mqm_7.csv"
    output_file = "common_segments_analysis.txt"

    try:
        # Run complete analysis with file output
        results = main_analysis(csv_file, output_file)

        # Optional: Quick console summary
        print(f"\nQuick summary:")
        for lp, data in results.items():
            total_pairs = data["total_pairs"]
            suitable_pairs = sum(
                1 for pair in data["pairwise_results"] if pair["common_segments"] >= 2
            )
            print(f"{lp}: {suitable_pairs}/{total_pairs} pairs suitable for SPA")

    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
        print("Please update the csv_file variable with the correct path to your data.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your CSV format and ensure it has the expected columns.")
