import pandas as pd
from collections import defaultdict


def find_best_system_combination_optimized(csv_file, n=10):
    """
    Optimized version: Find combination of n systems that maximizes intersection
    Uses greedy approach with smart pruning for much better performance

    Args:
        csv_file (str): Path to language pair CSV file
        n (int): Number of systems to select

    Returns:
        tuple: (language_pair, best_systems, intersection_count, intersection_sources)
    """
    df = pd.read_csv(csv_file)
    lp = df["lp"].iloc[0]

    system_sources = {}
    for system in df["system"].unique():
        system_sources[system] = set(df[df["system"] == system]["src"].tolist())

    systems = list(system_sources.keys())
    print(f"Total systems: {len(systems)}")

    if len(systems) < n:
        print(f"WARNING: Only {len(systems)} systems available, using all")
        n = len(systems)

    # find sources that appear in at least n systems
    src_system_count = defaultdict(set)
    for system, sources in system_sources.items():
        for src in sources:
            src_system_count[src].add(system)

    # only consider sources that appear in at least n systems
    viable_sources = {
        src: systems_set
        for src, systems_set in src_system_count.items()
        if len(systems_set) >= n
    }

    print(f"Sources appearing in at least {n} systems: {len(viable_sources)}")

    if not viable_sources:
        print("No sources appear in enough systems for intersection")
        return lp, systems[:n], 0, set()

    # start with system that has most viable sources
    system_viable_count = {}
    for system in systems:
        count = sum(1 for src in system_sources[system] if src in viable_sources)
        system_viable_count[system] = count

    selected_systems = []
    current_intersection = set(viable_sources.keys())

    # greedy selection
    remaining_systems = set(systems)

    for step in range(n):
        best_system = None
        best_intersection_size = -1
        best_intersection = None

        for system in remaining_systems:
            system_sources_set = system_sources[system]
            new_intersection = current_intersection.intersection(system_sources_set)

            if len(new_intersection) > best_intersection_size:
                best_intersection_size = len(new_intersection)
                best_system = system
                best_intersection = new_intersection

        if best_system is None:
            break

        selected_systems.append(best_system)
        remaining_systems.remove(best_system)
        current_intersection = best_intersection

        print(
            f"Step {step+1}: Added {best_system}, intersection: {len(current_intersection)}"
        )

    print("Optimizing selection...")
    improved = True
    iterations = 0
    max_iterations = 20

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i in range(len(selected_systems)):
            current_system = selected_systems[i]
            current_intersection_size = len(current_intersection)

            for candidate in remaining_systems:
                test_systems = selected_systems.copy()
                test_systems[i] = candidate

                test_intersection = set(viable_sources.keys())
                for sys in test_systems:
                    test_intersection = test_intersection.intersection(
                        system_sources[sys]
                    )

                if len(test_intersection) > current_intersection_size:
                    selected_systems[i] = candidate
                    remaining_systems.remove(candidate)
                    remaining_systems.add(current_system)
                    current_intersection = test_intersection
                    improved = True
                    print(
                        f"Swap improved: {current_system} -> {candidate}, intersection: {len(test_intersection)}"
                    )
                    break

            if improved:
                break

    final_intersection_count = len(current_intersection)

    print(f"{'='*60}")
    print(f"OPTIMIZED COMBINATION:")
    print(f"Systems: {selected_systems}")
    print(f"Common sources in ALL {n} systems: {final_intersection_count}")
    print(f"{'='*60}")

    if final_intersection_count > 0:
        print(f"Example common sources (first 3):")
        for i, src in enumerate(list(current_intersection)[:3]):
            print(f"  {i+1}. {src[:100]}...")

    return lp, selected_systems, final_intersection_count, current_intersection


def main(n=10):
    language_pairs = ["en-de", "en-ru", "zh-en"]
    all_results = {}

    print(f"OPTIMIZED SEARCH FOR BEST {n} SYSTEM COMBINATIONS")
    print("=" * 80)

    for lp in language_pairs:
        csv_file = f"benchmark_predictions/mqm_{lp}.csv"
        try:
            lp_name, best_systems, intersection_count, sources = (
                find_best_system_combination_optimized(csv_file, n)
            )
            all_results[lp_name] = {
                "systems": best_systems,
                "intersection_count": intersection_count,
                "sources": sources,
            }
        except FileNotFoundError:
            print(f"\nWARNING: {csv_file} not found. Skipping {lp}")
        except Exception as e:
            print(f"\nERROR analyzing {csv_file}: {e}")

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")

    for lp, result in all_results.items():
        print(f"\n{lp}:")
        print(f"  Selected systems: {result['systems']}")
        print(f"  Sources in ALL systems: {result['intersection_count']}")

    if all_results:
        best_lp = max(
            all_results.keys(), key=lambda x: all_results[x]["intersection_count"]
        )
        best_count = all_results[best_lp]["intersection_count"]
        print(f"\nBEST OVERALL: {best_lp} with {best_count} common sources")


if __name__ == "__main__":
    n = 3
    main(n)
