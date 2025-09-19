# modified from the data.py in https://github.com/google-research/mt-metrics-eval
import pandas as pd
import numpy as np
from typing import Any, Callable, Iterable, Sequence
import copy
from mt_metrics_eval import stats


class EvalSet:
    """Data for a test set and one language pair - modified to read from CSV."""

    def __init__(
        self,
        csv_path: str,
        name: str = "mqm",
        lp: str = None,
    ):
        """Load data from CSV file.

        Args:
          csv_path: Path to the CSV file containing the data
          name: Name of test set (for compatibility)
          lp: Language pair, will be inferred from CSV if None
          Other args: Kept for compatibility but not used
        """
        self.name = name

        # Read CSV data
        self.df = pd.read_csv(csv_path)

        # Infer language pair if not provided
        if lp is None:
            self.lp = self.df["lp"].iloc[0]
        else:
            self.lp = lp

        # Filter data for the specified language pair
        self.df = self.df[self.df["lp"] == self.lp].reset_index(drop=True)

        # Create mock info object for compatibility
        class MockInfo:
            def __init__(self):
                self.std_ref = "ref"
                self.std_gold = {"seg": "score"}  # Human scores are in 'score' column
                self.primary_metrics = set()
                self.outlier_systems = set()

        self.info = MockInfo()
        self._std_human_scores = self.info.std_gold
        self._primary_metrics = self.info.primary_metrics.copy()

        self._ReadDatasetFromCSV()

    @property
    def src_lang(self) -> str:
        return self.lp.split("-")[0]

    @property
    def tgt_lang(self) -> str:
        return self.lp.split("-")[1]

    @property
    def levels(self) -> set[str]:
        """Levels for which scores exist, subset of {sys, domain, doc, seg}."""
        return {"seg", "sys"}

    @property
    def domain_names(self) -> Sequence[str]:
        """Names of domains, in canonical order."""
        return sorted(self.df["domain"].unique())

    @property
    def ref_names(self) -> set[str]:
        """Names of available references."""
        return {"ref"}

    @property
    def std_ref(self) -> str:
        """Name of standard reference."""
        return "ref"

    @property
    def sys_names(self) -> set[str]:
        """Names of all 'systems' for which output is available."""
        return set(self.df["system"].unique())

    @property
    def human_sys_names(self) -> set[str]:
        """Names of systems in sys_names that are human output."""
        # Check if any systems have names that suggest they're references
        human_systems = set()
        for sys_name in self.sys_names:
            if sys_name.startswith("ref") or "human" in sys_name.lower():
                human_systems.add(sys_name)
        return human_systems

    @property
    def outlier_sys_names(self) -> set[str]:
        """Names of systems in sys_names considered to be outliers."""
        return self.info.outlier_systems

    @property
    def human_score_names(self) -> set[str]:
        """Names of available human scores."""
        return {"score"}

    def SetOutlierSysNames(self, outliers: set[str]) -> None:
        """Overwrites the list of outlier systems."""
        self.info.outlier_systems = outliers

    def StdHumanScoreName(self, level) -> str:
        """Name of standard human score for a given level in self.levels."""
        if level == "seg":
            return "score"
        return None

    @property
    def metric_names(self) -> set[str]:
        """Full names of available metrics."""
        # Get all metric columns (excluding non-metric columns)
        non_metric_cols = {
            "lp",
            "src",
            "mt",
            "ref",
            "mqm_score",
            "system",
            "annotators",
            "domain",
            "year",
            "score",
            "id",
        }
        metric_cols = set(self.df.columns) - non_metric_cols
        return metric_cols

    @property
    def metric_basenames(self) -> set[str]:
        """Basenames of available metrics."""
        return self.metric_names

    @property
    def primary_metrics(self) -> set[str]:
        """Base names of primary metrics, empty set if none."""
        return self._primary_metrics

    def BaseMetric(self, metric_name: str) -> str:
        """Base name for a given metric."""
        return metric_name

    def DisplayName(self, metric_name: str, fmt="spreadsheet") -> str:
        return metric_name

    def ReferencesUsed(self, metric_name: str) -> set[str]:
        """Reference(s) used by a metric."""
        return {"ref"}

    @property
    def domains(self) -> dict[str, list[list[int]]]:
        """Map from domain name to [[beg, end+1], ...] segment position lists."""
        return self._domains

    @property
    def docs(self) -> dict[str, list[int]]:
        """Map from doc name to [beg, end+1] segment positions."""
        return self._docs

    @property
    def src(self) -> list[str]:
        """Segments in the source text, in order."""
        return self.df["src"].tolist()

    @property
    def all_refs(self) -> dict[str, list[str]]:
        """Map from reference name to text for that reference."""
        return {"ref": self.df["ref"].tolist()}

    @property
    def sys_outputs(self) -> dict[str, list[str]]:
        """Map from system name to output text from that system."""
        return self._sys_outputs

    def Scores(self, level: str, scorer: str) -> dict[str, list[float]]:
        """Get stored scores assigned to text units at a given level.

        Args:
          level: Text units to which scores apply, one of 'sys', 'domain', 'doc', or 'seg'.
          scorer: Method used to produce scores, may be any string in
            human_score_names or metric_names.

        Returns:
          Mapping from system names to lists of float scores.
        """
        if level == "seg":
            if scorer == "score":
                # Return human scores grouped by system, ensuring consistent ordering
                result = {}
                # Sort by id to ensure consistent ordering across systems
                sorted_df = self.df.sort_values("id")
                for system in self.sys_names:
                    system_data = sorted_df[sorted_df["system"] == system]
                    result[system] = system_data["score"].tolist()
                return result
            elif scorer in self.metric_names:
                # Return metric scores grouped by system, ensuring consistent ordering
                result = {}
                sorted_df = self.df.sort_values("id")
                for system in self.sys_names:
                    system_data = sorted_df[sorted_df["system"] == system]
                    result[system] = system_data[scorer].tolist()
                return result
        elif level == "sys":
            if scorer == "score":
                # Return system-level human scores (averaged)
                result = {}
                for system in self.sys_names:
                    system_data = self.df[self.df["system"] == system]
                    result[system] = [system_data["score"].mean()]
                return result
            elif scorer in self.metric_names:
                # Return system-level metric scores (averaged)
                result = {}
                for system in self.sys_names:
                    system_data = self.df[self.df["system"] == system]
                    result[system] = [system_data[scorer].mean()]
                return result

        return None

    def Correlation(
        self,
        gold_scores: dict[str, list[float]],
        metric_scores: dict[str, list[float]],
        sys_names: Iterable[str] = None,
        indexes: Sequence[int] = None,
    ):
        """Get correlation statistics for given metric scores."""
        if sys_names is None:
            sys_names = set(gold_scores.keys())
        sys_names = set(sys_names)
        if not sys_names.issubset(metric_scores):
            raise ValueError(f"Missing metric scores: {sys_names - set(metric_scores)}")
        if not sys_names.issubset(gold_scores):
            raise ValueError(f"Missing gold scores: {sys_names - set(gold_scores)}")

        all_gold_scores, all_metric_scores = [], []
        for sys_name in sys_names:
            gscores, mscores = gold_scores[sys_name], metric_scores[sys_name]
            if indexes is not None:
                gscores = np.asarray(gscores)[indexes]
                mscores = np.asarray(mscores)[indexes]
            if len(gscores) != len(mscores):
                raise ValueError(
                    "Wrong number of scores for system %s: %d vs %d"
                    % (sys_name, len(gscores), len(mscores))
                )
            all_gold_scores.extend(gscores)
            all_metric_scores.extend(mscores)

        # Return the standard stats.Correlation object
        return stats.Correlation(len(sys_names), all_gold_scores, all_metric_scores)

    def _ReadDatasetFromCSV(self):
        """Read and organize data from the CSV."""
        # Create domain mappings
        domain_positions = {}
        current_pos = 0
        for domain in sorted(self.df["domain"].unique()):
            domain_data = self.df[self.df["domain"] == domain]
            domain_positions[domain] = [[current_pos, current_pos + len(domain_data)]]
            current_pos += len(domain_data)
        self._domains = domain_positions

        # Create document mappings (using domain for simplicity)
        self._docs = {domain: pos[0] for domain, pos in domain_positions.items()}

        # Create system outputs mapping with consistent ordering
        self._sys_outputs = {}
        sorted_df = self.df.sort_values("id")
        for system in self.sys_names:
            system_data = sorted_df[sorted_df["system"] == system]
            self._sys_outputs[system] = system_data["mt"].tolist()

    def ParseMetricName(self, metric_name: str) -> tuple[str, set[str]]:
        """Parse metric name to get basename and references used."""
        return metric_name, {"ref"}

    def SetPrimaryMetrics(self, metrics: set[str]):
        """Set primary metrics to the given set of basenames."""
        if metrics:
            self._primary_metrics = metrics.copy()
        else:
            self._primary_metrics = set()
