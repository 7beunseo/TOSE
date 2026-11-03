from __future__ import annotations

from typing import Any


FEATURE_NAMES: dict[str, list[str]] = {
    "adult": [
        "age", "workclass", "final sampling weight", "education", "education years",
        "marital status", "occupation", "relationship", "race", "sex",
        "capital gain", "capital loss", "hours per week", "native country",
    ],
    "bean": [
        "area", "perimeter", "major axis length", "minor axis length", "aspect ratio",
        "eccentricity", "convex area", "equivalent diameter", "extent", "solidity",
        "roundness", "compactness", "shape factor 1", "shape factor 2",
        "shape factor 3", "shape factor 4",
    ],
    "default": [
        "credit limit", "sex", "education", "marital status", "age",
        "repayment status month 1", "repayment status month 2", "repayment status month 3",
        "repayment status month 4", "repayment status month 5", "repayment status month 6",
        "bill amount month 1", "bill amount month 2", "bill amount month 3",
        "bill amount month 4", "bill amount month 5", "bill amount month 6",
        "payment amount month 1", "payment amount month 2", "payment amount month 3",
        "payment amount month 4", "payment amount month 5", "payment amount month 6",
    ],
    "gesture": [f"gesture sensor feature {idx + 1}" for idx in range(32)],
    "letter": [
        "horizontal box position", "vertical box position", "box width", "box height",
        "foreground pixel count", "mean horizontal pixel position", "mean vertical pixel position",
        "horizontal position variance", "vertical position variance", "horizontal vertical correlation",
        "horizontal squared vertical correlation", "horizontal vertical squared correlation",
        "mean horizontal edge count", "horizontal edge vertical correlation",
        "mean vertical edge count", "vertical edge horizontal correlation",
    ],
    "magic": [
        "shower major axis length", "shower minor axis width", "shower size", "shower concentration",
        "shower concentration in core", "shower asymmetry", "third moment along major axis",
        "third moment along minor axis", "shower orientation angle", "source distance",
    ],
    "news": [f"online news feature {idx + 1}" for idx in range(48)],
    "shoppers": [
        "administrative page count", "administrative page duration", "informational page count",
        "informational page duration", "product page count", "product page duration", "bounce rate",
        "exit rate", "page value", "special day proximity", "month", "operating system",
        "browser", "region", "traffic type", "visitor type", "weekend",
    ],
    "abalone": [
        "sex", "shell length", "shell diameter", "shell height", "whole weight",
        "shucked weight", "viscera weight", "shell weight",
    ],
    "bank": [
        "age", "job", "marital status", "education", "credit default", "account balance",
        "housing loan", "personal loan", "contact type", "contact day", "contact month",
        "contact duration", "campaign contacts", "days since prior contact",
        "number of prior contacts", "previous campaign outcome",
    ],
    "breast": [
        f"{statistic} {measurement}"
        for statistic in ("mean", "standard error", "worst")
        for measurement in (
            "radius", "texture", "perimeter", "area", "smoothness", "compactness",
            "concavity", "concave points", "symmetry", "fractal dimension",
        )
    ],
    "california": [
        "median income", "median house age", "average rooms", "average bedrooms",
        "population", "average household occupancy", "latitude", "longitude",
    ],
    "german": [
        "checking account status", "duration in months", "credit history", "credit purpose",
        "credit amount", "savings account status", "employment duration", "installment rate",
        "personal status and sex", "other debtors", "residence duration", "property",
        "age", "other installment plans", "housing", "existing credits", "job",
        "number of dependents", "telephone", "foreign worker",
    ],
    "mushroom": [
        "cap shape", "cap surface", "cap color", "bruises", "odor", "gill attachment",
        "gill spacing", "gill size", "gill color", "stalk shape", "stalk root",
        "stalk surface above ring", "stalk surface below ring", "stalk color above ring",
        "stalk color below ring", "veil type", "veil color", "ring number", "ring type",
        "spore print color", "population", "habitat",
    ],
}


def ordered_column_contexts(dataset: str, info: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return contexts in the benchmark's numeric-first, categorical-second logical order."""
    original_indices = [int(idx) for idx in info["num_col_idx"]] + [int(idx) for idx in info["cat_col_idx"]]
    known_names = FEATURE_NAMES.get(dataset, [])
    num_indices = {int(idx) for idx in info["num_col_idx"]}
    names: list[str] = []
    descriptions: list[str] = []
    for original_idx in original_indices:
        name = known_names[original_idx] if original_idx < len(known_names) else f"feature {original_idx}"
        feature_type = "numerical" if original_idx in num_indices else "categorical"
        names.append(f"feature_{original_idx}_{name.replace(' ', '_')}")
        descriptions.append(
            f"{name} is a {feature_type} attribute in the {dataset} tabular dataset."
        )
    return names, descriptions
