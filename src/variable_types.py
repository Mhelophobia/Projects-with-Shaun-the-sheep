"""Explicit variable roles and reference-category encoding, independent of storage type."""
import re

import numpy as np
import pandas as pd

ROLES = ["Needs review", "Numeric", "Binary / existing dummy", "Categorical", "Identifier"]


def suggest(series, name):
    values = series.dropna()
    if re.search(r"(^|_)(id|identifier|serial|code|zip|postal)($|_)", str(name).lower()):
        return "Identifier"
    numbers = pd.to_numeric(values, errors="coerce")
    if len(values) and numbers.notna().all() and np.isfinite(numbers).all():
        if set(numbers.unique()) <= {0, 1}:
            return "Binary / existing dummy"
        if numbers.nunique() <= 10 and (numbers % 1 == 0).all():
            return "Needs review"
        return "Numeric"
    if numbers.notna().any():
        return "Needs review"
    return "Categorical"


def levels(series):
    return sorted(series.dropna().astype(str).unique().tolist())


def prepare(df, roles, references):
    """Keep source categories for exploration; add k-1 numeric dummy columns.

    Missing categories stay missing in every dummy, never masquerade as baseline.
    """
    if set(roles) != set(df.columns):
        raise ValueError("Review every column before applying variable types.")
    out = df.copy()
    log = []
    for col, role in roles.items():
        if role not in ROLES or role == "Needs review":
            raise ValueError(f"Choose a type for '{col}': quantity, category, binary, or identifier.")
        if role in ("Numeric", "Binary / existing dummy"):
            converted = pd.to_numeric(df[col], errors="coerce")
            if (converted.isna() & df[col].notna()).any() or not np.isfinite(converted.dropna()).all():
                raise ValueError(f"'{col}' contains values that cannot be used as numbers.")
            if role == "Binary / existing dummy" and not set(converted.dropna().unique()) <= {0, 1}:
                raise ValueError(f"'{col}' must contain only 0, 1, or missing values.")
            out[col] = converted
            log.append(f"{col}: {role}; no dummy encoding applied.")
        elif role == "Identifier":
            out[col] = df[col].astype("string")
            log.append(f"{col}: identifier; excluded from numeric regression choices.")
        else:
            cats = levels(df[col])
            if not 2 <= len(cats) <= 50:
                raise ValueError(f"'{col}' needs 2–50 observed categories for encoding. "
                                 "Use Identifier for IDs, or review/group its values first.")
            reference = references.get(col)
            if reference not in cats:
                raise ValueError(f"Select a reference category for '{col}'.")
            out[col] = df[col].astype("string")
            created = []
            for category in cats:
                if category == reference:
                    continue
                base = f"{col}__{category}"
                name = base
                suffix = 2
                while name in out.columns:
                    name = f"{base}_{suffix}"
                    suffix += 1
                out[name] = out[col].eq(category).astype(float)
                created.append(name)
            log.append(f"{col}: categorical; reference = {reference}; created {', '.join(created)}. "
                       "Missing values remain missing.")
    return out, log
