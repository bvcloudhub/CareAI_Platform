"""Shared backend patient-access scope for analytics surfaces.

CareAI currently has organisation-level clinical roles plus optional patient
assignment fields. Analytics keeps the existing shared-care behaviour for
unassigned patients while preventing a nurse/GP from seeing patients explicitly
assigned to a different professional.
"""
from __future__ import annotations

from services.db import query_db

CLINICAL_ROLES = {"admin", "nurse", "gp"}


def patient_scope_clause(*, user_id: int, role: str, alias: str = "p") -> tuple[str, tuple]:
    """Return an SQL predicate and parameters for rows owned by ``alias``.

    Admins can see the organisation population. Nurses and GPs can see the
    unassigned shared-care pool plus patients assigned to them. This mirrors the
    current CareAI assignment model without creating a second permission table.
    """
    if role == "admin":
        return "1=1", ()
    if role == "nurse":
        return f"({alias}.assigned_nurse_id IS NULL OR {alias}.assigned_nurse_id=?)", (int(user_id),)
    if role == "gp":
        return f"({alias}.assigned_clinician_id IS NULL OR {alias}.assigned_clinician_id=?)", (int(user_id),)
    return "0=1", ()


def accessible_patient_ids(*, user_id: int, role: str, include_archived: bool = False) -> list[int]:
    clause, params = patient_scope_clause(user_id=user_id, role=role, alias="p")
    active_sql = "" if include_archived else "AND p.active=1"
    rows = query_db(f"SELECT p.id FROM patients p WHERE {clause} {active_sql} ORDER BY p.id", params)
    return [int(r["id"]) for r in rows]


def patient_is_accessible(patient_id: int, *, user_id: int, role: str, include_archived: bool = False) -> bool:
    clause, params = patient_scope_clause(user_id=user_id, role=role, alias="p")
    active_sql = "" if include_archived else "AND p.active=1"
    row = query_db(
        f"SELECT p.id FROM patients p WHERE p.id=? AND {clause} {active_sql}",
        (int(patient_id), *params),
        one=True,
    )
    return bool(row)
