from pathlib import Path
from datetime import datetime
import re
import shutil
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "services" / "dashboard_metrics_service.py"


def fail(message):
    print("\nERROR:")
    print(message)
    print("\nNo CareAI source file was changed.")
    sys.exit(1)


if not TARGET.exists():
    fail(f"File not found: {TARGET}")


original = TARGET.read_text(encoding="utf-8")
updated = original


# ============================================================
# 1. Import the shared current-priority resolver
# ============================================================

import_line = (
    "from services.clinical_evidence_service "
    "import priority_snapshot\n"
)

if import_line not in updated:
    marker = "from services.db import query_db\n"

    if marker not in updated:
        fail("Could not find services.db import.")

    updated = updated.replace(
        marker,
        marker + import_line,
        1,
    )


# ============================================================
# 2. Remove the OLD historical Agentic-priority query
#
# It currently treats the latest non-rejected run as current,
# even when that workflow is already COMPLETED.
# ============================================================

old_agentic_pattern = re.compile(
    r'''
    \n    \#\ Agentic\ priority\ is\ intentionally\ queried\ separately\..*?
    agentic_by_patient\ =\ \{int\(r\["patient_id"\]\):\ dict\(r\)\ for\ r\ in\ agentic_rows\}
    \n
    ''',
    re.DOTALL | re.VERBOSE,
)

updated, removed_count = old_agentic_pattern.subn(
    """
    # Current patient priority is resolved below with priority_snapshot().
    # Historical completed/rejected Agentic workflows remain in history
    # but must not override the current patient state.

""",
    updated,
    count=1,
)


# ============================================================
# 3. Replace the OLD Patients-page priority calculation
# ============================================================

old_block = '''        item = dict(row)
        agentic = agentic_by_patient.get(int(item["id"]))
        item["agentic_severity"] = agentic.get("severity") if agentic else None
        item["priority"] = _highest_priority(
            item.get("current_status"), item.get("risk_level"), item.get("agentic_severity")
        )
        item["priority_label"] = item["priority"].title()
        item["risk_score"] = int(item.get("risk_score") or 0)
'''

new_block = '''        item = dict(row)

        # Keep the raw Care Intelligence result available for
        # backwards compatibility and clinical explanation.
        item["care_risk_level"] = _normalise_priority(
            item.get("risk_level")
        )
        item["care_risk_score"] = int(
            item.get("risk_score") or 0
        )

        # IMPORTANT:
        # Use exactly the same CURRENT priority resolver as
        # Patient 360, Command Centre, Population and Reports.
        #
        # priority_snapshot() combines:
        #   1. current Care Intelligence overall risk
        #   2. an ACTIVE Agentic/Hospital workflow, if one exists
        #
        # Completed historical workflows do NOT override
        # current patient risk.
        snapshot = priority_snapshot(
            int(item["id"])
        )

        item["priority"] = _normalise_priority(
            snapshot.get("level")
        )

        item["priority_label"] = (
            item["priority"].title()
        )

        # Keep score and level from the SAME source.
        # This prevents invalid combinations such as:
        #     CRITICAL · 30
        #     CRITICAL · 12
        item["risk_score"] = int(
            snapshot.get("score") or 0
        )

        item["priority_source"] = (
            snapshot.get("source")
            or "Care intelligence"
        )

        item["priority_reason"] = (
            snapshot.get("reason")
            or ""
        )

        item["priority_run_id"] = (
            snapshot.get("run_id")
        )

        # Backwards-compatible field for any older template code.
        # It is populated only when an ACTIVE workflow currently
        # drives the patient's priority.
        item["agentic_severity"] = (
            item["priority"]
            if snapshot.get("run_id")
            else None
        )
'''


if old_block in updated:
    updated = updated.replace(
        old_block,
        new_block,
        1,
    )

elif "snapshot = priority_snapshot(" in updated:
    print(
        "Shared priority logic already appears to exist "
        "in this file."
    )

else:
    fail(
        "Could not safely locate the old Patients-page "
        "priority block."
    )


# ============================================================
# 4. Validate syntax BEFORE replacing current file
# ============================================================

try:
    compile(
        updated,
        str(TARGET),
        "exec",
    )
except SyntaxError as exc:
    fail(
        "Python validation failed:\n"
        + str(exc)
    )


# ============================================================
# 5. Back up current file
# ============================================================

stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

backup = (
    ROOT
    / f"dashboard_metrics_service_backup_{stamp}.py"
)

shutil.copy2(
    TARGET,
    backup,
)


# ============================================================
# 6. Write corrected file
# ============================================================

TARGET.write_text(
    updated,
    encoding="utf-8",
)


# ============================================================
# 7. Create ZIP containing ONLY the updated project file
# ============================================================

zip_path = (
    ROOT
    / "CareAI_patient_tab_consistency_fix.zip"
)

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as archive:

    archive.write(
        TARGET,
        arcname="services/dashboard_metrics_service.py",
    )


print()
print("========================================")
print(" CAREAI PATIENT TAB FIX COMPLETE")
print("========================================")

print()
print("Updated:")
print(" services/dashboard_metrics_service.py")

print()
print("Backup:")
print(backup)

print()
print("ZIP:")
print(zip_path)

print()
print(
    "No database, templates, CSS, risk formulas, "
    "Agentic workflow or other CareAI modules were changed."
)