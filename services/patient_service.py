"""Patient master-data management for CareAI.

This service extends the existing ``patients`` table. It deliberately does not own
vitals, risk scores, agentic events, medications, wearable measurements or other
dynamic clinical records. Those modules keep their existing patient_id foreign keys
and therefore automatically resolve the latest patient master data.
"""
from __future__ import annotations

import io
import re
import secrets
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from services.db import get_conn, query_db
from services.clinical_evidence_service import priority_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATIENT_UPLOAD_DIR = PROJECT_ROOT / "static" / "uploads" / "patients"
MAX_PHOTO_BYTES = 5 * 1024 * 1024

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
PHONE_RE = re.compile(r"^[0-9+()\- .]{6,30}$")
VALID_SEX = {"", "M", "F", "X", "U"}
VALID_LANGUAGES = {"nl", "en", "de", "fr", "other"}
PRIORITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "stable": 1}

COUNTRY_OPTIONS = (
    'Afghanistan',
    'Albania',
    'Algeria',
    'American Samoa',
    'Andorra',
    'Angola',
    'Anguilla',
    'Antarctica',
    'Antigua and Barbuda',
    'Argentina',
    'Armenia',
    'Aruba',
    'Australia',
    'Austria',
    'Azerbaijan',
    'Bahamas',
    'Bahrain',
    'Bangladesh',
    'Barbados',
    'Belarus',
    'Belgium',
    'Belize',
    'Benin',
    'Bermuda',
    'Bhutan',
    'Bolivia',
    'Bonaire, Sint Eustatius and Saba',
    'Bosnia and Herzegovina',
    'Botswana',
    'Bouvet Island',
    'Brazil',
    'British Indian Ocean Territory',
    'Brunei',
    'Bulgaria',
    'Burkina Faso',
    'Burundi',
    'Cabo Verde',
    'Cambodia',
    'Cameroon',
    'Canada',
    'Cayman Islands',
    'Central African Republic',
    'Chad',
    'Chile',
    'China',
    'Christmas Island',
    'Cocos (Keeling) Islands',
    'Colombia',
    'Comoros',
    'Congo',
    'Cook Islands',
    'Costa Rica',
    'Croatia',
    'Cuba',
    'Curaçao',
    'Cyprus',
    'Czechia',
    'Côte d’Ivoire',
    'Democratic Republic of the Congo',
    'Denmark',
    'Djibouti',
    'Dominica',
    'Dominican Republic',
    'Ecuador',
    'Egypt',
    'El Salvador',
    'Equatorial Guinea',
    'Eritrea',
    'Estonia',
    'Eswatini',
    'Ethiopia',
    'Falkland Islands (Malvinas)',
    'Faroe Islands',
    'Fiji',
    'Finland',
    'France',
    'French Guiana',
    'French Polynesia',
    'French Southern Territories',
    'Gabon',
    'Gambia',
    'Georgia',
    'Germany',
    'Ghana',
    'Gibraltar',
    'Greece',
    'Greenland',
    'Grenada',
    'Guadeloupe',
    'Guam',
    'Guatemala',
    'Guernsey',
    'Guinea',
    'Guinea-Bissau',
    'Guyana',
    'Haiti',
    'Heard Island and McDonald Islands',
    'Holy See (Vatican City State)',
    'Honduras',
    'Hong Kong',
    'Hungary',
    'Iceland',
    'India',
    'Indonesia',
    'Iran',
    'Iraq',
    'Ireland',
    'Isle of Man',
    'Israel',
    'Italy',
    'Jamaica',
    'Japan',
    'Jersey',
    'Jordan',
    'Kazakhstan',
    'Kenya',
    'Kiribati',
    'Kosovo',
    'Kuwait',
    'Kyrgyzstan',
    'Laos',
    'Latvia',
    'Lebanon',
    'Lesotho',
    'Liberia',
    'Libya',
    'Liechtenstein',
    'Lithuania',
    'Luxembourg',
    'Macao',
    'Madagascar',
    'Malawi',
    'Malaysia',
    'Maldives',
    'Mali',
    'Malta',
    'Marshall Islands',
    'Martinique',
    'Mauritania',
    'Mauritius',
    'Mayotte',
    'Mexico',
    'Micronesia',
    'Moldova',
    'Monaco',
    'Mongolia',
    'Montenegro',
    'Montserrat',
    'Morocco',
    'Mozambique',
    'Myanmar',
    'Namibia',
    'Nauru',
    'Nepal',
    'Netherlands',
    'New Caledonia',
    'New Zealand',
    'Nicaragua',
    'Niger',
    'Nigeria',
    'Niue',
    'Norfolk Island',
    'North Korea',
    'North Macedonia',
    'Northern Mariana Islands',
    'Norway',
    'Oman',
    'Pakistan',
    'Palau',
    'Palestine',
    'Panama',
    'Papua New Guinea',
    'Paraguay',
    'Peru',
    'Philippines',
    'Pitcairn',
    'Poland',
    'Portugal',
    'Puerto Rico',
    'Qatar',
    'Romania',
    'Russia',
    'Rwanda',
    'Réunion',
    'Saint Barthélemy',
    'Saint Helena, Ascension and Tristan da Cunha',
    'Saint Kitts and Nevis',
    'Saint Lucia',
    'Saint Martin (French part)',
    'Saint Pierre and Miquelon',
    'Saint Vincent and the Grenadines',
    'Samoa',
    'San Marino',
    'Sao Tome and Principe',
    'Saudi Arabia',
    'Senegal',
    'Serbia',
    'Seychelles',
    'Sierra Leone',
    'Singapore',
    'Sint Maarten (Dutch part)',
    'Slovakia',
    'Slovenia',
    'Solomon Islands',
    'Somalia',
    'South Africa',
    'South Georgia and the South Sandwich Islands',
    'South Korea',
    'South Sudan',
    'Spain',
    'Sri Lanka',
    'Sudan',
    'Suriname',
    'Svalbard and Jan Mayen',
    'Sweden',
    'Switzerland',
    'Syria',
    'Taiwan',
    'Tajikistan',
    'Tanzania',
    'Thailand',
    'Timor-Leste',
    'Togo',
    'Tokelau',
    'Tonga',
    'Trinidad and Tobago',
    'Tunisia',
    'Turkmenistan',
    'Turks and Caicos Islands',
    'Tuvalu',
    'Türkiye',
    'Uganda',
    'Ukraine',
    'United Arab Emirates',
    'United Kingdom',
    'United States',
    'United States Minor Outlying Islands',
    'Uruguay',
    'Uzbekistan',
    'Vanuatu',
    'Venezuela',
    'Vietnam',
    'Virgin Islands, British',
    'Virgin Islands, U.S.',
    'Wallis and Futuna',
    'Western Sahara',
    'Yemen',
    'Zambia',
    'Zimbabwe',
    'Åland Islands',
)

CONTACT_RELATIONSHIP_OPTIONS = (
    'Spouse',
    'Partner',
    'Husband',
    'Wife',
    'Parent',
    'Mother',
    'Father',
    'Step-parent',
    'Child',
    'Son',
    'Daughter',
    'Step-child',
    'Sibling',
    'Brother',
    'Sister',
    'Grandparent',
    'Grandmother',
    'Grandfather',
    'Grandchild',
    'Grandson',
    'Granddaughter',
    'Aunt',
    'Uncle',
    'Niece',
    'Nephew',
    'Cousin',
    'Family',
    'Relative',
    'Friend',
    'Neighbour',
    'Caregiver',
    'Legal guardian',
    'Power of attorney',
    'Other',
)

SENSITIVE_AUDIT_FIELDS = {
    "phone", "email", "address_line1", "postal_code", "allergies", "clinical_notes",
    "family_contact", "profile_photo_path",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _text(value: Any, limit: int = 500) -> str:
    return (str(value or "").strip())[:limit]


def _bool(value: Any) -> int:
    return 1 if str(value or "").strip().lower() in {"1", "true", "yes", "on"} else 0


def _priority(*values: Any) -> str:
    levels = []
    for raw in values:
        value = _text(raw, 20).lower()
        if value == "attention":
            value = "medium"
        if value == "stable" or value not in PRIORITY_RANK:
            value = "low"
        levels.append(value)
    return max(levels or ["low"], key=lambda x: PRIORITY_RANK.get(x, 1))


def patient_age(birth_date: str | None) -> int | None:
    if not birth_date:
        return None
    try:
        born = date.fromisoformat(str(birth_date)[:10])
    except ValueError:
        return None
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def assignment_options() -> dict[str, Any]:
    """Return Patient Management select-list options from one shared source.

    Staff assignment choices remain database-driven. Countries and contact
    relationships are fixed display lists so the create/edit form does not accept
    free-text values for normal user interaction.
    """
    nurses = [dict(r) for r in query_db(
        "SELECT id,display_name,role FROM users WHERE active=1 AND role='nurse' ORDER BY display_name"
    )]
    clinicians = [dict(r) for r in query_db(
        "SELECT id,display_name,role FROM users WHERE active=1 AND role IN ('gp','admin') ORDER BY role,display_name"
    )]
    return {
        "nurses": nurses,
        "clinicians": clinicians,
        "countries": list(COUNTRY_OPTIONS),
        "relationships": list(CONTACT_RELATIONSHIP_OPTIONS),
    }


def _parse_optional_int(value: Any) -> int | None:
    text = _text(value, 30)
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _validate_user_assignment(user_id: int | None, allowed_roles: tuple[str, ...], label: str, errors: dict) -> None:
    if not user_id:
        return
    placeholders = ",".join("?" for _ in allowed_roles)
    row = query_db(
        f"SELECT id FROM users WHERE id=? AND active=1 AND role IN ({placeholders})",
        (user_id, *allowed_roles),
        one=True,
    )
    if not row:
        errors[label] = "Selected user is not available for this assignment."


def normalise_patient_form(form: Any) -> dict[str, Any]:
    return {
        "first_name": _text(form.get("first_name"), 80),
        "last_name": _text(form.get("last_name"), 100),
        "birth_date": _text(form.get("birth_date"), 10),
        "sex": _text(form.get("sex"), 1).upper(),
        "phone": _text(form.get("phone"), 40),
        "email": _text(form.get("email"), 180).lower(),
        "address_line1": _text(form.get("address_line1"), 200),
        "city": _text(form.get("city"), 100),
        "postal_code": _text(form.get("postal_code"), 24).upper(),
        "country": _text(form.get("country"), 100) or "Netherlands",
        "preferred_language": _text(form.get("preferred_language"), 20).lower() or "nl",
        "living_setting": _text(form.get("living_setting"), 80),
        "gp_name": _text(form.get("gp_name"), 120),
        "mobility_status": _text(form.get("mobility_status"), 120),
        "allergies": _text(form.get("allergies"), 1000),
        "clinical_notes": _text(form.get("clinical_notes"), 2000),
        "assigned_nurse_id": _parse_optional_int(form.get("assigned_nurse_id")),
        "assigned_clinician_id": _parse_optional_int(form.get("assigned_clinician_id")),
        "consent_monitoring": _bool(form.get("consent_monitoring")),
        "conditions": _text(form.get("conditions"), 2000),
        "contact_name": _text(form.get("contact_name"), 120),
        "contact_relationship": _text(form.get("contact_relationship"), 80),
        "contact_phone": _text(form.get("contact_phone"), 40),
        "contact_email": _text(form.get("contact_email"), 180).lower(),
        "contact_authorised": _bool(form.get("contact_authorised")),
        "contact_notification_consent": _bool(form.get("contact_notification_consent")),
        "contact_consent_reference": _text(form.get("contact_consent_reference"), 120),
        "contact_expires_at": _text(form.get("contact_expires_at"), 25),
    }


def validate_patient_form(data: dict[str, Any], *, creation: bool = False) -> dict[str, str]:
    errors: dict[str, str] = {}
    if not data["first_name"]:
        errors["first_name"] = "First name is required."
    if not data["last_name"]:
        errors["last_name"] = "Last name is required."
    if not data["birth_date"]:
        errors["birth_date"] = "Date of birth is required."
    else:
        try:
            dob = date.fromisoformat(data["birth_date"])
            if dob > date.today():
                errors["birth_date"] = "Date of birth cannot be in the future."
            if dob.year < 1900:
                errors["birth_date"] = "Please enter a valid date of birth."
        except ValueError:
            errors["birth_date"] = "Please enter a valid date of birth."
    if data["sex"] not in VALID_SEX:
        errors["sex"] = "Choose a supported sex value."
    if data["email"] and not EMAIL_RE.match(data["email"]):
        errors["email"] = "Enter a valid email address."
    if data["phone"] and not PHONE_RE.match(data["phone"]):
        errors["phone"] = "Enter a valid phone number."
    if data["contact_email"] and not EMAIL_RE.match(data["contact_email"]):
        errors["contact_email"] = "Enter a valid family/emergency contact email."
    if data["contact_phone"] and not PHONE_RE.match(data["contact_phone"]):
        errors["contact_phone"] = "Enter a valid family/emergency contact phone number."
    if data["preferred_language"] not in VALID_LANGUAGES:
        errors["preferred_language"] = "Choose a supported preferred language."
    if data["contact_notification_consent"] and not data["contact_authorised"]:
        errors["contact_notification_consent"] = "Authorise the contact before enabling notifications."
    if (data["contact_phone"] or data["contact_email"] or data["contact_relationship"] or data["contact_authorised"] or data["contact_notification_consent"]) and not data["contact_name"]:
        errors["contact_name"] = "Contact name is required when contact details or permissions are supplied."
    if data["contact_expires_at"]:
        try:
            datetime.fromisoformat(data["contact_expires_at"].replace("Z", "+00:00"))
        except ValueError:
            errors["contact_expires_at"] = "Enter a valid consent expiry date/time."

    _validate_user_assignment(data["assigned_nurse_id"], ("nurse",), "assigned_nurse_id", errors)
    _validate_user_assignment(data["assigned_clinician_id"], ("gp", "admin"), "assigned_clinician_id", errors)
    return errors


def _condition_values(raw: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,\n;]+", raw or ""):
        value = part.strip()[:180]
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            values.append(value)
    return values


def possible_duplicates(data: dict[str, Any], exclude_id: int | None = None) -> list[dict]:
    if not data.get("first_name") or not data.get("last_name") or not data.get("birth_date"):
        return []
    sql = """
        SELECT id,external_ref,first_name,last_name,birth_date,city,active
        FROM patients
        WHERE lower(first_name)=lower(?) AND lower(last_name)=lower(?) AND birth_date=?
    """
    args: list[Any] = [data["first_name"], data["last_name"], data["birth_date"]]
    if exclude_id:
        sql += " AND id<>?"
        args.append(exclude_id)
    return [dict(r) for r in query_db(sql, tuple(args))]


def _next_external_ref(conn) -> str:
    row = conn.execute(
        """
        SELECT MAX(CAST(SUBSTR(external_ref,7) AS INTEGER)) AS max_no
        FROM patients
        WHERE external_ref GLOB 'NL-CR-[0-9]*'
        """
    ).fetchone()
    next_no = int(row["max_no"] or 0) + 1
    while True:
        candidate = f"NL-CR-{next_no:03d}"
        exists = conn.execute("SELECT 1 FROM patients WHERE external_ref=?", (candidate,)).fetchone()
        if not exists:
            return candidate
        next_no += 1


def _save_photo(file_storage) -> str | None:
    if not file_storage or not getattr(file_storage, "filename", ""):
        return None
    payload = file_storage.read(MAX_PHOTO_BYTES + 1)
    if len(payload) > MAX_PHOTO_BYTES:
        raise ValueError("Profile photo must be 5 MB or smaller.")
    if not payload:
        raise ValueError("Profile photo is empty.")
    try:
        image = Image.open(io.BytesIO(payload))
        image.verify()
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Profile photo must be a valid JPG, JPEG, PNG or WebP image.") from exc
    if (image.format or "").upper() not in {"JPEG", "PNG", "WEBP"}:
        raise ValueError("Profile photo must be JPG, JPEG, PNG or WebP.")

    # Re-encode to WebP to strip metadata and prevent uploaded filenames from becoming paths.
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGB")
    image.thumbnail((1200, 1200))
    PATIENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"patient_{secrets.token_hex(16)}.webp"
    target = PATIENT_UPLOAD_DIR / filename
    image.save(target, format="WEBP", quality=90, method=6)
    return f"uploads/patients/{filename}"


def _remove_photo(relative_path: str | None) -> None:
    if not relative_path or not str(relative_path).startswith("uploads/patients/"):
        return
    target = (PROJECT_ROOT / "static" / str(relative_path)).resolve()
    allowed = PATIENT_UPLOAD_DIR.resolve()
    if allowed not in target.parents:
        return
    try:
        target.unlink(missing_ok=True)
    except OSError:
        pass


def _sync_conditions(conn, patient_id: int, raw_conditions: str) -> bool:
    wanted = _condition_values(raw_conditions)
    existing = conn.execute(
        "SELECT id,display,active FROM conditions WHERE patient_id=?", (patient_id,)
    ).fetchall()
    by_name = {(r["display"] or "").strip().casefold(): r for r in existing if r["display"]}
    wanted_keys = {v.casefold() for v in wanted}
    changed = False
    for key, row in by_name.items():
        should_active = 1 if key in wanted_keys else 0
        if int(row["active"] or 0) != should_active:
            conn.execute("UPDATE conditions SET active=? WHERE id=?", (should_active, row["id"]))
            changed = True
    for value in wanted:
        key = value.casefold()
        if key not in by_name:
            conn.execute(
                "INSERT INTO conditions(patient_id,code,display,onset_date,active) VALUES(?,?,?,?,1)",
                (patient_id, None, value, date.today().isoformat()),
            )
            changed = True
    return changed


def _primary_contact(conn, patient_id: int):
    return conn.execute(
        """
        SELECT * FROM patient_contacts
        WHERE patient_id=? AND active=1
        ORDER BY authorised_for_updates DESC, notification_consent DESC, id ASC
        LIMIT 1
        """,
        (patient_id,),
    ).fetchone()


def _sync_primary_contact(conn, patient_id: int, data: dict[str, Any]) -> bool:
    existing = _primary_contact(conn, patient_id)
    now = _now()
    name = data["contact_name"]
    if not name:
        if existing:
            conn.execute("UPDATE patient_contacts SET active=0,updated_at=? WHERE id=?", (now, existing["id"]))
            return True
        return False

    values = {
        "display_name": name,
        "relationship": data["contact_relationship"] or "Emergency contact",
        "phone": data["contact_phone"] or None,
        "email": data["contact_email"] or None,
        "authorised_for_updates": int(data["contact_authorised"]),
        "notification_consent": int(data["contact_notification_consent"]),
        "consent_reference": data["contact_consent_reference"] or f"PATIENT-CONTACT-{patient_id}",
        "expires_at": data["contact_expires_at"] or None,
    }
    if existing:
        changed = any((existing[k] or None) != (v or None) for k, v in values.items())
        if changed:
            conn.execute(
                """
                UPDATE patient_contacts
                SET display_name=?,relationship=?,phone=?,email=?,authorised_for_updates=?,
                    notification_consent=?,consent_reference=?,expires_at=?,active=1,updated_at=?
                WHERE id=?
                """,
                (*values.values(), now, existing["id"]),
            )
        return changed

    conn.execute(
        """
        INSERT INTO patient_contacts(
          patient_id,display_name,relationship,phone,email,authorised_for_updates,
          notification_consent,consent_reference,expires_at,active,created_at,updated_at
        ) VALUES(?,?,?,?,?,?,?,?,?,1,?,?)
        """,
        (patient_id, *values.values(), now, now),
    )
    return True


def _patient_insert_values(data: dict[str, Any], actor_id: int, actor_role: str) -> dict[str, Any]:
    values = dict(data)
    if actor_role == "nurse":
        values["assigned_nurse_id"] = actor_id
        values["assigned_clinician_id"] = None
    return values


def create_patient(data: dict[str, Any], *, actor_id: int, actor_role: str, photo_file=None) -> tuple[int, str, dict[str, Any]]:
    photo_path = _save_photo(photo_file) if photo_file and getattr(photo_file, "filename", "") else None
    values = _patient_insert_values(data, actor_id, actor_role)
    now = _now()
    conn = get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        external_ref = _next_external_ref(conn)
        cur = conn.execute(
            """
            INSERT INTO patients(
              external_ref,first_name,last_name,birth_date,sex,city,country,preferred_language,
              living_setting,gp_name,phone,email,address_line1,postal_code,profile_photo_path,
              mobility_status,allergies,clinical_notes,assigned_nurse_id,assigned_clinician_id,
              emergency_contact_name,emergency_contact_relation,consent_monitoring,consent_updated_at,
              active,current_status,created_at,updated_at,updated_by
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,'stable',?,?,?)
            """,
            (
                external_ref, values["first_name"], values["last_name"], values["birth_date"], values["sex"] or None,
                values["city"] or None, values["country"], values["preferred_language"], values["living_setting"] or None,
                values["gp_name"] or None, values["phone"] or None, values["email"] or None,
                values["address_line1"] or None, values["postal_code"] or None, photo_path,
                values["mobility_status"] or None, values["allergies"] or None, values["clinical_notes"] or None,
                values["assigned_nurse_id"], values["assigned_clinician_id"], values["contact_name"] or None,
                values["contact_relationship"] or None, int(values["consent_monitoring"]), now, now, now, actor_id,
            ),
        )
        patient_id = int(cur.lastrowid)
        _sync_conditions(conn, patient_id, values["conditions"])
        _sync_primary_contact(conn, patient_id, values)
        conn.execute(
            "INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
            (patient_id, "patient_created", "patient_management", f"Patient record {external_ref} created through Patient Management"),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        _remove_photo(photo_path)
        raise
    finally:
        conn.close()
    changes = {"created": external_ref, "fields": ["patient master", "conditions", "family contact"]}
    if photo_path:
        changes["profile_photo"] = "uploaded"
    return patient_id, external_ref, changes


ADMIN_FIELDS = {
    "first_name", "last_name", "birth_date", "sex", "assigned_nurse_id", "assigned_clinician_id",
}
COMMON_EDIT_FIELDS = {
    "phone", "email", "address_line1", "city", "postal_code", "country", "preferred_language",
    "living_setting", "gp_name", "mobility_status", "allergies", "clinical_notes", "consent_monitoring",
}


def update_patient(patient_id: int, data: dict[str, Any], *, actor_id: int, actor_role: str, photo_file=None) -> dict[str, tuple[Any, Any]]:
    row = query_db("SELECT * FROM patients WHERE id=?", (patient_id,), one=True)
    if not row:
        raise LookupError("Patient not found")
    old = dict(row)
    allowed = set(COMMON_EDIT_FIELDS)
    if actor_role == "admin":
        allowed |= ADMIN_FIELDS
    new_photo = _save_photo(photo_file) if photo_file and getattr(photo_file, "filename", "") else None
    now = _now()
    changes: dict[str, tuple[Any, Any]] = {}
    conn = get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        for field in sorted(allowed):
            old_value = old.get(field)
            new_value = data.get(field)
            if field in {"assigned_nurse_id", "assigned_clinician_id"}:
                new_value = new_value or None
            elif field == "consent_monitoring":
                new_value = int(new_value)
            else:
                new_value = new_value or None
            if old_value != new_value:
                conn.execute(f"UPDATE patients SET {field}=? WHERE id=?", (new_value, patient_id))
                changes[field] = (old_value, new_value)

        if new_photo:
            conn.execute("UPDATE patients SET profile_photo_path=? WHERE id=?", (new_photo, patient_id))
            changes["profile_photo_path"] = (old.get("profile_photo_path"), new_photo)

        if _sync_conditions(conn, patient_id, data["conditions"]):
            changes["conditions"] = ("previous active conditions", "updated active conditions")
        if _sync_primary_contact(conn, patient_id, data):
            changes["family_contact"] = ("previous contact settings", "updated contact settings")
            conn.execute(
                "UPDATE patients SET emergency_contact_name=?,emergency_contact_relation=? WHERE id=?",
                (data["contact_name"] or None, data["contact_relationship"] or None, patient_id),
            )

        if changes:
            conn.execute(
                "UPDATE patients SET updated_at=?,updated_by=?,consent_updated_at=CASE WHEN ? THEN ? ELSE consent_updated_at END WHERE id=?",
                (now, actor_id, 1 if "consent_monitoring" in changes else 0, now, patient_id),
            )
            conn.execute(
                "INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
                (patient_id, "patient_updated", "patient_management", f"Patient master record updated ({', '.join(sorted(changes))})"),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        _remove_photo(new_photo)
        raise
    finally:
        conn.close()

    if new_photo and changes:
        _remove_photo(old.get("profile_photo_path"))
    return changes


def archive_patient(patient_id: int, *, actor_id: int) -> None:
    active_agentic = query_db(
        "SELECT id FROM agentic_runs WHERE patient_id=? AND status IN ('running','awaiting_approval','responding') ORDER BY id DESC LIMIT 1",
        (patient_id,), one=True,
    )
    if active_agentic:
        raise ValueError("Patient cannot be archived while an Agentic Care workflow is active.")
    try:
        active_admission = query_db(
            "SELECT id FROM hospital_admissions WHERE patient_id=? AND status='admitted' ORDER BY id DESC LIMIT 1",
            (patient_id,), one=True,
        )
    except Exception:
        active_admission = None
    if active_admission:
        raise ValueError("Patient cannot be archived while an active hospital admission exists.")

    now = _now()
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            UPDATE patients
            SET active=0,archived_at=?,archived_by=?,updated_at=?,updated_by=?
            WHERE id=? AND active=1
            """,
            (now, actor_id, now, actor_id, patient_id),
        )
        if cur.rowcount:
            conn.execute(
                "INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
                (patient_id, "patient_archived", "patient_management", "Patient archived; historical clinical records retained"),
            )
        conn.commit()
    finally:
        conn.close()


def reactivate_patient(patient_id: int, *, actor_id: int) -> None:
    now = _now()
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            UPDATE patients
            SET active=1,archived_at=NULL,archived_by=NULL,updated_at=?,updated_by=?
            WHERE id=? AND active=0
            """,
            (now, actor_id, patient_id),
        )
        if cur.rowcount:
            conn.execute(
                "INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
                (patient_id, "patient_reactivated", "patient_management", "Patient reactivated through Patient Management"),
            )
        conn.commit()
    finally:
        conn.close()


def primary_contact(patient_id: int) -> dict[str, Any] | None:
    row = query_db(
        """
        SELECT * FROM patient_contacts
        WHERE patient_id=? AND active=1
        ORDER BY authorised_for_updates DESC,notification_consent DESC,id ASC LIMIT 1
        """,
        (patient_id,),
        one=True,
    )
    if row:
        return dict(row)

    # Preserve the existing family-user demo as a read-compatible fallback without
    # turning every legacy emergency contact into an authorised notification target.
    family = query_db(
        """
        SELECT id,display_name,relationship,NULL phone,email,1 authorised_for_updates,
               consent_granted notification_consent,consent_reference,expires_at,active
        FROM family_users WHERE patient_id=? AND active=1 ORDER BY id LIMIT 1
        """,
        (patient_id,),
        one=True,
    )
    if family:
        result = dict(family)
        result["source"] = "family_user"
        return result

    legacy = query_db(
        "SELECT emergency_contact_name display_name,emergency_contact_relation relationship FROM patients WHERE id=?",
        (patient_id,), one=True,
    )
    if legacy and legacy["display_name"]:
        return {
            "display_name": legacy["display_name"], "relationship": legacy["relationship"],
            "phone": None, "email": None, "authorised_for_updates": 0,
            "notification_consent": 0, "consent_reference": None, "expires_at": None,
            "active": 1, "source": "legacy_emergency_contact",
        }
    return None


def patient_form_data(patient_id: int) -> dict[str, Any]:
    row = query_db("SELECT * FROM patients WHERE id=?", (patient_id,), one=True)
    if not row:
        raise LookupError("Patient not found")
    data = dict(row)
    conditions = query_db(
        "SELECT display FROM conditions WHERE patient_id=? AND active=1 ORDER BY display", (patient_id,)
    )
    data["conditions"] = "\n".join(r["display"] for r in conditions if r["display"])
    contact = primary_contact(patient_id) or {}
    data.update({
        "contact_name": contact.get("display_name", ""),
        "contact_relationship": contact.get("relationship", ""),
        "contact_phone": contact.get("phone", ""),
        "contact_email": contact.get("email", ""),
        "contact_authorised": int(contact.get("authorised_for_updates") or 0),
        "contact_notification_consent": int(contact.get("notification_consent") or 0),
        "contact_consent_reference": contact.get("consent_reference", "") or "",
        "contact_expires_at": contact.get("expires_at", "") or "",
    })
    return data


def _list_base_rows(status: str = "active") -> list[dict]:
    """Return Patient Management rows using one canonical CURRENT priority.

    Patient Management must show the same current patient priority as:
    - Patient 360
    - Command Centre
    - Population
    - Reports
    - Home dashboard

    Completed/rejected historical Agentic workflows remain available in
    Agentic Care history, but they do not override the patient's current
    Care Intelligence status.
    """

    where = ""

    if status == "active":
        where = "WHERE p.active=1"
    elif status == "archived":
        where = "WHERE p.active=0"

    rows = query_db(
        f"""
        SELECT
            p.*,

            n.display_name AS assigned_nurse_name,
            cuser.display_name AS assigned_clinician_name,

            (
                SELECT r.level
                FROM risk_scores r
                WHERE r.patient_id=p.id
                  AND r.risk_type='overall'
                ORDER BY r.id DESC
                LIMIT 1
            ) AS risk_level,

            (
                SELECT r.score
                FROM risk_scores r
                WHERE r.patient_id=p.id
                  AND r.risk_type='overall'
                ORDER BY r.id DESC
                LIMIT 1
            ) AS risk_score,

            (
                SELECT GROUP_CONCAT(c.display, ', ')
                FROM conditions c
                WHERE c.patient_id=p.id
                  AND c.active=1
            ) AS condition_summary

        FROM patients p

        LEFT JOIN users n
            ON n.id=p.assigned_nurse_id

        LEFT JOIN users cuser
            ON cuser.id=p.assigned_clinician_id

        {where}

        ORDER BY
            p.active DESC,
            p.last_name,
            p.first_name
        """
    )

    result = []

    for row in rows:
        item = dict(row)

        item["age"] = patient_age(
            item.get("birth_date")
        )

        # ----------------------------------------------------------
        # Preserve database values for audit/debugging only.
        # ----------------------------------------------------------

        item["stored_current_status"] = (
            item.get("current_status")
        )

        item["care_risk_level"] = _priority(
            item.get("risk_level")
        )

        item["care_risk_score"] = int(
            item.get("risk_score") or 0
        )

        # ----------------------------------------------------------
        # Resolve ONE canonical CURRENT patient priority.
        #
        # priority_snapshot() uses:
        #
        # 1. latest overall Care Intelligence score
        # 2. only genuinely ACTIVE Agentic/Hospital workflows
        #
        # Completed historical workflows are therefore not allowed
        # to keep a recovered patient Critical.
        # ----------------------------------------------------------

        try:
            snapshot = priority_snapshot(
                int(item["id"])
            )

        except sqlite3.OperationalError:
            # Safe fallback for an incomplete/older database.
            #
            # Level and score still come from the same Care
            # Intelligence assessment.
            snapshot = {
                "level": item["care_risk_level"],
                "score": item["care_risk_score"],
                "reason": (
                    "Current Care Intelligence risk assessment"
                ),
                "source": "Care intelligence",
                "run_id": None,
                "module_key": None,
                "updated_at": None,
            }

        resolved_level = _priority(
            snapshot.get("level")
        )

        resolved_score = int(
            snapshot.get("score") or 0
        )

        # ----------------------------------------------------------
        # These fields now ALWAYS belong to the same current snapshot.
        #
        # This prevents impossible combinations such as:
        #
        # CRITICAL · 30
        # CRITICAL · 12
        # CRITICAL · 40
        # ----------------------------------------------------------

        item["priority"] = resolved_level

        item["priority_label"] = (
            resolved_level.title()
        )

        item["risk_score"] = resolved_score

        item["priority_score"] = resolved_score

        item["current_priority"] = resolved_level

        item["current_priority_score"] = (
            resolved_score
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

        item["priority_module_key"] = (
            snapshot.get("module_key")
        )

        item["priority_updated_at"] = (
            snapshot.get("updated_at")
        )

        # ----------------------------------------------------------
        # Backwards-compatible current_status for any existing
        # Patient Management component that still reads it.
        #
        # IMPORTANT:
        # this modifies only this Python dictionary.
        # It does NOT UPDATE the patients table.
        # ----------------------------------------------------------

        item["current_status"] = resolved_level

        # ----------------------------------------------------------
        # Backwards-compatible Agentic field.
        #
        # Only an ACTIVE workflow receives a value here.
        # Completed/rejected historical workflows do not.
        # ----------------------------------------------------------

        item["agentic_severity"] = (
            resolved_level
            if snapshot.get("run_id")
            else None
        )

        result.append(item)

    return result


def list_patients(*, q: str = "", status: str = "active", risk: str = "all", condition: str = "", nurse_id: int | None = None) -> list[dict]:
    status = status if status in {"active", "archived", "all"} else "active"
    risk = risk if risk in {"all", "high_critical", "medium", "low"} else "all"
    qf = _text(q, 120).casefold()
    cf = _text(condition, 120).casefold()
    result = []
    for item in _list_base_rows(status):
        haystack = " ".join(str(item.get(k) or "") for k in (
            "external_ref", "first_name", "last_name", "city", "gp_name", "assigned_nurse_name", "assigned_clinician_name"
        )).casefold()
        if qf and qf not in haystack:
            continue
        if cf and cf not in str(item.get("condition_summary") or "").casefold():
            continue
        if nurse_id and int(item.get("assigned_nurse_id") or 0) != int(nurse_id):
            continue
        if risk == "high_critical" and item["priority"] not in {"high", "critical"}:
            continue
        if risk in {"medium", "low"} and item["priority"] != risk:
            continue
        result.append(item)
    result.sort(key=lambda p: (-PRIORITY_RANK.get(p["priority"], 1), p["last_name"].casefold(), p["first_name"].casefold()))
    return result


def patient_summary_counts() -> dict[str, int]:
    active = _list_base_rows("active")
    archived_row = query_db("SELECT COUNT(*) c FROM patients WHERE active=0", one=True)
    total_row = query_db("SELECT COUNT(*) c FROM patients", one=True)
    return {
        "total": int(total_row["c"] if total_row else 0),
        "active": len(active),
        "high_critical": sum(1 for p in active if p["priority"] in {"high", "critical"}),
        "archived": int(archived_row["c"] if archived_row else 0),
    }


def audit_change_details(changes: dict[str, tuple[Any, Any]], role: str) -> dict[str, Any]:
    safe: list[dict[str, Any]] = []
    for field, (old, new) in changes.items():
        if field in SENSITIVE_AUDIT_FIELDS or field in {"conditions"}:
            safe.append({"field": field, "old": "[redacted]", "new": "[changed]"})
        else:
            safe.append({"field": field, "old": old, "new": new})
    return {"role": role, "changes": safe}
