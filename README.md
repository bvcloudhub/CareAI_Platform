# Care.AI — Full Elderly Remote Care Prototype

A Flask + SQLite demo platform implementing the requested Sprint 2–4 functionality with a theme inspired by the supplied dark-green / coral / orange visual reference.

## Included

### Sprint 2 — Intelligence
- Explainable risk engine
- COPD deterioration scenario
- Fall + immobility scenario
- Medication adherence scenario
- Cardiac, infection and frailty risk signals
- AI-style explainable care insights

### Sprint 3 — Agentic workflow
- Monitoring Agent
- Risk Agent
- Care Agent
- Agent orchestrator
- Task orchestration
- Alert creation
- Patient/family/care-team notification queue
- Human-in-the-loop actions

### Sprint 4 — Enterprise polish
- FHIR R4-style synthetic export
- Audit log
- Consent, lawful basis, purpose, retention and EU residency metadata
- Role-based access: admin / nurse / GP
- Hospital / remote-care command centre
- Population analytics
- Security & GDPR governance screen
- Integrations architecture
- 3D-style interactive home digital twin
- Synthetic European demo dataset

## Run

**Requires CPython 3.10-3.13.** TensorFlow publishes no wheels for 3.14, and the
lung imaging models need it.

```bash
py -3.13 -m venv .venv
.venv/Scripts/activate                  # source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5002

The first analysis on each imaging module loads its model into memory, which
takes 10-30 seconds. Subsequent runs are fast.

### Demo users
- nurse@care.ai / demo123
- gp@care.ai / demo123
- admin@care.ai / demo123

## Recommended demo
1. Open **Jan de Vries**
2. Run **COPD deterioration**
3. Review Care Intelligence and generated task / alert
4. Open **3D Digital Twin**
5. Run **Fall + immobility**
6. Return to Command Centre
7. Open Population Analytics
8. Show FHIR, Security & GDPR, and Audit Trail

## Clinical / regulatory disclaimer
This is a synthetic-data prototype. The rules, risk scores, alerts and outputs are not validated medical-device functions and must not be used for diagnosis, treatment, emergency response or real patient care.

## Production security roadmap
For production, replace SQLite with an enterprise database, use OIDC/SSO + MFA, TLS, secret manager/KMS, encryption at rest, row-level access policy, SIEM/immutable audit logging, backup/DR, DPIA, DPO/privacy review, DSR tooling, security testing and formal MDR/AI Act assessment based on intended purpose.


## v3 additions
- Independent scenario reset so COPD/fall/medication demos do not contaminate each other
- Latest-only risk cards (no duplicate historical cards)
- Separate Fall Risk Prediction vs Active Fall Event Detection
- 94% synthetic sensor-fusion fall event using IMU impact, orientation, edge-camera posture and immobility
- Live 3D fall animation and stepwise Agent activity
- Netherlands/EU device integration catalogue (Apple Watch, Withings, OMRON, Sensara/radar, BLE devices, cameras)


## v4 — Care.AI Copilot
- Clinician Copilot across all synthetic patients
- Grounded patient health summaries
- Questions about COPD, fall risk, active fall events, medication adherence, trends and care tasks
- Patient-facing chat in simpler language
- Patient → nurse / GP question routing
- Clinician inbox and responses
- Urgent-language safety routing
- Chat history stored in SQLite
- Patient and clinician conversations remain separate
- Prototype uses deterministic database-grounded responses so the demo does not fabricate clinical facts
- Architecture is ready for replacement with an enterprise LLM/RAG layer later

### Demo
Clinician:
- Open **Care Copilot**
- Select Theo
- Ask "Summarize current health status"
- Ask "What is the fall risk and why?"
- Ask "What should the care team review next?"

Patient:
- Open a Patient 360 page
- Click **Patient Chat**
- Ask "How am I doing today?"
- Send a question to Nurse/GP
- Open **Patient Inbox** as the clinician and respond

### Important
The patient portal in this prototype intentionally has no real identity/authentication layer. Production requires strong patient authentication, MFA/passkeys, explicit authorization and patient-access auditing.

## v6 — Video Teleconsultation + Family Access
- Patient can request a video consultation with nurse or GP
- Browser camera preview via WebRTC getUserMedia for the prototype
- Consultation room includes live Care.AI vitals/risk context and chat
- Family access portal with delegated-consent metadata
- Dummy family users for NL-CR-003 Pieter Bakker
- Laura Bakker (daughter): laura.bakker@demo.nl / family123
- Thomas Bakker (son): thomas.bakker@demo.nl / family123
- Family can view Pieter's health summary, current alerts/tasks, recent timeline
- Family can ask Care.AI questions about Pieter
- Family can message nurse/GP
- Laura's demo access includes teleconsultation permission
- App remains on port 5002

### Teleconsultation production note
The prototype uses the local browser camera and a simulated remote-clinician tile. Production should use an EU-hosted WebRTC/SFU provider or self-hosted stack, strong room authentication, short-lived tokens, consent, encryption, audit metadata and retention controls.

## v7 — Multilingual + AI Diagnostics
- English / Dutch / German language selector in the top UI
- Language preference stored in the Flask session and applied across major screens
- New public landing page with older-adult + clinician visual
- New AI Diagnostics module in Command Centre
- Skin AI demo
- Lung Cancer AI demo
- Wound AI demo
- Upload-based deterministic demo analysis with explainable result and next-step text
- Diagnostic demo is clearly marked as non-clinical / not a medical device
- Existing family portal, patient Copilot, clinician inbox, digital twin and teleconsultation retained
- Default port remains 5002

## v8 — Full reference-theme rewrite
- Entire navigation moved to a top green navigation bar matching the supplied Care.AI visual
- Public landing page rebuilt around the supplied theme
- Elderly patient + clinician hero visual
- Full NL / EN / DE language selector in top menu
- Added Patients and Reports pages
- AI Diagnostics remains available from top navigation and Command Centre
- Skin AI, Lung Cancer AI and Wound AI demo modules retained
- Existing Patient 360, 3D Digital Twin, Agentic AI, Copilot, Patient Inbox, Family Access, Teleconsultation, Integrations, Security/GDPR and Audit remain
- Port stays 5002


## v9 - Real lung imaging models

The AI Diagnostics tab now runs actual Keras models, ported from
`Lung_Cancer_V3/third_party/lung_ai_flask`, instead of the hash-based demo stub.
Skin AI and Wound AI remain rule-based demos and are labelled as such.

### Models (`models/`)

| File | Task | Classes | Input | Architecture |
|---|---|---|---|---|
| `lung_ct_malignancy_densenet.h5` | Lung CT malignancy screening | Malignant, Normal | 256x256 | DenseNet201 |
| `lung_histology_subtype.h5` | Lung histopathology subtype | Adenocarcinoma, Benign, Squamous cell carcinoma | 224x224 | EfficientNetB0 |
| `chest_xray_pneumonia_resnet50.h5` | Chest X-ray pneumonia screening | Normal, Pneumonia | 224x224 | ResNet50 |

All three take **raw 0-255 RGB**. Each ships beside a `*_classes.json` where the
source provided one.

### Naming

The source shipped the pneumonia model as `lung_cancer_resnet50.h5`. Its own
`config.py`, `xray_classes.json` and `train_xray.py` all give its classes as
NORMAL/PNEUMONIA, and its test images come from the Kermany pneumonia dataset -
it does not detect cancer. It is renamed here and surfaced as a separate
"adjacent screening" module so the Lung Cancer AI tab does not overstate what it
can do.

### Fixes carried over the port

- **Double rescaling.** The source fed the X-ray model images scaled to [0,1],
  but that graph opens with `Rescaling(1/255)`, so inputs were divided by 255
  twice. The effect is not cosmetic: a normal chest X-ray came back as
  PNEUMONIA at 100% confidence. All three models are now fed raw 0-255.
- **Inflated confidence.** The source mapped every probability into a fixed
  0.85-0.95 band before display, so a genuine 51% prediction showed as 85%. The
  real probability is now reported, and anything under 60% is flagged.
- **Grad-CAM legibility.** A flat 60/40 blend tinted whole slices blue. The
  overlay now fades in with the heat, leaving cold regions readable.

### Loading legacy weights

The `.h5` files are Keras 2.15 artifacts and Keras 3 cannot read two of them, so
`services/lung_imaging_service.py` tries three strategies in order: Keras 3,
then `tf-keras` (Keras 2 compat), then `tf-keras` with the model config patched.
The patch exists for the histology model, whose `Lambda` layer holds Python
bytecode marshalled by the training interpreter and unreadable on 3.13; the
wrapped function (`efficientnet.preprocess_input`) is a documented no-op in
current Keras, so it is replaced with a linear `Activation`. Which strategy was
used is shown in the Technical details panel.

### Sample scans

`sample_images/` holds CT, histopathology and chest X-ray examples copied from
the source project's test set, for checking the modules end to end.

### Clinical status
Unchanged: research prototype, not a medical device, not validated for
diagnosis, every output requires clinician review.


## v10 - Real wound imaging models

Wound AI now runs the PyTorch pipeline ported from `WoundAI`, replacing the
demo stub. Skin AI is the only rule-based demo left.

### Pipeline

Two routes, mirroring the source's `analyze()` flow:

- **Wound type unknown:** image-quality check → wound/non-wound gate →
  4-class wound type with Grad-CAM → boundary segmentation. A confident
  surgical-wound call also runs the infection-risk screen. That follow-up is
  new: the source only ran the screen when a clinician pre-selected "surgical".
- **Known surgical wound:** quality check → gate (reject limit 0.02 instead of
  0.10, as incisions look unlike open ulcers) → infection-risk screen →
  boundary segmentation. No wound-type model and no Grad-CAM, as in the source.

Images the quality check or gate screens out are deleted, as in the source.

### Models (`models/`)

| File | Task | Output | Evaluation |
|---|---|---|---|
| `wound_gate_efficientnet_b0.pt` | Wound vs non-wound gate | wound / non-wound | held-out: accuracy 97.8%, non-wound specificity 95.1% (n=224) |
| `wound_type_efficientnet_b3.pt` + `wound_type_calibration.json` | Wound type | diabetic, pressure, surgical, venous ulcer | held-out: accuracy 80.3%, macro-F1 0.78 (n=183); pressure-ulcer sensitivity only 56% |
| `wound_segmentation_deeplabv3.pt` | Boundary and measurement | mask; area, length, width, perimeter in px | held-out: Dice 0.84, IoU 0.74 |
| `surgical_wound_risk_v9_biomedclip_5fold_delta.pt` + `surgical_wound_risk_v8a_densenet121_5fold.pt` + `surgical_wound_risk_ensemble.json` | Surgical infection-risk image screen | elevated / low category and a score | development folds only: sensitivity 80%, specificity 75% (n=549). Never run on a held-out test set |

Each ships beside its source `*_test_metrics.json` where one existed, and the
UI shows the held-out figures next to each result.

### Naming

Every source classifier was saved as `best.pt` inside its run folder, so the
files are renamed to task + architecture. The ensemble manifest pointed at
absolute paths on the original author's machine; those are replaced with
relative paths, and the manifest now carries the ensemble's development
evaluation so the UI can state it.

### Deliberate differences from the source

- **Safe checkpoint loading.** `services/checkpoint_loading.py` tries
  `torch.load(weights_only=True)` first and falls back to full unpickling only
  when a file needs it. The Technical details panel records which path ran.
- **Grad-CAM lock.** The source registered a hook per request on a shared
  model, so two concurrent requests could read each other's activations.
- **Grad-CAM and boundary as plain overlays.** The source burnt English captions
  and measurements into the images; CareAI shows them in the page, translated.
- **EXIF orientation.** Phone photos are transposed upright before inference,
  so overlays line up with the upload as the browser displays it.
- **V9 fold deltas are validated.** The source applied them with
  `load_state_dict(strict=False)`, which silently skips renamed keys. A
  mismatch against the installed open_clip now fails loudly instead.

### What it does not do

- No infection diagnosis, and no healing-stage, necrosis, slough, dehiscence,
  burn or trauma assessment.
- Measurements are pixels only; there is no physical-scale input.
- Not ported: OTP sign-up, patient and wound records, PDF reports, manual
  boundary annotation, the structured infection-risk rules (they score form
  data, not images) and the OpenAI explanation layer. That last one needs the
  source's API key, and its `.env` was not copied.

### Runtime

- torch 2.14 (CPU), OpenCV, open_clip and transformers; see `requirements.txt`.
- The ensemble's BiomedCLIP base
  (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, ~0.8 GB) is
  downloaded from HuggingFace on first use into the local HuggingFace cache.
- On CPU the first surgical screen takes about 13 s including model load; later
  screens take 4-5 s.

### Sample images

`sample_images/wound/` holds two held-out test images per wound type, the
source README's demo image (`surgical_wound_source_demo_104.jpg`), and two
non-wound images the gate rejects.

## v11 - Floating care assistant (chatbot)

A floating care assistant now sits in the bottom-right corner of every page. It is modelled on the
MediCare AI chatbot in Lung_Cancer_V3 (`services/clinical_chat.py` + `templates/home.html`): describe
a symptom and it asks a short series of yes/no questions, then gives an assessment with an urgency
level and self-care advice. On the patient portal and the family view it can also send that
summary to the patient's nurse inbox - CareAI's version of MediCare's "book an appointment".

### Guided symptom flows (EN / NL / DE)
- From MediCare AI: fever, cough, coughing up blood, chest pain, headache, body pain.
- Added for older adults: breathlessness, fall or dizziness, sudden confusion, urinary problems,
  swollen legs.

11 flows and 46 questions, in `services/care_bot_flows.py` and `services/care_bot_flows_elderly.py`.
Each ends in one of four levels: emergency (call 112), see a doctor today, book a GP appointment,
or self-care. When a message mentions several symptoms, the most dangerous flow wins.

### Questions outside the flows
Answered by OpenAI through the existing Hospital AI switch (`CAREAI_USE_OPENAI=1`, `OPENAI_API_KEY`,
`OPENAI_MODEL` in `.env`), with a 20-second timeout. The model sees only the person's recent
messages in that chat - no names or patient identifiers - and every AI reply is labelled as coming
from an external AI service. With the switch off, or if the call fails, the bot lists what it can
help with instead. OpenAI is never used for the guided flows or for emergencies. MediCare's fixed
COVID-19 and lung-cancer treatment texts were not ported; those questions now take the AI path.

### Safety changes compared with MediCare AI
- Every message is screened for emergency and crisis phrases in all three languages before
  anything else. A hit ends the flow and tells the person to call 112; crisis replies add
  113 Zelfmoordpreventie and TelefonSeelsorge.
- Warning-sign questions come first, and a "yes" ends the flow at once. MediCare asked every
  question, and in some flows the red flag came last.
- "Don't know" to a warning-sign question raises the result to at least same-day review.
- The result is the most urgent level any rule reaches, not the first rule that matches.
- Coughing up blood defaults to same-day review; MediCare offered a routine appointment.
- Progress is stored server-side. MediCare rebuilt it by searching old messages for exact question
  text, so a reworded question, or an answer like "yes, since Monday", broke the flow.
- Replies are structured JSON rendered as text. MediCare returned raw `<button>` HTML inside
  replies and inserted it with innerHTML.

### How it fits in
- Engine: `services/care_bot_service.py`; shared text: `services/care_bot_content.py`.
- Widget: `templates/_care_bot.html` (included from `base.html`), `static/js/care_bot.js`,
  section 19 of `static/css/app.css`.
- Routes: `GET /care-bot/session`, `POST /care-bot/message`, `POST /care-bot/restart`,
  `POST /care-bot/handoff`. The POST routes accept only an `application/json` body, which a
  cross-site form cannot send - the app still has no CSRF tokens.
- Storage: `care_bot_sessions` and `care_bot_messages` tables; the Flask session holds only the
  chat id. "Start over" opens a new session and keeps the old transcript.
- Handoff writes one `patient_questions` row (assigned to the nurse) and a `care_events` entry per
  assessment. The note is in English, like the rest of CareAI's care events.
- Family view: the patient comes from the family login, never from the request. Patient portal:
  the page supplies the patient id - the same trust level as the portal's existing "Ask your
  nurse" form, because the portal still has no login.

### Setup note
`python-dotenv` is listed in requirements.txt but was missing from the venv. `app.py` loads `.env`
inside a try/except, so it failed silently and the OpenAI settings - for Hospital AI as well - never
took effect. It is now installed.

### Clinical status
The flows follow common public triage guidance but are NOT clinically validated. Research
prototype only; every outcome still needs a clinician.

## v12 - Care.AI Copilot, functional

The Care Copilot page (`/copilot`) now answers from the whole patient record instead of a handful
of keywords. Before this, 7 of 12 routine questions ("Does he have diabetes?", "Any recent
alerts?", "Who is his GP?", "Draft a handover note"...) got the same generic paragraph. "How has
his oxygen changed?" returned only the latest value, although a week of readings is stored. The
quick-prompt buttons only filled the text box, every question reloaded the page, and a patient
with no vitals crashed the breathing answer.

### How a question is answered
1. Safety first. A question describing an acute event ("he can't breathe", "has collapsed") gets
   the escalation message. The patterns are present tense only, so "has he had chest pain before?"
   is treated as a history question.
2. Record lookup. 13 intents are answered straight from SQLite, with values, units and
   timestamps:
   - status summary
   - 7-day trends
   - open alerts
   - conditions ("does he have diabetes?")
   - latest readings
   - GP and contacts
   - deterioration risk
   - score explanations
   - falls
   - breathing
   - medication and missed doses
   - open tasks
   - an SBAR handover note
3. Grounded AI. Two kinds of question go to OpenAI: anything lookup can't match, and questions
   that ask for interpretation ("could", "linked", "consistent with", "worried"). The model
   gets:
   - a snapshot of the record
   - the last six messages of the thread
   - up to two matching demo protocols from `data/clinical_protocols.json`

   It is told to answer only from that snapshot, to reply "Not recorded in Care.AI" otherwise,
   and not to diagnose or prescribe. The call has a 25-second timeout.
4. Fallback. With AI off, or if the call fails, the Copilot gives a record overview and lists
   what lookup covers.

Every answer is labelled with its source: "From the record", "AI answer · grounded in the
record" or "Safety rule". The label shows in the chat and is stored with the thread in a new
`chat_messages.source` column, added by a migration in `init_db`. The audit log records the
source with each question.

### What goes to OpenAI
The Copilot uses the same switch as Hospital AI and the chatbot: `CAREAI_USE_OPENAI=1`,
`OPENAI_API_KEY` and `OPENAI_MODEL`. Set `CAREAI_USE_OPENAI=0` to keep the Copilot on record
lookup only. The page header shows whether AI answers are on.

The snapshot carries:
- age, sex and living setting
- conditions and medication
- vitals and 7-day trends
- risk scores
- open tasks and alerts
- care events
- the latest fall assessment

It carries no name, birth date, city, GP or contacts. The patient is identified only by the
synthetic record reference. Names of patients, staff, family and GPs are replaced with "[name]"
in free-text notes and in the question.

### Page
- Questions are sent in place, without a page reload (`static/js/copilot.js`,
  `POST /copilot/<id>/ask`). The endpoint accepts only a JSON body, the same CSRF approach as the
  chatbot.
- Enter sends; Shift+Enter adds a line. The old form POST still works without JavaScript.
- Eight quick prompts send immediately: health summary, recent changes, open alerts, fall risk,
  breathing, medication, next action and handover note.
- New answers are rendered as text nodes, and Jinja escapes the stored history.

### Files
- `services/clinical_copilot.py` (new): record, intents, snapshot, AI call.
- `services/copilot_service.py`:
  - `clinician_answer` now delegates to the new module.
  - `save_message` takes a `source`.
  - Patient-portal and family answers no longer crash when a patient has no vitals.
- `app.py`, `templates/copilot.html`, `static/js/copilot.js`, section 20 of
  `static/css/app.css`, and the migration in `services/db.py`.

### Clinical status
Decision support on synthetic data. Lookups are exact, but they reflect only what CareAI stores.
Risk scores are rule-based. AI answers can still be wrong and are labelled as AI. A clinician
remains responsible.

## v13 - Care assistant: AI interviews for any health issue

The floating care assistant now handles any health issue, not only the 11 symptoms it has fixed
questions for. Those 11 keep their tested questions. For everything else OpenAI runs the
interview and picks each next question from the answers so far. At the end OpenAI analyses the
answers and writes the result, for both kinds of interview. This replaces v11's rule that OpenAI
never touches the guided flows; it still never decides an emergency.

### How a conversation goes
1. Every message is screened for emergency and crisis phrases first, as before.
2. A symptom one of the 11 flows covers (fever, cough, chest pain, breathlessness, falls...)
   starts that flow's fixed yes/no questions. A free-text reply such as "only when I climb the
   stairs" is now read by OpenAI as yes / no / don't know, instead of getting "Sorry, I didn't
   catch that".
3. Any other health issue ("my knee hurts", "ik slaap slecht", "mir ist übel") starts an OpenAI
   interview:
   - warning signs first, then onset, severity, pattern, other symptoms, what the person has
     tried, and relevant conditions and medicines;
   - one question at a time, usually 3-6 and never more than 8, with 2-5 answers to tap or
     room to type;
   - warning-sign questions always offer yes / no / don't know. The model states which answer
     means the warning sign is present, and how serious it is.
4. A general question ("Is it safe to drink alcohol with blood pressure tablets?") gets a
   direct answer. With AI on, a question that isn't about the person's own problem no longer
   opens a symptom checklist because of a keyword.

Each AI step takes about 3-8 seconds; the widget shows "Thinking…" meanwhile.

### The result card
- The urgency level and what to do: call 112, see a doctor today, book a GP appointment, or
  self-care.
- "What your answers suggest": 2-4 sentences on why.
- What you can do now, and the signs that mean getting help sooner.
- Up to 3 possible explanations, each marked more likely / possible / less likely, under the
  heading "not a diagnosis".
- The answers the result is based on, and a note that the analysis came from OpenAI.
- On the portal and family view, "Send summary to my care team" now also includes a short
  English note for the nurse, written by the AI.

### Safety: the rules outrank the model
- A warning answer ends the interview at once. For an emergency, the 112 advice appears
  without waiting for the model. For a question about self-harm, the crisis lines appear
  (112, 113 Zelfmoordpreventie, TelefonSeelsorge).
- The model must state which answer is the warning one, so "Can you keep fluids down?"
  escalates on "no". Missing or unclear metadata is read the cautious way: "yes" means same-day
  review.
- "Don't know" to a warning sign raises the result to at least same-day review.
- The final level is the most urgent of the rules and the model. The model can raise the level
  a rule-based flow gives, never lower it.
- If OpenAI fails mid-interview, or ignores the 8-question limit, the result is at least "book
  a GP appointment", with a note that the AI analysis wasn't available. At the end of a
  rule-based flow, the rules then decide alone.
- Replies must match a strict JSON schema and are then cleaned: unknown levels are rejected,
  lists are capped and text is trimmed. Everything is rendered as text.

### What goes to OpenAI
The same switch as before: `CAREAI_USE_OPENAI=1` with `OPENAI_API_KEY` and `OPENAI_MODEL`. Set
it to `0` and the bot works as in v11. It sends:
- what the person typed in this chat, with names known to CareAI replaced by "[name]";
- on the patient portal and family view only: age, sex, active conditions and medication.

It never sends a name, birth date, city, contact or record reference. Staff pages have no
patient, so no profile is sent from them.

### Files
- `services/care_bot_ai.py` (new): prompts, schemas, validation and the patient profile.
- `services/care_bot_service.py`: routing between the flows, the AI interview and direct
  answers; the shared result card; the handoff note.
- `services/care_bot_content.py`: new EN/NL/DE texts and likelihood labels.
- `static/js/care_bot.js` and section 19 of `static/css/app.css`: the new result sections.

### Clinical status
Research prototype. The AI questions and analysis are not clinically validated. The safety
rules above limit what the model can do, but it can still be wrong, and every result says it is
not a diagnosis.

## v14 - Downloadable PDF report for every AI analysis

Every AI diagnosis now has a "Download report (PDF)" button on its result: lung CT, chest X-ray
(pneumonia screening), Wound AI (wound type and the surgical infection-risk screen) and the
rule-based skin demo.

### What is in the PDF
- Header: model, modality, source image, run id, when it was analysed, when the report was
  generated and who downloaded it.
- The result: confidence or ensemble score, the prediction, the risk or review badge, the
  finding, and any low-confidence or screening warning.
- Class probabilities with bars - and for a surgical screen the two ensemble components with
  their weights and the reasons flagged for review.
- The uploaded image beside the Grad-CAM overlay, and for wounds the boundary overlay,
  measurements, the quality and wound checks that ran before the models, and the limitations.
- Recommended next steps, technical details (model versions, thresholds, device) and the
  prototype disclaimer.
- Blank lines for the case reference, the reviewing clinician and the date. An analysis is not
  linked to a patient, so the PDF carries no patient identity.

### How it works
- `services/diagnostic_report.py` builds the PDF with reportlab (added to requirements.txt).
- When an analysis finishes, the result the page rendered is written next to the uploaded image
  as `uploads/diagnostics/reports/<run id>.json`. `GET /ai-diagnostics/report/<run id>.pdf`
  rebuilds the PDF from that file, so the numbers in the report are exactly the ones on screen
  and the model never runs twice. Nothing is stored in the database and there is no report
  history page.
- The route needs a signed-in clinician, the run id must look like one (hex), and images are
  resolved by basename inside the uploads and Grad-CAM folders only.
- Images are embedded at print resolution (max 1300 px, JPEG). A chest X-ray straight from the
  scanner made a 4.7 MB report; it is now about 0.4 MB. Reports run 1-3 pages: about 120 KB for
  a CT, 400 KB for an X-ray, 15 KB for a wound.
- Every download is written to the audit log as `AI_REPORT_DOWNLOAD`.
- Text is mapped to what the PDF font can print, so an arrow in the interface text becomes "->"
  instead of a black box.
