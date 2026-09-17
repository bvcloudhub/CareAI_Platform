"""
Guided symptom flows for the Care.AI care chatbot.

The first six are ported from MediCare AI (Lung_Cancer_V3
services/clinical_chat.py FLOW_QUESTIONS) and translated to NL/DE. The
elderly-care flows live in care_bot_flows_elderly.py and are merged below.

Changes from the source, all for safety:
- Red-flag questions are asked FIRST, and a "yes" stops the flow at once.
  MediCare asked every question, and in some flows the red-flag one came last.
- Each disposition is the most urgent level any rule reaches, not the first
  rule to match.
- MediCare offered a non-urgent appointment for coughing up blood with no
  breathlessness; for an older adult that is a same-day GP matter, so the
  default here is "urgent".

Flow schema:
  title / topic   display names (topic is used mid-sentence)
  keywords        per-language trigger phrases, matched on word boundaries
  questions       id, text, short (summary label), stop_on_yes
  rules           {"any" | "all": [question ids], "level": LEVEL}
  default         level when no rule matches
  advice          per-language self-help lines shown in the summary
"""

from services.care_bot_flows_elderly import ELDERLY_FLOWS

FLOWS = {
    "fever": {
        "title": {"en": "Fever", "nl": "Koorts", "de": "Fieber"},
        "topic": {"en": "your fever", "nl": "uw koorts", "de": "Ihr Fieber"},
        "keywords": {
            "en": ["fever", "feverish", "high temperature", "temperature", "chills", "shivering"],
            "nl": ["koorts", "verhoging", "koude rillingen", "rillingen", "temperatuur"],
            "de": ["fieber", "erhöhte temperatur", "temperatur", "schüttelfrost"],
        },
        "questions": [
            {"id": "fever_red", "stop_on_yes": True,
             "text": {"en": "Right now, do you have any of these: severe headache with a stiff neck, new confusion, trouble breathing, chest pain, or a rash that does not fade when pressed?",
                      "nl": "Heeft u op dit moment een van deze klachten: hevige hoofdpijn met een stijve nek, nieuwe verwardheid, moeite met ademen, pijn op de borst, of huiduitslag die niet wegdrukbaar is?",
                      "de": "Haben Sie gerade eines davon: starke Kopfschmerzen mit steifem Nacken, neue Verwirrtheit, Atemnot, Brustschmerzen oder einen Ausschlag, der beim Draufdrücken nicht verblasst?"},
             "short": {"en": "Warning signs with fever", "nl": "Alarmsignalen bij koorts", "de": "Warnzeichen bei Fieber"}},
            {"id": "fever_high", "stop_on_yes": False,
             "text": {"en": "Is your measured temperature 38.0 °C or higher right now?",
                      "nl": "Is uw gemeten temperatuur nu 38,0 °C of hoger?",
                      "de": "Ist Ihre gemessene Temperatur jetzt 38,0 °C oder höher?"},
             "short": {"en": "Temperature 38.0 °C or higher", "nl": "Temperatuur 38,0 °C of hoger", "de": "Temperatur 38,0 °C oder höher"}},
            {"id": "fever_long", "stop_on_yes": False,
             "text": {"en": "Has the fever lasted 3 days or longer?",
                      "nl": "Duurt de koorts al 3 dagen of langer?",
                      "de": "Hält das Fieber schon 3 Tage oder länger an?"},
             "short": {"en": "Lasting 3 days or longer", "nl": "Al 3 dagen of langer", "de": "Seit 3 Tagen oder länger"}},
        ],
        "rules": [
            {"any": ["fever_red"], "level": "emergency"},
            {"all": ["fever_high", "fever_long"], "level": "urgent"},
            {"any": ["fever_high", "fever_long"], "level": "soon"},
        ],
        "default": "self_care",
        "advice": {
            "en": ["Drink plenty of fluids and rest.", "Paracetamol can ease fever and aches if it is safe for you.",
                   "Check your temperature again in a few hours."],
            "nl": ["Drink voldoende en neem rust.", "Paracetamol kan koorts en pijn verlichten als dat voor u veilig is.",
                   "Meet uw temperatuur over een paar uur opnieuw."],
            "de": ["Trinken Sie ausreichend und ruhen Sie sich aus.",
                   "Paracetamol kann Fieber und Schmerzen lindern, wenn es für Sie geeignet ist.",
                   "Messen Sie Ihre Temperatur in einigen Stunden erneut."],
        },
    },

    "cough": {
        "title": {"en": "Cough", "nl": "Hoesten", "de": "Husten"},
        "topic": {"en": "your cough", "nl": "uw hoest", "de": "Ihren Husten"},
        "keywords": {
            "en": ["cough", "coughing", "phlegm", "sputum", "mucus", "a cold", "common cold"],
            "nl": ["hoest", "hoesten", "slijm", "verkouden", "verkoudheid"],
            "de": ["husten", "huste", "schleim", "auswurf", "erkältet", "erkältung"],
        },
        "questions": [
            {"id": "cough_red", "stop_on_yes": True,
             "text": {"en": "Right now, are you coughing up blood, very short of breath, or having chest pain?",
                      "nl": "Hoest u op dit moment bloed op, bent u erg kortademig of heeft u pijn op de borst?",
                      "de": "Husten Sie gerade Blut, sind Sie sehr kurzatmig oder haben Sie Brustschmerzen?"},
             "short": {"en": "Blood, severe breathlessness or chest pain", "nl": "Bloed, ernstige benauwdheid of borstpijn",
                       "de": "Blut, starke Atemnot oder Brustschmerzen"}},
            {"id": "cough_fever", "stop_on_yes": False,
             "text": {"en": "Do you have a fever of 38.0 °C or higher?",
                      "nl": "Heeft u koorts van 38,0 °C of hoger?",
                      "de": "Haben Sie Fieber von 38,0 °C oder mehr?"},
             "short": {"en": "Fever 38.0 °C or higher", "nl": "Koorts 38,0 °C of hoger", "de": "Fieber 38,0 °C oder mehr"}},
            {"id": "cough_long", "stop_on_yes": False,
             "text": {"en": "Has the cough lasted 3 weeks or longer?",
                      "nl": "Hoest u al 3 weken of langer?",
                      "de": "Husten Sie schon 3 Wochen oder länger?"},
             "short": {"en": "Lasting 3 weeks or longer", "nl": "Al 3 weken of langer", "de": "Seit 3 Wochen oder länger"}},
            {"id": "cough_weight", "stop_on_yes": False,
             "text": {"en": "Have you had unexplained weight loss or night sweats?",
                      "nl": "Bent u onbedoeld afgevallen of heeft u nachtzweten?",
                      "de": "Haben Sie ungewollt abgenommen oder Nachtschweiß?"},
             "short": {"en": "Weight loss or night sweats", "nl": "Afvallen of nachtzweten", "de": "Gewichtsverlust oder Nachtschweiß"}},
        ],
        "rules": [
            {"any": ["cough_red"], "level": "emergency"},
            {"any": ["cough_fever"], "level": "urgent"},
            {"any": ["cough_long", "cough_weight"], "level": "soon"},
        ],
        "default": "self_care",
        "advice": {
            "en": ["Drink warm fluids; honey can soothe a cough.", "Avoid smoking and smoky rooms.",
                   "If you have a reliever inhaler, use it as prescribed."],
            "nl": ["Drink warme dranken; honing kan hoest verzachten.", "Vermijd roken en rokerige ruimtes.",
                   "Gebruik uw luchtwegverwijder zoals voorgeschreven, als u die heeft."],
            "de": ["Trinken Sie warme Getränke; Honig kann den Husten lindern.", "Meiden Sie Rauch und verrauchte Räume.",
                   "Wenn Sie ein Bedarfsspray haben, verwenden Sie es wie verordnet."],
        },
    },

    "hemoptysis": {
        "title": {"en": "Coughing up blood", "nl": "Bloed ophoesten", "de": "Bluthusten"},
        "topic": {"en": "coughing up blood", "nl": "het ophoesten van bloed", "de": "das Bluthusten"},
        "keywords": {
            "en": ["coughing blood", "coughing up blood", "cough up blood", "coughed up blood", "blood in my sputum",
                   "blood in sputum", "blood in phlegm", "blood in my phlegm", "hemoptysis", "haemoptysis"],
            "nl": ["bloed ophoesten", "bloed opgehoest", "bloed in slijm", "bloed in het slijm", "bloed bij het hoesten"],
            "de": ["blut husten", "bluthusten", "blut abhusten", "blut im auswurf", "blut im schleim"],
        },
        "questions": [
            {"id": "hemo_breath", "stop_on_yes": True,
             "text": {"en": "Are you short of breath or having chest pain right now?",
                      "nl": "Bent u op dit moment kortademig of heeft u pijn op de borst?",
                      "de": "Sind Sie gerade kurzatmig oder haben Sie Brustschmerzen?"},
             "short": {"en": "Breathless or chest pain now", "nl": "Nu kortademig of borstpijn", "de": "Jetzt Atemnot oder Brustschmerzen"}},
            {"id": "hemo_amount", "stop_on_yes": True,
             "text": {"en": "Is it more than a teaspoon of blood at a time?",
                      "nl": "Is het meer dan een theelepel bloed per keer?",
                      "de": "Ist es mehr als ein Teelöffel Blut auf einmal?"},
             "short": {"en": "More than a teaspoon at a time", "nl": "Meer dan een theelepel per keer", "de": "Mehr als ein Teelöffel auf einmal"}},
            {"id": "hemo_thinners", "stop_on_yes": False,
             "text": {"en": "Do you take blood thinners (for example warfarin, apixaban or rivaroxaban)?",
                      "nl": "Gebruikt u bloedverdunners (bijvoorbeeld acenocoumarol, apixaban of rivaroxaban)?",
                      "de": "Nehmen Sie Blutverdünner (zum Beispiel Marcumar, Apixaban oder Rivaroxaban)?"},
             "short": {"en": "Takes blood thinners", "nl": "Gebruikt bloedverdunners", "de": "Nimmt Blutverdünner"}},
            {"id": "hemo_long", "stop_on_yes": False,
             "text": {"en": "Has it gone on for 3 days or longer?",
                      "nl": "Duurt het al 3 dagen of langer?",
                      "de": "Hält es schon 3 Tage oder länger an?"},
             "short": {"en": "Lasting 3 days or longer", "nl": "Al 3 dagen of langer", "de": "Seit 3 Tagen oder länger"}},
        ],
        "rules": [
            {"any": ["hemo_breath", "hemo_amount"], "level": "emergency"},
        ],
        "default": "urgent",
        "advice": {
            "en": ["Sit upright and stay calm.", "Note how often it happens and roughly how much.",
                   "Do not stop prescribed medicines without talking to your doctor."],
            "nl": ["Ga rechtop zitten en blijf rustig.", "Noteer hoe vaak het gebeurt en ongeveer hoeveel.",
                   "Stop niet op eigen houtje met voorgeschreven medicijnen."],
            "de": ["Setzen Sie sich aufrecht hin und bleiben Sie ruhig.",
                   "Notieren Sie, wie oft es auftritt und ungefähr wie viel.",
                   "Setzen Sie verordnete Medikamente nicht eigenmächtig ab."],
        },
    },

    "chest_pain": {
        "title": {"en": "Chest pain", "nl": "Pijn op de borst", "de": "Brustschmerzen"},
        "topic": {"en": "your chest pain", "nl": "uw pijn op de borst", "de": "Ihre Brustschmerzen"},
        "keywords": {
            "en": ["chest pain", "chest pressure", "chest tightness", "tight chest", "pain in my chest", "pressure in my chest"],
            "nl": ["pijn op de borst", "borstpijn", "druk op de borst", "beklemmend gevoel", "benauwd op de borst"],
            "de": ["brustschmerz", "brustschmerzen", "druck auf der brust", "engegefühl", "schmerzen in der brust"],
        },
        "questions": [
            {"id": "cp_red", "stop_on_yes": True,
             "text": {"en": "Is the pain severe or pressing, does it spread to your arm, jaw or back — or do you feel short of breath, sweaty, dizzy or faint?",
                      "nl": "Is de pijn hevig of drukkend, straalt die uit naar arm, kaak of rug, of bent u kortademig, zweterig, duizelig of voelt u zich flauw?",
                      "de": "Ist der Schmerz stark oder drückend, strahlt er in Arm, Kiefer oder Rücken aus, oder sind Sie kurzatmig, verschwitzt, schwindlig oder einer Ohnmacht nahe?"},
             "short": {"en": "Severe, spreading, or with breathlessness/sweating", "nl": "Hevig, uitstralend, of met benauwdheid/zweten",
                       "de": "Stark, ausstrahlend oder mit Atemnot/Schwitzen"}},
            {"id": "cp_rest", "stop_on_yes": True,
             "text": {"en": "Has the pain lasted more than 15 minutes, or does it come on at rest?",
                      "nl": "Duurt de pijn langer dan 15 minuten, of ontstaat die in rust?",
                      "de": "Hält der Schmerz länger als 15 Minuten an oder tritt er in Ruhe auf?"},
             "short": {"en": "Over 15 minutes or at rest", "nl": "Langer dan 15 minuten of in rust", "de": "Über 15 Minuten oder in Ruhe"}},
            {"id": "cp_exertion", "stop_on_yes": False,
             "text": {"en": "Does it come on with exertion (walking, stairs) and ease with rest?",
                      "nl": "Ontstaat de pijn bij inspanning (lopen, traplopen) en zakt die in rust?",
                      "de": "Tritt der Schmerz bei Belastung auf (Gehen, Treppensteigen) und lässt er in Ruhe nach?"},
             "short": {"en": "Comes on with exertion", "nl": "Bij inspanning", "de": "Bei Belastung"}},
            {"id": "cp_breath", "stop_on_yes": False,
             "text": {"en": "Does it get worse when you breathe in deeply or press on your chest?",
                      "nl": "Wordt het erger bij diep inademen of als u op uw borst drukt?",
                      "de": "Wird es schlimmer, wenn Sie tief einatmen oder auf die Brust drücken?"},
             "short": {"en": "Worse on deep breath or pressure", "nl": "Erger bij diep inademen of drukken", "de": "Schlimmer beim Einatmen oder Drücken"}},
        ],
        "rules": [
            {"any": ["cp_red", "cp_rest"], "level": "emergency"},
            {"any": ["cp_exertion"], "level": "urgent"},
        ],
        "default": "soon",
        "advice": {
            "en": ["Stop what you are doing and rest.",
                   "If you have been prescribed a nitroglycerin spray, use it as instructed.",
                   "Do not drive yourself if the pain returns."],
            "nl": ["Stop met wat u doet en neem rust.", "Gebruik uw nitroglycerinespray zoals voorgeschreven, als u die heeft.",
                   "Rijd niet zelf als de pijn terugkomt."],
            "de": ["Unterbrechen Sie, was Sie tun, und ruhen Sie sich aus.",
                   "Wenn Ihnen ein Nitrospray verordnet wurde, wenden Sie es wie angewiesen an.",
                   "Fahren Sie nicht selbst Auto, wenn der Schmerz wiederkommt."],
        },
    },

    "headache": {
        "title": {"en": "Headache", "nl": "Hoofdpijn", "de": "Kopfschmerzen"},
        "topic": {"en": "your headache", "nl": "uw hoofdpijn", "de": "Ihre Kopfschmerzen"},
        "keywords": {
            "en": ["headache", "migraine", "head pain", "head hurts"],
            "nl": ["hoofdpijn", "migraine"],
            "de": ["kopfschmerz", "kopfschmerzen", "migräne"],
        },
        "questions": [
            {"id": "ha_red", "stop_on_yes": True,
             "text": {"en": "Did it start suddenly like a thunderclap, or is it the worst headache of your life — or do you have a stiff neck with fever, new weakness, slurred speech or loss of vision?",
                      "nl": "Begon de hoofdpijn plotseling als een donderslag, is het de ergste hoofdpijn ooit, of heeft u een stijve nek met koorts, nieuw krachtsverlies, onduidelijke spraak of verlies van zicht?",
                      "de": "Kam der Kopfschmerz plötzlich wie ein Donnerschlag, ist er der schlimmste Ihres Lebens, oder haben Sie einen steifen Nacken mit Fieber, neue Schwäche, verwaschene Sprache oder Sehverlust?"},
             "short": {"en": "Thunderclap, worst ever, or neurological signs", "nl": "Donderslag, ergste ooit, of neurologische klachten",
                       "de": "Donnerschlag, schlimmster je, oder neurologische Zeichen"}},
            {"id": "ha_injury", "stop_on_yes": False,
             "text": {"en": "Did you hit your head recently?",
                      "nl": "Heeft u recent uw hoofd gestoten?",
                      "de": "Haben Sie sich kürzlich den Kopf gestoßen?"},
             "short": {"en": "Recent head injury", "nl": "Recent hoofd gestoten", "de": "Kürzlich Kopfverletzung"}},
            {"id": "ha_severe", "stop_on_yes": False,
             "text": {"en": "Is it severe (7 out of 10 or more) or getting rapidly worse?",
                      "nl": "Is de hoofdpijn hevig (7 op 10 of meer) of wordt die snel erger?",
                      "de": "Ist der Kopfschmerz stark (7 von 10 oder mehr) oder wird er rasch schlimmer?"},
             "short": {"en": "Severe or rapidly worsening", "nl": "Hevig of snel erger", "de": "Stark oder rasch schlimmer"}},
            {"id": "ha_long", "stop_on_yes": False,
             "text": {"en": "Has it lasted 3 days or longer?",
                      "nl": "Duurt de hoofdpijn al 3 dagen of langer?",
                      "de": "Hält der Kopfschmerz schon 3 Tage oder länger an?"},
             "short": {"en": "Lasting 3 days or longer", "nl": "Al 3 dagen of langer", "de": "Seit 3 Tagen oder länger"}},
        ],
        "rules": [
            {"any": ["ha_red"], "level": "emergency"},
            {"any": ["ha_injury", "ha_severe"], "level": "urgent"},
            {"any": ["ha_long"], "level": "soon"},
        ],
        "default": "self_care",
        "advice": {
            "en": ["Rest in a quiet, dim room and drink water.", "Paracetamol can help if it is safe for you.",
                   "Keep a note of what seems to trigger it."],
            "nl": ["Rust in een rustige, donkere kamer en drink water.", "Paracetamol kan helpen als dat voor u veilig is.",
                   "Houd bij wat de hoofdpijn lijkt uit te lokken."],
            "de": ["Ruhen Sie in einem ruhigen, abgedunkelten Raum und trinken Sie Wasser.",
                   "Paracetamol kann helfen, wenn es für Sie geeignet ist.",
                   "Notieren Sie, was die Kopfschmerzen auszulösen scheint."],
        },
    },

    "body_pain": {
        "title": {"en": "Body pain", "nl": "Pijn in het lichaam", "de": "Gliederschmerzen"},
        "topic": {"en": "your pain", "nl": "uw pijn", "de": "Ihre Schmerzen"},
        "keywords": {
            "en": ["body pain", "body ache", "body aches", "aching all over", "muscle pain", "joint pain", "back pain", "myalgia"],
            "nl": ["spierpijn", "gewrichtspijn", "rugpijn", "pijn in mijn lichaam", "overal pijn"],
            "de": ["gliederschmerzen", "muskelschmerzen", "gelenkschmerzen", "rückenschmerzen", "körperschmerzen"],
        },
        "questions": [
            {"id": "bp_neuro", "stop_on_yes": True,
             "text": {"en": "Do you have sudden weakness or numbness in an arm or leg?",
                      "nl": "Heeft u plotseling krachtsverlies of een doof gevoel in een arm of been?",
                      "de": "Haben Sie plötzliche Schwäche oder Taubheit in einem Arm oder Bein?"},
             "short": {"en": "Sudden weakness or numbness", "nl": "Plotseling krachtsverlies of doof gevoel", "de": "Plötzliche Schwäche oder Taubheit"}},
            {"id": "bp_hot", "stop_on_yes": False,
             "text": {"en": "Is a joint or calf hot, red and swollen — or did the pain start after a fall or injury?",
                      "nl": "Is een gewricht of kuit warm, rood en gezwollen — of begon de pijn na een val of ongeluk?",
                      "de": "Ist ein Gelenk oder eine Wade heiß, gerötet und geschwollen — oder begannen die Schmerzen nach einem Sturz oder Unfall?"},
             "short": {"en": "Hot swollen joint/calf, or after a fall", "nl": "Warm gezwollen gewricht/kuit, of na een val",
                       "de": "Heißes geschwollenes Gelenk/Wade oder nach Sturz"}},
            {"id": "bp_severe", "stop_on_yes": False,
             "text": {"en": "Is the pain severe (7 out of 10 or more) or stopping your daily activities?",
                      "nl": "Is de pijn hevig (7 op 10 of meer) of houdt die u tegen bij dagelijkse bezigheden?",
                      "de": "Ist der Schmerz stark (7 von 10 oder mehr) oder hindert er Sie an Ihren täglichen Aktivitäten?"},
             "short": {"en": "Severe or limiting daily life", "nl": "Hevig of beperkend", "de": "Stark oder einschränkend"}},
            {"id": "bp_long", "stop_on_yes": False,
             "text": {"en": "Has the pain lasted 7 days or longer?",
                      "nl": "Heeft u de pijn al 7 dagen of langer?",
                      "de": "Haben Sie die Schmerzen schon 7 Tage oder länger?"},
             "short": {"en": "Lasting 7 days or longer", "nl": "Al 7 dagen of langer", "de": "Seit 7 Tagen oder länger"}},
        ],
        "rules": [
            {"any": ["bp_neuro"], "level": "emergency"},
            {"any": ["bp_hot"], "level": "urgent"},
            {"any": ["bp_severe", "bp_long"], "level": "soon"},
        ],
        "default": "self_care",
        "advice": {
            "en": ["Keep gently moving as far as the pain allows.", "Warmth or a cold pack can ease sore muscles.",
                   "Paracetamol can help if it is safe for you."],
            "nl": ["Blijf rustig bewegen zover de pijn het toelaat.", "Warmte of een koud kompres kan pijnlijke spieren verlichten.",
                   "Paracetamol kan helpen als dat voor u veilig is."],
            "de": ["Bewegen Sie sich sanft, soweit es die Schmerzen zulassen.",
                   "Wärme oder eine Kühlpackung kann schmerzende Muskeln lindern.",
                   "Paracetamol kann helfen, wenn es für Sie geeignet ist."],
        },
    },
}

FLOWS.update(ELDERLY_FLOWS)

# When a message mentions several symptoms, the most dangerous flow wins —
# "a cough and I'm short of breath" must not be triaged as a cough.
PRIORITY = [
    "hemoptysis", "chest_pain", "breathlessness", "confusion", "fall_dizziness",
    "leg_swelling", "urinary", "fever", "headache", "cough", "body_pain",
]
assert set(PRIORITY) == set(FLOWS), "PRIORITY must list every flow exactly once"
