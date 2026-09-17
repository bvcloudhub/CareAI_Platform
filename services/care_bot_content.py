"""
Shared multilingual content for the Care.AI care chatbot (EN / NL / DE).

The symptom flows themselves live in care_bot_flows.py and
care_bot_flows_elderly.py; this module holds what every flow shares: triage
levels, the emergency phrases screened on every message, yes/no vocabulary and
the bot's own framing messages.
"""

LANGS = ("en", "nl", "de")

# Most urgent first. Each flow's rules resolve to one of these.
LEVELS = {
    "emergency": {
        "rank": 3,
        "label": {"en": "Emergency — act now", "nl": "Spoed — nu handelen", "de": "Notfall — jetzt handeln"},
        "action": {
            "en": "Call your emergency number (112) now. Do not wait for a chat reply and do not drive yourself.",
            "nl": "Bel nu 112. Wacht niet op een chatbericht en rijd niet zelf.",
            "de": "Rufen Sie jetzt den Notruf 112 an. Warten Sie nicht auf eine Chat-Antwort und fahren Sie nicht selbst.",
        },
    },
    "urgent": {
        "rank": 2,
        "label": {"en": "See a doctor today", "nl": "Vandaag contact met de huisarts", "de": "Heute ärztlich abklären"},
        "action": {
            "en": "Contact your GP practice today. Out of hours, call the GP out-of-hours service.",
            "nl": "Bel vandaag uw huisarts. Buiten kantoortijden belt u de huisartsenpost.",
            "de": "Kontaktieren Sie heute Ihre Hausarztpraxis. Außerhalb der Sprechzeiten den ärztlichen Bereitschaftsdienst (116117).",
        },
    },
    "soon": {
        "rank": 1,
        "label": {"en": "Book a GP appointment", "nl": "Afspraak bij de huisarts", "de": "Hausarzttermin vereinbaren"},
        "action": {
            "en": "Make an appointment with your GP within the next few days.",
            "nl": "Maak binnen enkele dagen een afspraak bij uw huisarts.",
            "de": "Vereinbaren Sie in den nächsten Tagen einen Termin in Ihrer Hausarztpraxis.",
        },
    },
    "self_care": {
        "rank": 0,
        "label": {"en": "Self-care and keep an eye on it", "nl": "Zelfzorg en in de gaten houden",
                  "de": "Selbst versorgen und beobachten"},
        "action": {
            "en": "You can likely manage this at home. Contact your care team if it gets worse or does not improve.",
            "nl": "U kunt dit waarschijnlijk thuis opvangen. Neem contact op met uw zorgteam als het erger wordt of niet verbetert.",
            "de": "Das können Sie wahrscheinlich zu Hause versorgen. Wenden Sie sich an Ihr Pflegeteam, wenn es schlimmer oder nicht besser wird.",
        },
    },
}

# Screened on EVERY message, before any flow or AI call. Substring match on
# lower-cased text, so keep each phrase specific enough not to misfire.
EMERGENCY_PHRASES = {
    "en": ["can't breathe", "cannot breathe", "can not breathe", "not breathing", "stopped breathing",
           "unconscious", "not responding", "unresponsive", "won't wake", "crushing chest pain",
           "severe chest pain", "stroke", "face drooping", "face droop", "slurred speech",
           "severe bleeding", "bleeding heavily", "blue lips", "choking", "seizure", "having a fit",
           "coughing up a lot of blood", "anaphylaxis", "throat swelling"],
    "nl": ["kan niet ademen", "krijg geen lucht", "ademt niet", "bewusteloos", "reageert niet",
           "niet wakker te krijgen", "hevige pijn op de borst", "beroerte", "scheve mond",
           "onduidelijke spraak", "hevige bloeding", "blauwe lippen", "stikt", "stikken",
           "epileptische aanval", "stuipen", "veel bloed ophoesten", "keel zwelt"],
    "de": ["kann nicht atmen", "bekomme keine luft", "atmet nicht", "bewusstlos", "reagiert nicht",
           "nicht wach zu bekommen", "starke brustschmerzen", "schlaganfall", "hängender mundwinkel",
           "verwaschene sprache", "starke blutung", "blaue lippen", "erstickt", "ersticken",
           "krampfanfall", "viel blut husten", "hals schwillt"],
}

CRISIS_PHRASES = {
    "en": ["suicide", "kill myself", "end my life", "want to die", "overdose", "self-harm", "hurt myself"],
    "nl": ["zelfmoord", "suïcide", "mezelf van kant", "wil dood", "overdosis", "mezelf pijn doen"],
    "de": ["suizid", "selbstmord", "mich umbringen", "will sterben", "überdosis", "mir etwas antun"],
}

MESSAGES = {
    "emergency": {
        "en": "What you describe may be an emergency. Call 112 now, or ask someone nearby to call. Do not wait for this chat.",
        "nl": "Wat u beschrijft kan een noodgeval zijn. Bel nu 112, of vraag iemand in de buurt om te bellen. Wacht niet op deze chat.",
        "de": "Was Sie beschreiben, kann ein Notfall sein. Rufen Sie jetzt 112 an oder bitten Sie jemanden in der Nähe darum. Warten Sie nicht auf diesen Chat.",
    },
    "crisis": {
        "en": "I'm really glad you told me. You deserve support right now. If you are in danger, call 112. "
              "You can also talk to someone now: in the Netherlands 113 Zelfmoordpreventie (0800-0113), "
              "in Germany TelefonSeelsorge (0800 111 0 111).",
        "nl": "Fijn dat u het mij vertelt. U verdient nu steun. Bent u in gevaar, bel dan 112. "
              "U kunt ook nu met iemand praten: 113 Zelfmoordpreventie, gratis via 0800-0113.",
        "de": "Danke, dass Sie es mir sagen. Sie verdienen jetzt Unterstützung. Wenn Sie in Gefahr sind, rufen Sie 112 an. "
              "Sie können auch sofort mit jemandem sprechen: TelefonSeelsorge, kostenlos unter 0800 111 0 111.",
    },
    "greeting": {
        "en": "Hello, I'm the Care.AI care assistant. Tell me what's bothering you — for example "
              "\"I have a cough\" or \"I feel dizzy\" — and I'll ask a few questions to help you decide what to do next.",
        "nl": "Hallo, ik ben de Care.AI-zorgassistent. Vertel me wat u dwarszit — bijvoorbeeld \"ik hoest\" "
              "of \"ik ben duizelig\" — en ik stel een paar vragen om te helpen bepalen wat u het beste kunt doen.",
        "de": "Hallo, ich bin der Care.AI-Pflegeassistent. Erzählen Sie mir, was Sie beschäftigt — zum Beispiel "
              "\"ich huste\" oder \"mir ist schwindlig\" — und ich stelle ein paar Fragen, damit Sie wissen, was als Nächstes zu tun ist.",
    },
    "greeting_ai": {
        "en": "Hello, I'm the Care.AI care assistant. Tell me what's bothering you — for example \"I have a cough\", "
              "\"my knee hurts\" or \"I sleep badly\" — or ask me a health question. I'll ask a few questions and "
              "then tell you what I think you should do next.",
        "nl": "Hallo, ik ben de Care.AI-zorgassistent. Vertel me wat u dwarszit — bijvoorbeeld \"ik hoest\", "
              "\"mijn knie doet pijn\" of \"ik slaap slecht\" — of stel een gezondheidsvraag. Ik stel een paar vragen "
              "en vertel u daarna wat u volgens mij het beste kunt doen.",
        "de": "Hallo, ich bin der Care.AI-Pflegeassistent. Erzählen Sie mir, was Sie beschäftigt — zum Beispiel "
              "\"ich huste\", \"mein Knie tut weh\" oder \"ich schlafe schlecht\" — oder stellen Sie eine "
              "Gesundheitsfrage. Ich stelle ein paar Fragen und sage Ihnen dann, was Sie als Nächstes tun sollten.",
    },
    "flow_intro": {
        "en": "I'll ask up to {n} short questions about {topic}. Answer yes or no.",
        "nl": "Ik stel maximaal {n} korte vragen over {topic}. Antwoord met ja of nee.",
        "de": "Ich stelle bis zu {n} kurze Fragen zu {topic}. Antworten Sie mit Ja oder Nein.",
    },
    "question_counter": {"en": "Question {i} of {n}", "nl": "Vraag {i} van {n}", "de": "Frage {i} von {n}"},
    "reprompt": {
        "en": "Sorry, I didn't catch that. Please answer yes or no — or say \"don't know\".",
        "nl": "Sorry, dat begreep ik niet. Antwoord met ja of nee — of zeg \"weet ik niet\".",
        "de": "Entschuldigung, das habe ich nicht verstanden. Bitte antworten Sie mit Ja oder Nein — oder sagen Sie \"weiß nicht\".",
    },
    "stopped_early": {
        "en": "Based on that answer I'll stop the questions here.",
        "nl": "Op basis van dat antwoord stop ik hier met de vragen.",
        "de": "Aufgrund dieser Antwort beende ich die Fragen hier.",
    },
    "summary_title": {"en": "Assessment — {topic}", "nl": "Beoordeling — {topic}", "de": "Einschätzung — {topic}"},
    "your_answers": {"en": "Your answers", "nl": "Uw antwoorden", "de": "Ihre Antworten"},
    "what_you_can_do": {"en": "What you can do now", "nl": "Wat u nu kunt doen", "de": "Was Sie jetzt tun können"},
    "disclaimer": {
        "en": "This is guidance from a research prototype, not a diagnosis. If things get worse, contact your care team or call 112.",
        "nl": "Dit is advies van een onderzoeksprototype, geen diagnose. Wordt het erger, neem dan contact op met uw zorgteam of bel 112.",
        "de": "Dies ist eine Orientierung aus einem Forschungsprototyp, keine Diagnose. Wenn es schlimmer wird, wenden Sie sich an Ihr Pflegeteam oder rufen Sie 112 an.",
    },
    "fallback": {
        "en": "I can guide you through symptoms such as fever, cough, coughing up blood, chest pain, headache, "
              "body pain, breathlessness, a fall or dizziness, sudden confusion, urinary problems and swollen legs. "
              "Tell me which one — or ask your care team directly.",
        "nl": "Ik kan u helpen bij klachten zoals koorts, hoesten, bloed ophoesten, pijn op de borst, hoofdpijn, "
              "pijn in het lichaam, kortademigheid, een val of duizeligheid, plotselinge verwardheid, plasklachten en dikke benen. "
              "Vertel me welke — of stel uw vraag direct aan uw zorgteam.",
        "de": "Ich kann Sie bei Beschwerden wie Fieber, Husten, Bluthusten, Brustschmerzen, Kopfschmerzen, "
              "Gliederschmerzen, Atemnot, Sturz oder Schwindel, plötzlicher Verwirrtheit, Blasenbeschwerden und geschwollenen Beinen begleiten. "
              "Sagen Sie mir, welche — oder fragen Sie direkt Ihr Pflegeteam.",
    },
    "ai_notice": {
        "en": "Answered by an external AI service (OpenAI) — general information only, not a diagnosis.",
        "nl": "Beantwoord door een externe AI-dienst (OpenAI) — alleen algemene informatie, geen diagnose.",
        "de": "Beantwortet von einem externen KI-Dienst (OpenAI) — nur allgemeine Informationen, keine Diagnose.",
    },
    "ai_intro": {
        "en": "I'll ask a few questions to understand this better. Tap an answer or type your own.",
        "nl": "Ik stel een paar vragen om dit beter te begrijpen. Tik op een antwoord of typ uw eigen antwoord.",
        "de": "Ich stelle ein paar Fragen, um das besser zu verstehen. Tippen Sie auf eine Antwort oder schreiben Sie selbst.",
    },
    "ai_counter": {"en": "Question {i}", "nl": "Vraag {i}", "de": "Frage {i}"},
    "ai_question_notice": {
        "en": "These questions are chosen by an external AI service (OpenAI).",
        "nl": "Deze vragen worden gekozen door een externe AI-dienst (OpenAI).",
        "de": "Diese Fragen wählt ein externer KI-Dienst (OpenAI) aus.",
    },
    "ai_analysis_notice": {
        "en": "Analysis written by an external AI service (OpenAI) from your answers — not a diagnosis.",
        "nl": "Analyse door een externe AI-dienst (OpenAI) op basis van uw antwoorden — geen diagnose.",
        "de": "Analyse eines externen KI-Dienstes (OpenAI) auf Grundlage Ihrer Antworten — keine Diagnose.",
    },
    "ai_unavailable": {
        "en": "The AI analysis isn't available right now, so this advice comes from the safety rules only. "
              "If you're unsure, contact your GP.",
        "nl": "De AI-analyse is nu niet beschikbaar, dus dit advies komt alleen uit de veiligheidsregels. "
              "Twijfelt u, neem dan contact op met uw huisarts.",
        "de": "Die KI-Analyse ist gerade nicht verfügbar, daher beruht dieser Rat nur auf den Sicherheitsregeln. "
              "Wenn Sie unsicher sind, wenden Sie sich an Ihre Hausarztpraxis.",
    },
    "analysis_title": {"en": "What your answers suggest", "nl": "Wat uw antwoorden laten zien",
                       "de": "Was Ihre Antworten nahelegen"},
    "causes_title": {"en": "Possible explanations — not a diagnosis", "nl": "Mogelijke verklaringen — geen diagnose",
                     "de": "Mögliche Erklärungen — keine Diagnose"},
    "watch_title": {"en": "Get help sooner if", "nl": "Neem eerder contact op als", "de": "Holen Sie früher Hilfe, wenn"},
    "handoff_done": {
        "en": "Sent to {name}'s care team. A nurse will see it in the clinician inbox.",
        "nl": "Verstuurd naar het zorgteam van {name}. Een verpleegkundige ziet het in de inbox.",
        "de": "An das Pflegeteam von {name} gesendet. Eine Pflegekraft sieht es im Posteingang.",
    },
    "reset": {
        "en": "Okay, let's start again. What's bothering you?",
        "nl": "Goed, we beginnen opnieuw. Wat zit u dwars?",
        "de": "Gut, fangen wir neu an. Was beschäftigt Sie?",
    },
}

ANSWER_LABELS = {
    "yes": {"en": "Yes", "nl": "Ja", "de": "Ja"},
    "no": {"en": "No", "nl": "Nee", "de": "Nein"},
    "unknown": {"en": "Don't know", "nl": "Weet ik niet", "de": "Weiß nicht"},
}

# How likely an AI-suggested explanation is, as shown on the result card.
LIKELIHOOD = {
    "more_likely": {"en": "More likely", "nl": "Waarschijnlijker", "de": "Wahrscheinlicher"},
    "possible": {"en": "Possible", "nl": "Mogelijk", "de": "Möglich"},
    "less_likely": {"en": "Less likely", "nl": "Minder waarschijnlijk", "de": "Weniger wahrscheinlich"},
}

# Whole-word matching after lower-casing. "Don't know" is checked first so
# "I don't know" is not read as "no".
UNKNOWN_WORDS = ["don't know", "dont know", "not sure", "unsure", "no idea", "weet ik niet", "weet niet",
                 "geen idee", "weiß nicht", "weiss nicht", "keine ahnung", "unsicher"]
YES_WORDS = ["yes", "y", "yeah", "yep", "yup", "correct", "sure", "true",
             "ja", "jazeker", "jawel", "klopt", "zeker", "jawohl", "genau", "stimmt", "doch"]
NO_WORDS = ["no", "n", "nope", "not really", "none", "never", "i don't", "i do not", "i'm not",
            "nee", "neen", "niet", "nooit", "nein", "nicht", "nie", "keine", "kein"]


def pick(texts, lang):
    """Return the text for `lang`, falling back to English."""
    return texts.get(lang) or texts.get("en", "")
