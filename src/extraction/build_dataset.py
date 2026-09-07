import os
import json
import ast
import re

import pandas as pd


# ============================================================
# BLUEHYREAI - EXTRACTION DATASET BUILDER V5.1
# ============================================================
#
# PURPOSE
# -------
# Correct behavioral field assignments using QUESTION + ANSWER
# context while preserving the V5 hierarchical schema.
#
# V5:
#   V4 flat fields -> V5 hierarchy
#
# V5.1:
#   V5 hierarchy + contextual behavioral relabeling
#
# IMPORTANT
# ---------
# - No model/API/GPU.
# - Preserves V5 train/validation/test membership.
# - Does not touch previous datasets or runs.
# - other_explicit_facts always remains a dictionary.
#
# ============================================================


BASE_DIR = r"F:\BlueHyre Project Final\03_Extraction"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "01_extraction_v5"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Results",
    "01_extraction_v5_1"
)


TRAIN_FILE = os.path.join(
    INPUT_DIR,
    "train.csv"
)

VAL_FILE = os.path.join(
    INPUT_DIR,
    "validation.csv"
)

TEST_FILE = os.path.join(
    INPUT_DIR,
    "test.csv"
)


# ============================================================
# HIERARCHICAL SCHEMA
# ============================================================

TOP_LEVEL_FIELDS = {
    "profile",
    "behavioral",
    "availability",
    "motivation",
    "role_fit",
    "other_explicit_facts"
}


PROFILE_FIELDS = {
    "experience",
    "previous_responsibilities",
    "skills",
    "education_certifications"
}


BEHAVIORAL_FIELDS = {
    "problem_solving",
    "teamwork",
    "communication",
    "conflict_resolution",
    "prioritization"
}


AVAILABILITY_FIELDS = {
    "availability",
    "notice_period",
    "salary_expectations"
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
आप BlueHyreAI के उम्मीदवार सूचना निष्कर्षण मॉडल हैं।

इंटरव्यू प्रश्न और उम्मीदवार के उत्तर से केवल वही जानकारी
निकालें जो उत्तर में स्पष्ट रूप से दी गई है।

सभी स्पष्ट तथ्यों को सुरक्षित रखें। अनुमान या unsupported
information न जोड़ें।

Hierarchical schema का उपयोग करें।

BEHAVIORAL FIELD BOUNDARIES:

problem_solving:
समस्या की पहचान, investigation, diagnosis, troubleshooting,
fix, resolution, mitigation, recovery, implementation या
solution.

teamwork:
टीम के साथ काम करना, collaboration, coordination, delegation,
task assignment/division, leadership, coaching, mentoring,
team management.

communication:
inform करना, update देना, report करना, explain करना,
notify करना, feedback देना, stakeholder/client communication,
status/risk communicate करना.

conflict_resolution:
disagreement, dispute, negotiation, mediation, compromise,
persuasion या interpersonal conflict resolve करना.

prioritization:
किसे पहले करना है, priority order, urgency, deferment,
trade-off, selection या resource priority तय करना.

महत्वपूर्ण:
यदि communication का action किसी problem scenario में है,
तो उसे केवल problem_solving में न डालें।
यदि team coordination किसी problem के दौरान है, तो केवल
problem_solving में न डालें।
यदि कोई action यह तय करता है कि क्या पहले होगा, तो
prioritization का उपयोग करें।
यदि कोई action disagreement/negotiation से संबंधित है,
तो conflict_resolution का उपयोग करें।

एक उत्तर में अलग-अलग semantic facts हों तो उन्हें अलग-अलग
fields में रखें।

केवल वैध hierarchical JSON object लौटाएं।
"""


# ============================================================
# HELPERS
# ============================================================

def clean(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def normalize_text(value):

    text = clean(value).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def parse_dict(value):

    text = clean(value)

    if not text:
        return {}


    try:
        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            dict
        ):

            return parsed

    except Exception:
        pass


    try:
        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            dict
        ):

            return parsed

    except Exception:
        pass


    return {}


def is_empty(value):

    if value is None:
        return True

    if value == "":
        return True

    if isinstance(value, list):
        return len(value) == 0

    if isinstance(value, dict):
        return len(value) == 0

    return False


def merge_value(current, new_value):

    if is_empty(new_value):
        return current

    if is_empty(current):
        return new_value


    current_values = (
        list(current)
        if isinstance(current, list)
        else [current]
    )


    new_values = (
        list(new_value)
        if isinstance(new_value, list)
        else [new_value]
    )


    combined = (
        current_values
        +
        new_values
    )


    result = []
    seen = set()


    for item in combined:

        marker = json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True
        )


        if marker not in seen:

            seen.add(marker)

            result.append(
                item
            )


    if len(result) == 1:
        return result[0]

    return result


def contains_any(
    text,
    phrases
):

    return any(
        phrase in text
        for phrase in phrases
    )


# ============================================================
# QUESTION INTENT
# ============================================================

def get_intents(
    question,
    category
):

    q = normalize_text(
        question
    )

    c = normalize_text(
        category
    )


    intents = set()


    # --------------------------------------------------------
    # Conflict
    # --------------------------------------------------------

    if contains_any(
        q,
        [
            "conflict",
            "disagreement",
            "dispute",
            "argument",
            "negotiat",
            "mediat",
            "compromise",
            "असहमति",
            "विवाद",
            "झगड़ा",
            "मतभेद"
        ]
    ):

        intents.add(
            "conflict"
        )


    # --------------------------------------------------------
    # Prioritization
    # --------------------------------------------------------

    if contains_any(
        q,
        [
            "priorit",
            "priority",
            "which should",
            "what first",
            "do first",
            "first priority",
            "urgent",
            "two tasks",
            "two projects",
            "पहले किसे",
            "पहले क्या",
            "प्राथमिकता",
            "तुरंत किसे",
            "सबसे पहले"
        ]
    ):

        intents.add(
            "prioritization"
        )


    if "priorit" in c:

        intents.add(
            "prioritization"
        )


    # --------------------------------------------------------
    # Communication
    # --------------------------------------------------------

    if contains_any(
        q,
        [
            "communicat",
            "inform",
            "update",
            "report",
            "explain",
            "notify",
            "feedback",
            "stakeholder",
            "client",
            "customer",
            "बताएं",
            "बताएंगे",
            "सूचित",
            "अपडेट",
            "रिपोर्ट",
            "समझाएं"
        ]
    ):

        intents.add(
            "communication"
        )


    # --------------------------------------------------------
    # Teamwork
    # --------------------------------------------------------

    if contains_any(
        q,
        [
            "team",
            "teamwork",
            "coworker",
            "colleague",
            "collaborat",
            "coordinate",
            "delegat",
            "leadership",
            "mentor",
            "coach",
            "cross-functional",
            "टीम",
            "सहकर्मी",
            "सहयोग",
            "समन्वय"
        ]
    ):

        intents.add(
            "teamwork"
        )


    # --------------------------------------------------------
    # Problem solving
    # --------------------------------------------------------

    if contains_any(
        q,
        [
            "problem",
            "issue",
            "incident",
            "failure",
            "troubleshoot",
            "diagnos",
            "root cause",
            "solve",
            "fix",
            "resolve",
            "mitigat",
            "recover",
            "debug",
            "समस्या",
            "दिक्कत",
            "इश्यू",
            "कैसे हल",
            "समाधान"
        ]
    ):

        intents.add(
            "problem_solving"
        )


    # --------------------------------------------------------
    # Skills
    # --------------------------------------------------------

    if contains_any(
        q,
        [
            "skill",
            "skills",
            "tool",
            "technology",
            "technologies",
            "framework",
            "कौशल",
            "टूल",
            "तकनीक"
        ]
    ):

        intents.add(
            "skills"
        )


    # --------------------------------------------------------
    # Motivation
    # --------------------------------------------------------

    if contains_any(
        q,
        [
            "motivat",
            "why do you want",
            "why interested",
            "career goal",
            "career",
            "क्यों करना चाहते",
            "रुचि",
            "मोटिवेशन"
        ]
    ):

        intents.add(
            "motivation"
        )


    # --------------------------------------------------------
    # Role fit
    # --------------------------------------------------------

    if contains_any(
        q,
        [
            "role",
            "suitable",
            "fit",
            "requirements",
            "why should we hire",
            "compliance",
            "regulation",
            "security requirement",
            "मानकों",
            "आवश्यकता",
            "सुरक्षा"
        ]
    ):

        intents.add(
            "role_fit"
        )


    return intents


# ============================================================
# SEMANTIC CLASSIFICATION OF A FACT
# ============================================================

def classify_behavioral_fact(
    fact_key,
    fact_value,
    question,
    answer,
    category
):

    key = normalize_text(
        fact_key
    )

    value = normalize_text(
        fact_value
    )

    q = normalize_text(
        question
    )

    a = normalize_text(
        answer
    )


    context = (
        q
        +
        " "
        +
        a
    )


    intents = get_intents(
        question,
        category
    )


    # ========================================================
    # 1. EXPLICIT CONFLICT SIGNALS
    # ========================================================

    if contains_any(
        key,
        [
            "conflict",
            "disagreement",
            "dispute",
            "negotiat",
            "mediat",
            "compromise",
            "persuasion",
            "argument"
        ]
    ):

        return "conflict_resolution"


    if contains_any(
        value,
        [
            "conflict",
            "disagreement",
            "dispute",
            "negotiate",
            "mediation",
            "compromise",
            "असहमति",
            "विवाद",
            "समझौता"
        ]
    ):

        return "conflict_resolution"


    # ========================================================
    # 2. EXPLICIT PRIORITIZATION
    # ========================================================

    if contains_any(
        key,
        [
            "priorit",
            "priority",
            "urgency",
            "priority_order",
            "selection_criteria",
            "priority_action",
            "decision_priority",
            "prioritized"
        ]
    ):

        return "prioritization"


    if contains_any(
        value,
        [
            "first",
            "पहले",
            "priority",
            "urgent",
            "most important",
            "प्राथमिकता",
            "तुरंत"
        ]
    ) and (
        "prioritization"
        in intents
    ):

        return "prioritization"


    # ========================================================
    # 3. COMMUNICATION ACTIONS
    # ========================================================

    communication_key = contains_any(
        key,
        [
            "communicat",
            "stakeholder",
            "client_communication",
            "reporting",
            "notification",
            "update",
            "feedback",
            "explanation",
            "transparency",
            "inform"
        ]
    )


    communication_value = contains_any(
        value,
        [
            "inform",
            "update",
            "report",
            "explain",
            "notify",
            "feedback",
            "communicat",
            "बताऊंगा",
            "बताएंगे",
            "सूचित",
            "अपडेट",
            "रिपोर्ट"
        ]
    )


    if (
        communication_key
        or
        communication_value
    ):

        return "communication"


    # ========================================================
    # 4. TEAMWORK ACTIONS
    # ========================================================

    teamwork_key = contains_any(
        key,
        [
            "collaboration",
            "collaborat",
            "teamwork",
            "coordination",
            "team_coordination",
            "team_management",
            "leadership",
            "delegation",
            "delegat",
            "task_assignment",
            "task_allocation",
            "work_division",
            "team_division",
            "coaching",
            "mentoring"
        ]
    )


    teamwork_value = contains_any(
        value,
        [
            "team",
            "collaborat",
            "coordinate",
            "delegate",
            "delegation",
            "together",
            "साथ",
            "टीम",
            "सहयोग",
            "समन्वय"
        ]
    )


    if (
        teamwork_key
        or
        teamwork_value
    ):

        # If the question is explicitly asking
        # about solving a technical incident, a single
        # coordination action may still be problem-solving.
        #
        # But if people/team interaction is itself the
        # semantic subject, teamwork wins.

        if (
            "teamwork"
            in intents
        ):

            return "teamwork"


        if (
            "communication"
            in intents
            and
            not "teamwork" in intents
        ):

            return "communication"


        return "teamwork"


    # ========================================================
    # 5. PROBLEM-SOLVING ACTIONS
    # ========================================================

    problem_key = contains_any(
        key,
        [
            "solution",
            "proposed_solution",
            "technical_solution",
            "troubleshoot",
            "troubleshooting",
            "diagnos",
            "investigation",
            "root_cause",
            "mitigation",
            "mitigation_strategy",
            "resolution",
            "fix",
            "incident_response",
            "recovery",
            "remediation",
            "problem",
            "issue",
            "analysis",
            "testing",
            "validation",
            "implementation",
            "architecture",
            "technical_approach"
        ]
    )


    problem_value = contains_any(
        value,
        [
            "solve",
            "fix",
            "investigate",
            "diagnos",
            "root cause",
            "troubleshoot",
            "resolve",
            "mitigat",
            "recover",
            "rollback",
            "debug",
            "समस्या",
            "समाधान",
            "ठीक",
            "जांच"
        ]
    )


    if (
        problem_key
        or
        problem_value
    ):

        # A priority question takes precedence if the
        # fact is explicitly about ordering.
        if "prioritization" in intents:

            if contains_any(
                key,
                [
                    "priority",
                    "prioritization",
                    "priority_order",
                    "selection"
                ]
            ):

                return "prioritization"


        return "problem_solving"


    # ========================================================
    # 6. GENERIC ACTION / STRATEGY
    # ========================================================

    if key in {
        "strategy",
        "strategies",
        "approach",
        "method",
        "methods",
        "action",
        "actions",
        "action_plan",
        "action_steps",
        "first_action",
        "next_action",
        "first_step",
        "next_step",
        "initial_step",
        "immediate_action",
        "immediate_steps",
        "proposed_action",
        "suggested_action",
        "recommendation",
        "proposal",
        "plan",
        "planning"
    }:

        # Strict semantic priority order.
        if "conflict" in intents:
            return "conflict_resolution"

        if "prioritization" in intents:
            return "prioritization"

        if "communication" in intents:
            return "communication"

        if "teamwork" in intents:
            return "teamwork"

        return "problem_solving"


    # ========================================================
    # 7. DEADLINE / CONSTRAINTS
    # ========================================================

    if contains_any(
        key,
        [
            "deadline",
            "timeline",
            "timeframe",
            "time_frame",
            "time_limit",
            "time_constraint",
            "urgency",
            "constraint",
            "constraints"
        ]
    ):

        if "prioritization" in intents:

            return "prioritization"

        return "problem_solving"


    # ========================================================
    # 8. DECISION / CRITERIA
    # ========================================================

    if (
        "decision" in key
        or
        "criteria" in key
        or
        "criterion" in key
        or
        "rationale" in key
        or
        "justification" in key
        or
        "tradeoff" in key
        or
        "trade_off" in key
        or
        key in {
            "choice",
            "factors",
            "factors_considered",
            "considerations",
            "comparison",
            "comparison_factors",
            "decision_basis"
        }
    ):

        if "prioritization" in intents:

            return "prioritization"

        if "conflict" in intents:

            return "conflict_resolution"

        return "problem_solving"


    # ========================================================
    # 9. RISK
    # ========================================================

    if (
        "risk" in key
        or
        "concern" in key
        or
        "risk" in value
        or
        "risk" in context
    ):

        if "prioritization" in intents:
            return "prioritization"

        return "problem_solving"


    # ========================================================
    # 10. REMAINING TEAM SIGNALS
    # ========================================================

    if (
        "team" in context
        and
        contains_any(
            key,
            [
                "role",
                "management",
                "allocation",
                "assignment",
                "division"
            ]
        )
    ):

        return "teamwork"


    # ========================================================
    # 11. DEFAULT
    # ========================================================

    if "problem_solving" in intents:

        return "problem_solving"


    if "teamwork" in intents:

        return "teamwork"


    if "communication" in intents:

        return "communication"


    if "prioritization" in intents:

        return "prioritization"


    if "conflict" in intents:

        return "conflict_resolution"


    return None


# ============================================================
# FLATTEN EXISTING V5 FACTS
# ============================================================

def collect_flat_facts(
    v5_target
):

    facts = []


    if not isinstance(
        v5_target,
        dict
    ):

        return facts


    # --------------------------------------------------------
    # Profile
    # --------------------------------------------------------

    profile = v5_target.get(
        "profile",
        {}
    )


    if isinstance(
        profile,
        dict
    ):

        for field in PROFILE_FIELDS:

            value = profile.get(
                field
            )

            if not is_empty(value):

                facts.append(
                    (
                        field,
                        value,
                        "profile"
                    )
                )


    # --------------------------------------------------------
    # Behavioral
    # --------------------------------------------------------

    behavioral = v5_target.get(
        "behavioral",
        {}
    )


    if isinstance(
        behavioral,
        dict
    ):

        for field in BEHAVIORAL_FIELDS:

            value = behavioral.get(
                field
            )

            if not is_empty(value):

                facts.append(
                    (
                        field,
                        value,
                        "behavioral"
                    )
                )


    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    availability = v5_target.get(
        "availability",
        {}
    )


    if isinstance(
        availability,
        dict
    ):

        for field in AVAILABILITY_FIELDS:

            value = availability.get(
                field
            )

            if not is_empty(value):

                facts.append(
                    (
                        field,
                        value,
                        "availability"
                    )
                )


    # --------------------------------------------------------
    # Single top-level fields
    # --------------------------------------------------------

    if not is_empty(
        v5_target.get(
            "motivation"
        )
    ):

        facts.append(
            (
                "motivation",
                v5_target[
                    "motivation"
                ],
                "top"
            )
        )


    if not is_empty(
        v5_target.get(
            "role_fit"
        )
    ):

        facts.append(
            (
                "role_fit",
                v5_target[
                    "role_fit"
                ],
                "top"
            )
        )


    return facts


# ============================================================
# ROUTE EXISTING BEHAVIORAL FACTS
# ============================================================

def build_v51_target(
    v5_target,
    question,
    answer,
    category
):

    # --------------------------------------------------------
    # Fresh target
    # --------------------------------------------------------

    target = {

        "profile": {

            "experience":
                None,

            "previous_responsibilities":
                None,

            "skills":
                None,

            "education_certifications":
                None
        },

        "behavioral": {

            "problem_solving":
                None,

            "teamwork":
                None,

            "communication":
                None,

            "conflict_resolution":
                None,

            "prioritization":
                None
        },

        "availability": {

            "availability":
                None,

            "notice_period":
                None,

            "salary_expectations":
                None
        },

        "motivation":
            None,

        "role_fit":
            None,

        "other_explicit_facts":
            {}
    }


    # ========================================================
    # COPY PROFILE
    # ========================================================

    source_profile = (
        v5_target.get(
            "profile",
            {}
        )
        if isinstance(
            v5_target,
            dict
        )
        else {}
    )


    if isinstance(
        source_profile,
        dict
    ):

        for field in PROFILE_FIELDS:

            value = source_profile.get(
                field
            )

            if not is_empty(value):

                target[
                    "profile"
                ][field] = value


    # ========================================================
    # COPY AVAILABILITY
    # ========================================================

    source_availability = (
        v5_target.get(
            "availability",
            {}
        )
        if isinstance(
            v5_target,
            dict
        )
        else {}
    )


    if isinstance(
        source_availability,
        dict
    ):

        for field in AVAILABILITY_FIELDS:

            value = source_availability.get(
                field
            )

            if not is_empty(value):

                target[
                    "availability"
                ][field] = value


    # ========================================================
    # COPY MOTIVATION / ROLE FIT
    # ========================================================

    if isinstance(
        v5_target,
        dict
    ):

        target[
            "motivation"
        ] = v5_target.get(
            "motivation"
        )


        target[
            "role_fit"
        ] = v5_target.get(
            "role_fit"
        )


    # ========================================================
    # RE-ROUTE BEHAVIORAL FACTS
    # ========================================================

    source_behavioral = (
        v5_target.get(
            "behavioral",
            {}
        )
        if isinstance(
            v5_target,
            dict
        )
        else {}
    )


    if not isinstance(
        source_behavioral,
        dict
    ):

        source_behavioral = {}


    for old_field in BEHAVIORAL_FIELDS:

        old_value = source_behavioral.get(
            old_field
        )


        if is_empty(old_value):

            continue


        # ----------------------------------------------------
        # Convert potentially list-valued facts into individual
        # facts where possible.
        # ----------------------------------------------------

        if isinstance(
            old_value,
            list
        ):

            fact_values = old_value

        else:

            fact_values = [
                old_value
            ]


        for fact_value in fact_values:

            target_field = classify_behavioral_fact(

                fact_key=old_field,

                fact_value=fact_value,

                question=question,

                answer=answer,

                category=category
            )


            # ------------------------------------------------
            # If classifier cannot improve the assignment,
            # retain original field instead of losing data.
            # ------------------------------------------------

            if target_field is None:

                target_field = old_field


            if target_field not in BEHAVIORAL_FIELDS:

                target_field = old_field


            target[
                "behavioral"
            ][target_field] = merge_value(

                target[
                    "behavioral"
                ][target_field],

                fact_value
            )


    # ========================================================
    # PRESERVE OTHER FACTS
    # ========================================================

    source_other = (
        v5_target.get(
            "other_explicit_facts",
            {}
        )
        if isinstance(
            v5_target,
            dict
        )
        else {}
    )


    if isinstance(
        source_other,
        dict
    ):

        target[
            "other_explicit_facts"
        ] = dict(
            source_other
        )


    return target


# ============================================================
# VALIDATION
# ============================================================

def validate_target(
    target
):

    if set(
        target.keys()
    ) != TOP_LEVEL_FIELDS:

        return False


    if set(
        target[
            "profile"
        ].keys()
    ) != PROFILE_FIELDS:

        return False


    if set(
        target[
            "behavioral"
        ].keys()
    ) != BEHAVIORAL_FIELDS:

        return False


    if set(
        target[
            "availability"
        ].keys()
    ) != AVAILABILITY_FIELDS:

        return False


    if not isinstance(
        target[
            "other_explicit_facts"
        ],
        dict
    ):

        return False


    return True


# ============================================================
# MODEL RECORD
# ============================================================

def build_record(
    question,
    answer,
    target
):

    return {

        "messages": [

            {
                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT.strip()
            },

            {
                "role":
                    "user",

                "content": (
                    "इंटरव्यू प्रश्न:\n"
                    f"{question}\n\n"
                    "उम्मीदवार का उत्तर:\n"
                    f"{answer}"
                )
            },

            {
                "role":
                    "assistant",

                "content":
                    json.dumps(
                        target,
                        ensure_ascii=False
                    )
            }
        ]
    }


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split_name,
    input_path
):

    print()
    print("=" * 80)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print("=" * 80)


    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig"
    )


    rows = []

    changed_rows = 0

    total_behavioral_values = 0


    for _, row in df.iterrows():

        question = clean(
            row[
                "question_hi"
            ]
        )

        answer = clean(
            row[
                "candidate_answer_hi"
            ]
        )

        category = clean(
            row.get(
                "bluehyre_category",
                ""
            )
        )


        v5_target = parse_dict(
            row[
                "normalized_facts"
            ]
        )


        if not isinstance(
            v5_target,
            dict
        ):

            raise RuntimeError(
                f"Invalid V5 target for "
                f"{row['record_id']}"
            )


        old_behavioral = (
            v5_target.get(
                "behavioral",
                {}
            )
        )


        old_signature = json.dumps(
            old_behavioral,
            ensure_ascii=False,
            sort_keys=True
        )


        for field in BEHAVIORAL_FIELDS:

            value = (
                old_behavioral.get(
                    field
                )
                if isinstance(
                    old_behavioral,
                    dict
                )
                else None
            )


            if isinstance(
                value,
                list
            ):

                total_behavioral_values += len(
                    value
                )

            elif not is_empty(value):

                total_behavioral_values += 1


        v51_target = build_v51_target(

            v5_target,

            question,

            answer,

            category
        )


        new_signature = json.dumps(
            v51_target[
                "behavioral"
            ],
            ensure_ascii=False,
            sort_keys=True
        )


        if (
            old_signature
            !=
            new_signature
        ):

            changed_rows += 1


        if not validate_target(
            v51_target
        ):

            raise RuntimeError(

                f"V5.1 schema validation failed "
                f"for {row['record_id']}"
            )


        rows.append({

            "record_id":
                clean(
                    row[
                        "record_id"
                    ]
                ),

            "source_question_id":
                clean(
                    row[
                        "source_question_id"
                    ]
                ),

            "question_hi":
                question,

            "candidate_answer_hi":
                answer,

            "normalized_facts":
                json.dumps(
                    v51_target,
                    ensure_ascii=False
                )
        })


    output_df = pd.DataFrame(
        rows
    )


    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{split_name}.csv"
    )


    output_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )


    jsonl_path = os.path.join(
        OUTPUT_DIR,
        f"{split_name}.jsonl"
    )


    with open(
        jsonl_path,
        "w",
        encoding="utf-8"
    ) as file:

        for _, row in output_df.iterrows():

            target = parse_dict(
                row[
                    "normalized_facts"
                ]
            )


            record = build_record(

                question=row[
                    "question_hi"
                ],

                answer=row[
                    "candidate_answer_hi"
                ],

                target=target
            )


            file.write(

                json.dumps(
                    record,
                    ensure_ascii=False
                )
                +
                "\n"
            )


    return {

        "examples":
            len(output_df),

        "questions":
            output_df[
                "source_question_id"
            ].nunique(),

        "changed_rows":
            changed_rows,

        "behavioral_values":
            total_behavioral_values,

        "csv":
            csv_path,

        "jsonl":
            jsonl_path
    }


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print(
    "BLUEHYREAI - EXTRACTION DATASET V5.1"
)
print("=" * 80)
print()

print(
    "Source:"
)

print(
    INPUT_DIR
)

print()

print(
    "Behavioral routing:"
)

print(
    "problem_solving | teamwork | communication | "
    "conflict_resolution | prioritization"
)

print()


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(
    INPUT_DIR
):

    raise FileNotFoundError(
        f"V5 directory not found:\n"
        f"{INPUT_DIR}"
    )


for filename in [

    "train.csv",
    "validation.csv",
    "test.csv"

]:

    path = os.path.join(
        INPUT_DIR,
        filename
    )


    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing V5 split:\n{path}"
        )


# ============================================================
# CREATE OUTPUT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# BUILD
# ============================================================

summary = {}


summary[
    "train"
] = process_split(

    "train",

    TRAIN_FILE
)


summary[
    "validation"
] = process_split(

    "validation",

    VAL_FILE
)


summary[
    "test"
] = process_split(

    "test",

    TEST_FILE
)


# ============================================================
# SPLIT INTEGRITY
# ============================================================

train_df = pd.read_csv(
    summary[
        "train"
    ]["csv"],
    encoding="utf-8-sig"
)

val_df = pd.read_csv(
    summary[
        "validation"
    ]["csv"],
    encoding="utf-8-sig"
)

test_df = pd.read_csv(
    summary[
        "test"
    ]["csv"],
    encoding="utf-8-sig"
)


train_q = set(
    train_df[
        "source_question_id"
    ]
)

val_q = set(
    val_df[
        "source_question_id"
    ]
)

test_q = set(
    test_df[
        "source_question_id"
    ]
)


if train_q & val_q:

    raise RuntimeError(
        "Train/validation question leakage."
    )


if train_q & test_q:

    raise RuntimeError(
        "Train/test question leakage."
    )


if val_q & test_q:

    raise RuntimeError(
        "Validation/test question leakage."
    )


# ============================================================
# TWO ANSWERS PER QUESTION
# ============================================================

for split_name, df in [

    ("train", train_df),

    ("validation", val_df),

    ("test", test_df)

]:

    counts = (
        df
        .groupby(
            "source_question_id"
        )
        .size()
    )


    invalid = counts[
        counts != 2
    ]


    if len(invalid):

        raise RuntimeError(

            f"{split_name}: "
            f"{len(invalid)} questions "
            "do not have exactly 2 answers."
        )


# ============================================================
# COMBINED REPORT
# ============================================================

report = pd.DataFrame(

    [

        {

            "split":
                split_name,

            "examples":
                result[
                    "examples"
                ],

            "questions":
                result[
                    "questions"
                ],

            "behavioral_values":
                result[
                    "behavioral_values"
                ],

            "changed_rows":
                result[
                    "changed_rows"
                ]
        }

        for split_name, result
        in summary.items()
    ]
)


report_file = os.path.join(
    OUTPUT_DIR,
    "v5_1_dataset_report.csv"
)


report.to_csv(
    report_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {

    "dataset_version":
        "BlueHyreAI Extraction V5.1",

    "source_dataset":
        "llama_extraction_v5",

    "schema_type":
        "hierarchical_context_corrected",

    "split_preserved":
        True,

    "train_examples":
        summary[
            "train"
        ]["examples"],

    "validation_examples":
        summary[
            "validation"
        ]["examples"],

    "test_examples":
        summary[
            "test"
        ]["examples"],

    "train_questions":
        summary[
            "train"
        ]["questions"],

    "validation_questions":
        summary[
            "validation"
        ]["questions"],

    "test_questions":
        summary[
            "test"
        ]["questions"],

    "behavioral_fields":
        sorted(
            BEHAVIORAL_FIELDS
        ),

    "exactly_two_answers_per_question":
        True,

    "question_leakage":
        False,

    "other_explicit_facts_always_dict":
        True,

    "purpose":
        "Context-based correction of behavioral field assignments"
}


manifest_file = os.path.join(
    OUTPUT_DIR,
    "manifest.json"
)


with open(
    manifest_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

total_examples = sum(
    result[
        "examples"
    ]

    for result
    in summary.values()
)


total_changed = sum(
    result[
        "changed_rows"
    ]

    for result
    in summary.values()
)


total_behavioral = sum(
    result[
        "behavioral_values"
    ]

    for result
    in summary.values()
)


print()
print("=" * 80)
print(
    "BLUEHYREAI EXTRACTION DATASET V5.1 COMPLETE"
)
print("=" * 80)

print()

print(
    f"Total examples          : "
    f"{total_examples:,}"
)

print(
    f"Train examples          : "
    f"{summary['train']['examples']:,}"
)

print(
    f"Validation examples     : "
    f"{summary['validation']['examples']:,}"
)

print(
    f"Test examples           : "
    f"{summary['test']['examples']:,}"
)

print()

print(
    f"Behavioral fact values  : "
    f"{total_behavioral:,}"
)

print(
    f"Rows context-corrected : "
    f"{total_changed:,}"
)

print()

print(
    "BEHAVIORAL SCHEMA:"
)

for field in [
    "problem_solving",
    "teamwork",
    "communication",
    "conflict_resolution",
    "prioritization"
]:

    print(
        f"  behavioral.{field}"
    )


print()

print(
    "OUTPUT:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "REPORT:"
)

print(
    report_file
)

print()

print(
    "MANIFEST:"
)

print(
    manifest_file
)

print()
print("=" * 80)
print(
    "V5.1 READY FOR REVIEW"
)
print("=" * 80)