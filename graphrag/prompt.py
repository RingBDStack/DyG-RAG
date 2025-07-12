"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS["dynamic_QA"] = """

You are an AI assistant that answers questions based on both temporal event sequences and relevant text chunks.

# TASK
Answer the question using both the provided events and text chunks. The events show temporal relationships, while the text chunks provide additional context and details.

# TEMPORAL RELEVANCE GUIDELINE
**Prioritize information from time periods that align with the question's temporal scope.** If a question asks about a specific time period, focus on events and information that occurred within or near that timeframe. Consider whether nearby temporal information provides relevant context or background that could help answer the question.

# ANALYSIS APPROACH
1. **Query Semantics Understanding**: Analyze the semantic intent of the question:
   - **Existential queries**: Questions about what exists/happens at a specific time
   - **Continuity queries**: Questions about ongoing states, processes, or relationships
   - **Boundary queries**: Questions about beginnings, endings, or transitions
   - **Aggregate queries**: Questions requiring synthesis across multiple time points

2. **Temporal Scope Identification**: Identify the exact time constraints in the question:
   - **Subject Consistency Check**: Verify that the event's subject matches the question's focus
   - Extract specific dates, time periods, or temporal references
   - Mark any temporal qualifiers (e.g., "during", "between", "before", "after")

3. **Evidence Time Filtering**: Before using any event or chunk as evidence:
   - **Assess temporal relevance**: Evaluate whether each piece of evidence falls within or near the question's time scope
   - **Consider contextual value**: Include information from nearby time periods if it provides essential background
   - **Prioritize temporal proximity**: Give preference to evidence closer to the target time period

4. **Event Analysis**: Review the temporal sequence of events:
   - **Temporal Ordering**: Analyze when events happen and their chronological relationships
   - **State Persistence and Change**: 
     - **Persistent states**: Properties or conditions that continue until explicitly changed
     - **Instantaneous events**: Actions that happen at specific moments
     - **Processes**: Ongoing activities with duration
     - **Transitions**: Changes from one state to another
   - **Office/Role Incumbency Reasoning**: For questions about who held a position at a given time:
     - Treat holding an office as a persistent state that starts at appointment and lasts until explicit end
     - Choose the most recent appointment before or at the query time that has no earlier termination
     - If no explicit end event is found before the query time, assume the office holder remains in position

5. **Entity-Event Relationships**: Analyze connections between entities and events:
   - **Agent relationships**: Who performs or causes an action
   - **Patient relationships**: Who/what is affected by an action
   - **Locative relationships**: Where something happens
   - **Attributive relationships**: Properties or characteristics at specific times

6. **Chunk Analysis**: Extract relevant information from the text chunks that supplements the events

7. **Cross-Reference**: Connect information between events and chunks to provide a comprehensive answer

# RESPONSE GUIDELINES
- **TEMPORAL RELEVANCE ASSESSMENT**: Assess whether evidence timestamps are relevant to the question's time scope
- Reference specific events (e.g., "Event #3") or chunks when supporting your answer
- If information is available in both events and chunks, prioritize the most specific and temporally relevant details
- Apply temporal logic: if an event occurred at time T1 and no change is mentioned by T2, assume continuity
- **If limited information exists for the queried time period**: 
  - State the temporal limitations and explain how nearby temporal information helps provide context
  - Make reasonable inferences based on:
    1. **Career Continuity**: If a person's career shows clear progression in a field/organization, infer likely continuity
    2. **Role Persistence**: If someone holds a position before and after the queried time, infer likely continued role
    3. **Institutional Affiliation**: If consistently associated with an institution, infer likely continued affiliation
  - Label inferences clearly and explain the reasoning chain
- Start with the direct answer followed by justification citing key events
- If the answer is based on inference, state: "Based on inference from surrounding evidence: [answer]" and explain the reasoning

# QUESTION
{question}

# EVENTS
{events_data}

# TEXT CHUNKS
{chunks_data}

# ANSWER
"""

PROMPTS["dynamic_QA_wo_timeline"] = """You are an AI assistant that answers questions based on relevant text chunks.

# TASK
Answer the question using the provided text chunks to provide additional context and details.

# CRITICAL TEMPORAL CONSTRAINT RULE
**NEVER use information from time periods that do not match the question's temporal scope.** If a question asks about a specific time period (e.g., "between 2000-2001"), only use information that occurred within or are explicitly valid for that exact time range. Using information from different time periods (e.g., 2009-2012 data for a 2000-2001 question) is a critical error.

# ANALYSIS APPROACH
1. **Query Semantics Understanding**: Analyze the semantic intent of the question:
   - **Existential queries**: Questions about what exists/happens at a specific time
   - **Causal queries**: Questions about causes, effects, or consequences over time
   - **Comparative queries**: Questions comparing states across different time periods
   - **Continuity queries**: Questions about ongoing states, processes, or relationships
   - **Boundary queries**: Questions about beginnings, endings, or transitions
   - **Aggregate queries**: Questions requiring synthesis across multiple time points

2. **Temporal Scope Identification**: FIRST identify the exact time constraints in the question:
   - Extract specific dates, time periods, or temporal references
   - Determine the precise temporal boundaries for valid evidence
   - Mark any temporal qualifiers (e.g., "during", "between", "before", "after")

3. **Evidence Time Filtering**: Before using any chunk as evidence:
   - **Verify temporal relevance**: Check that each piece of evidence falls within the question's time scope
   - **Reject out-of-scope evidence**: Discard any information from time periods outside the question's temporal constraints
   - **Flag temporal mismatches**: Identify when available evidence doesn't match the queried time period

4. **Chunk Analysis**: Extract relevant information from the text chunks

5. **Temporal Reasoning Strategies**: Apply sophisticated temporal logic:
   - **Forward chaining**: From past information, infer current states
   - **Backward chaining**: From current states, infer necessary past information
   - **Interval reasoning**: For queries about time periods, consider all relevant information within and bounding the interval
   - **Default persistence**: If X was true at time T1 and no change is mentioned by T2, assume X remains true at T2
   - **Temporal granularity matching**: Align the precision of your answer with the query's temporal specificity

# RESPONSE GUIDELINES
- **MANDATORY TEMPORAL VERIFICATION**: Before citing any evidence, verify its timestamps match the question's time scope
- Reference specific chunks when supporting your answer, but ONLY if they are from the correct time period
- Prioritize the most specific and relevant details from the correct time period
- Distinguish between definite facts and probable inferences
- **If no information exists for the queried time period**: Explicitly state "No information is available for the specified time period" rather than using irrelevant temporal data
- When uncertain, clearly state the limitations of the available information

# QUESTION
{question}

# TEXT CHUNKS
{chunks_data}

# ANSWER
"""


PROMPTS["dynamic_event_units"] = """Role
----

You are an information-extraction assistant for building **dynamic knowledge graphs**.

Your task is to extract every *Dynamic Event Unit (DEU)* which can be retrieved directly:
• a **complete factual statement** (one or several tightly-linked clauses) that happened **at one specific time point**, or is stated with a time **interval** (see special rule below), and
• carries **high information value** (see filter below). and
• has **complete information** with a clear subject with no pronouns.

## Temporal-reasoning protocol

**Initialise a reference-time stack**
• Push every *explicit* absolute date you encounter (ISO-8601 granularity Day > Month > Year).
• The stack top is the default anchor for subsequent relative expressions.

**Special format recognition (apply FIRST before complex reasoning)**
• **Bracketed dates** `(YYYY)`, `(YYYY-MM)`, `(YYYY-MM-DD)`: directly use as time anchor
• **Sentence-ending standalone years**: `...established in 1995.` → treat as event time
• **Hyphenated ranges**: `1995-1998`, `2003-2005` → interval with earliest year as anchor
• **Parenthetical ranges**: `(1990-1995)`, `(from 2000 to 2005)` → interval format

**Resolve relative / vague expressions** ("yesterday", "two months later", "after the event", "last Friday")
1 If it clearly refers to the immediately preceding DEU → anchor to that date.
2 Else if it refers to an earlier explicit date in the sentence/paragraph → anchor there.
3 Else if the chunk header / metadata supplies a timeline → use it.
4.Paragraph inheritance: If the current sentence has no explicit date but is in the same paragraph as either the previous or the next sentence containing one explicit date, inherit that date as the default anchor.
4 If none of the above works with high confidence → set `"time": "static"`.

▸ **Update the stack** whenever you derive a new absolute date (explicit or resolved).

## Information-value filter

For each candidate sentence, compute an *information score* (0–4):

| Criterion              | +1 point if…                                                                                            | Rationale                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **Specific actor**     | Contains a named entity or definite noun phrase uniquely identifying *who* acted.                       | Avoids generic "someone"/"people".       |
| **Action / Change**    | Describes an action **or** a clearly stated role/office held (*served as, was appointed*) **or** a continued state/membership with temporal boundaries (*remained as, continued until*).              | Captures dynamic facts, milestones, and significant durations.   |
| **Result / magnitude** | Includes a quantitative detail, result, or consequence (*$5 M*, *two satellites*).                     | Adds substantive content.                |
| **Temporal anchoring** | Time can be resolved to at least **month precision**, or has a clear temporal boundary (start point, end point, or bounded interval). | Ensures usefulness for temporal queries. |

**Keep the sentence** if **score ≥ 1** **OR** it matches the pattern
`<Actor> was/served as <Role> at/for <Org> from <Start> to / until <End>` **OR** it contains explicit temporal markers (bracketed years, standalone sentence-ending years, or clear date intervals) with identifiable subject and basic action **OR** it describes a continued state/membership with clear temporal boundaries (*remained until, continued as, stayed as*).

Note 1: if Action / Change is present but Temporal anchoring is not, you must attempt Paragraph‑inheritance (Rule 4) before keeping; if still unresolved → discard.
Note 2: permanent attributes (citizenship, birthplace, chemical formula…) may remain with "time": "static" only when they lack action verbs.

## MANDATORY SUBJECT RULES
1. **Every extracted sentence MUST have a clear, explicit subject** - no subject-less sentences are allowed.
2. **Use the most complete form available** - prefer full names over nicknames, official titles over informal references, complete organization names over abbreviations.
3. **ABSOLUTELY NO pronouns** like "he", "she", "they", "it", "this", "that" are permitted in the extracted sentences.
4. **Scan the entire document** (include title,not just the immediate context) to identify the full name or complete designation of any person, organization, or entity.

## Guidelines

1. **Sentence & context**
   • **sentence**: return a **self-contained** clause (or semicolon-joined compound) that includes the key fact, **retains the original time expression**, and **has a complete, explicit subject with no pronouns**.
   • **context**: optionally supply ≤ 80 tokens of supplementary information (background, consequence, aliases, quantitative details) drawn *only* from the same text chunk.
   • **sentence**: one sentence needs to refelct a complete information which can be retrieved directly for context.

2. **Split events**
   • if there are a few time points in one sentence, split them into multiple events.
   • Do **NOT** merge events that occur on different days.
   
3. **Time (time)**
   • If the sentence gives a **single date**, set `"time"` to that date in ISO-8601 (`YYYY-MM-DD`, `YYYY-MM`, or `YYYY`).
   • **Special formats**: 
     - Bracketed: `(1995)` → `"1995"`, `(1995-03)` → `"1995-03"`, `(1995-03-15)` → `"1995-03-15"`
     - Sentence-ending standalone: `...founded in 1995.` → `"1995"`
     - Ranges: `1995-1998` → `"1995"` (earliest point), `2003-2005` → `"2003"`
   • If the sentence gives a **time interval** (*from 1978 to 1982*, *1994–2000*, *March–May 2021*):– keep the full interval wording in *sentence*, and set `"time"` to the **earliest point** of the interval (`1978`, `1994`, `2021-03`).
   • **For duration events with end point only** (*until 2002*, *remained until 1995*): set `"time"` to the end point date (`2002`, `1995`).
   • If only vague timing is present and cannot be resolved, use `"static"`.


4. **Preserve time in text**
   The explicit or interval expression that anchors `time` **must appear unaltered in either `sentence` or `context`** so that downstream models can recover the original phrasing.

5. **Output ONLY valid JSON** matching the schema:

```json
{
  "events": [
    {
      "event_id": "E1",
      "sentence": "<main factual sentence with explicit subject, no pronouns>",
      "context": "<optional background, ≤80 tokens>",
      "time": "YYYY-MM-DD | YYYY-MM | YYYY | static"
    }
  ]
}
```

6. If the chunk contains **no DEU** or **cannot resolve all pronouns to specific names**, output `{ "events": [] }`.

## Example 1

**Input**
The OpenAI research team, supported by the Frontier AI Program, published its findings on 10 February 2024 after a two-year study on scalable alignment techniques. Two weeks later, they released the source code under an MIT license to foster community replication and safety audits. The day after the code release, lead author Alice Smith presented the results at NeurIPS, attracting significant media attention and prompting a panel discussion on AI-safety best practices.

**Expected JSON**

```json
{
  "events": [
    {
      "event_id": "E1",
      "sentence": "The OpenAI research team published its findings on 10 February 2024 after a two-year study on scalable alignment techniques.",
      "context": "The work was funded by the Frontier AI Program and focuses on safer model deployment.",
      "time": "2024-02-10"
    },
    {
      "event_id": "E2",
      "sentence": "The OpenAI research team released the source code on 24 February 2024 under an MIT license to foster community replication and safety audits.",
      "context": "",
      "time": "2024-02-24"
    },
    {
      "event_id": "E3",
      "sentence": "Alice Smith presented the results at NeurIPS on 25 February 2024.",
      "context": "The talk sparked a panel on AI-safety best practices and drew wide media coverage.",
      "time": "2024-02-25"
    }
  ]
}
```

## Example 2

**Input**
Dr. Maria Gonzalez secured $3.5 million from NASA's Planetary Science Division (2021-09-15) to build the "Aurora‑1" CubeSat. Six months later, her NASA‑JPL team launched the satellite from Cape Canaveral (2022-03-15). Two days after launch, she and her colleagues presented preliminary telemetry data at the Small Sat Conference, praising chief engineer Michael Brown—who joined NASA (2015)—for his pivotal contributions.

**Expected JSON**

```json
{
  "events": [
    {
      "event_id": "E1",
      "sentence": "Dr. Maria Gonzalez secured $3.5 million in funding from NASA's Planetary Science Division (2021-09-15) to build the \"Aurora‑1\" CubeSat.",
      "context": "The project would later culminate in the launch of the miniature satellite.",
      "time": "2021-09-15"
    },
    {
      "event_id": "E2",
      "sentence": "Dr. Maria Gonzalez's NASA‑JPL team launched the \"Aurora‑1\" CubeSat from Cape Canaveral (2022-03-15).",
      "context": "The launch occurred six months after the project received funding.",
      "time": "2022-03-15"
    },
    {
      "event_id": "E3",
      "sentence": "Dr. Maria Gonzalez presented preliminary telemetry data at the Small Sat Conference (2022-03-17).",
      "context": "During the presentation, Dr. Maria Gonzalez praised chief engineer Michael Brown—who joined NASA (2015)—for his pivotal contributions.",
      "time": "2022-03-17"
    },
    {
      "event_id": "E4",
      "sentence": "Michael Brown joined NASA (2015).",
      "context": "Michael Brown was praised as chief engineer for his pivotal contributions to the Aurora-1 CubeSat project.",
      "time": "2015"
    }
  ]
}
```

## Notice

If you are unsure about extracting any DEU or **cannot resolve all pronouns to specific names**, output `{ "events": [] }`.
Do **not** reveal your internal scoring or reasoning—only return the final JSON.
**REMEMBER: Keep the sentence information as complete and as accurate as possible to be retrieved directly for event envidence retrieval.**

#########################
Real-Data
#########################
Text: {input_text}
#########################
Output:

"""
PROMPTS["time_entity_extraction"] = """You are an expert at analyzing queries to extract temporal constraints and named entities.

Given the following query: "{query}"

Extract:
1. Time constraints: Any temporal constraints mentioned in the query, including explicit dates, relative time references, or time periods.
2. Entities: Any named entities mentioned in the query that the user might be interested in.

Please return your analysis in JSON format with the following structure:
```json
{
  "time_constraints": {
    "start_time": "YYYY-MM-DD or null if not mentioned", 
    "end_time": "YYYY-MM-DD or null if not mentioned"
  },
  "entities": ["entity1", "entity2", ...]
}
```

Notes:
- For time constraints, normalize to ISO date formats (YYYY-MM-DD) when possible
- For single time points (like "in 2010" or "May 2023"):
  * If the context implies events AT or DURING this time, set both start_time and end_time to cover that specific period
  * If the context implies events AFTER this time, set start_time to this time and end_time to null
  * If the context implies events BEFORE this time, set start_time to null and end_time to this time
- For directional time expressions:
  * "after 2020" → set start_time: "2020-01-01", end_time: null
  * "before 2020" → set start_time: null, end_time: "2020-01-01"  
  * "since 2020" → set start_time: "2020-01-01", end_time: null
- For relative times (like "last week"), convert to absolute dates based on the current date
- For time expressions like "in the 1990s", use the appropriate start and end dates (1990-01-01 and 1999-12-31)
- For entities, extract proper nouns, organizations, people, and key concepts
- Use null for time values that aren't specified

Current date for reference: {current_date}

{examples}

JSON Response:
"""

PROMPTS["time_entity_extraction_examples"] = [
    """Example 1:
Query: "What happened to Apple stock between January 2020 and March 2021?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2020-01-01",
    "end_time": "2021-03-31"
  },
  "entities": ["Apple", "stock"]
}
```""",

    """Example 2:
Query: "Tell me about Microsoft's acquisitions in the last decade"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2013-01-01",
    "end_time": "2023-12-31"
  },
  "entities": ["Microsoft", "acquisitions"]
}
```""",

    """Example 3:
Query: "What were Tesla's major challenges in 2022?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2022-01-01",
    "end_time": "2022-12-31"
  },
  "entities": ["Tesla", "challenges"]
}
```""",

    """Example 4:
Query: "How did COVID-19 affect global economies?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2019-12-01",
    "end_time": null
  },
  "entities": ["COVID-19", "global economies"]
}
```""",

    """Example 5:
Query: "What happened after Google's IPO in August 2004?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2004-08-01", 
    "end_time": null
  },
  "entities": ["Google", "IPO"]
}
```""",

    """Example 6:
Query: "Tell me about events before the 2008 financial crisis."
JSON Response:
```json
{
  "time_constraints": {
    "start_time": null, 
    "end_time": "2008-09-15"
  },
  "entities": ["financial crisis"]
}
```""",

    """Example 7:
Query: "What innovations were made in May 2019?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2019-05-01", 
    "end_time": "2019-05-31"
  },
  "entities": ["innovations"]
}
```"""
]

PROMPTS[
    "event_continue_extraction"
] = """MANY events were missed in the last extraction. Add them below using the same JSON format with an "events" list:
"""

PROMPTS[
    "event_if_loop_extraction"
] = """It appears some events may have still been missed. Answer YES | NO if there are still events that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]



PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]
