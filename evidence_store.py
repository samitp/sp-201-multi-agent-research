# Evidence store (SDD 6.0, SRS 3.3 and 3.5.6). One SQLite file holding the five
# tables of the SDD data design: question, sub_question, evidence, claim and
# claim_evidence. Agents never pass passage text around; they pass the integer
# evidence_id this module hands out (req. 3.3.3, 3.3.4), and read the text back
# through read_evidence.
#
# Where this differs from the SDD text. The SDD contradicts itself in two places,
# and the ERD (Figure 4) is followed in both:
#   1. sub_question.text is the sub-question text. SDD 6.1.1 labels that column
#      submitted_at by mistake.
#   2. evidence.sub_question_id is an integer foreign key to sub_question. SDD 6.2
#      lists a TEXT column named sub_question, but the ERD draws sub_question
#      "yields" evidence.
# Two columns are added because the pipeline needs them:
#   3. sub_question.plan_number is the Manager's own sub_question_id (1, 2, 3 in
#      each run). SearchResults refers to sub-questions by that number, while the
#      database id is unique across runs.
#   4. claim.claim_number is the draft-local number the Fact-Checker's feedback
#      refers to. claim.claim_id is assigned on insert.
# Items 1 to 4 should be copied back into the SDD.
#
# The default file is sp201_evidence.db, not evidence_store.db, so it cannot
# collide with the older table search_agent.py still creates (README open item 5).
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import NamedTuple

from schemas import DraftReport
from schemas import EvidenceRecord
from schemas import FactCheckVerdict
from schemas import ManagerPlan
from schemas import SourceType
from schemas import Verdict

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "sp201_evidence.db"

# Longest passage split_into_passages produces by default (README Decision F)
DEFAULT_PASSAGE_CHARS = 1500


# Raised for anything the store refuses: unknown ids, cross-question citations,
# an unchecked draft, or a database in the wrong shape
class EvidenceStoreError(ValueError):
    pass


# One row of the evidence table, as the Synthesizer and Fact-Checker read it
class StoredEvidence(NamedTuple):
    evidence_id: int
    source_type: str
    source_title: str
    source_url: str
    published_date: str
    passage: str


# Renders values as a quoted SQL list, so CHECK constraints follow the enums
def quoted_list(values):
    quoted = []
    for value in values:
        quoted.append("'" + value + "'")
    return ", ".join(quoted)

SOURCE_TYPE_LIST = quoted_list([source.value for source in SourceType])
VERDICT_LIST = quoted_list([verdict.value for verdict in Verdict])

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS question (
    question_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    question_text TEXT NOT NULL,
    submitted_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sub_question (
    sub_question_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id     INTEGER NOT NULL REFERENCES question(question_id),
    plan_number     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    UNIQUE (question_id, plan_number)
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    sub_question_id INTEGER NOT NULL REFERENCES sub_question(sub_question_id),
    source_url      TEXT NOT NULL,
    source_title    TEXT NOT NULL,
    published_date  TEXT,
    passage         TEXT NOT NULL,
    retrieved_at    TEXT NOT NULL,
    source_type     TEXT NOT NULL CHECK (source_type IN ({SOURCE_TYPE_LIST}))
);

CREATE TABLE IF NOT EXISTS claim (
    claim_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id  INTEGER NOT NULL REFERENCES question(question_id),
    iteration    INTEGER NOT NULL CHECK (iteration >= 1),
    claim_number INTEGER NOT NULL,
    sentence     TEXT NOT NULL,
    verdict      TEXT NOT NULL CHECK (verdict IN ({VERDICT_LIST})),
    reason       TEXT NOT NULL,
    restatement  TEXT,
    UNIQUE (question_id, iteration, claim_number)
);

CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id    INTEGER NOT NULL REFERENCES claim(claim_id),
    evidence_id INTEGER NOT NULL REFERENCES evidence(evidence_id),
    PRIMARY KEY (claim_id, evidence_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_sub_question ON evidence(sub_question_id);
CREATE INDEX IF NOT EXISTS idx_claim_question ON claim(question_id, iteration);
"""

# Columns the evidence table must have; used to spot an old-shaped file
REQUIRED_EVIDENCE_COLUMNS = {
    "evidence_id", "sub_question_id", "source_url", "source_title",
    "published_date", "passage", "retrieved_at", "source_type",
}

# Current time as an ISO 8601 UTC string, the format every timestamp column uses
def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Splits a large page into passages of at most max_chars, breaking on paragraph
# boundaries first and on spaces only when one paragraph is itself too long.
# Small neighbouring paragraphs are merged. This only splits; choosing which
# passages are relevant to the sub-question is the Search agent's job.
def split_into_passages(text, max_chars=DEFAULT_PASSAGE_CHARS):
    pieces = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        while len(paragraph) > max_chars:
            cut = paragraph.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            pieces.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if paragraph:
            pieces.append(paragraph)
    passages = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) <= max_chars:
            current = current + "\n\n" + piece
        else:
            if current:
                passages.append(current)
            current = piece
    if current:
        passages.append(current)
    return passages


class EvidenceStore:
    # Opens (and if needed creates) the database at db_path
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.init_db()

    # Opens one connection with foreign keys enforced (SQLite leaves them off by default)
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # Creates the five tables if they are missing, and refuses an old-shaped file
    def init_db(self):
        with closing(self.connect()) as conn:
            existing = set()
            for column in conn.execute("PRAGMA table_info(evidence)").fetchall():
                existing.add(column[1])
            if existing and not REQUIRED_EVIDENCE_COLUMNS.issubset(existing):
                raise EvidenceStoreError(
                    f"{self.db_path} has an evidence table with the old shape "
                    "(string ids, raw_content). It is a scratch file; delete it and rerun."
                )
            with conn:
                conn.executescript(SCHEMA_SQL)

    # ----- questions and sub-questions (SDD 6.1) -----

    # Records the submitted question and returns its question_id
    def add_question(self, question_text):
        if not question_text or not question_text.strip():
            raise EvidenceStoreError("question_text is empty")
        with closing(self.connect()) as conn:
            with conn:
                cursor = conn.execute(
                    "INSERT INTO question (question_text, submitted_at) VALUES (?, ?)",
                    (question_text.strip(), utc_now()),
                )
            return cursor.lastrowid

    # Records the Manager's (already capped) sub-questions. Returns a dict from the
    # plan's own sub_question_id to the database sub_question_id.
    def add_sub_questions(self, question_id, plan):
        if not isinstance(plan, ManagerPlan):
            plan = ManagerPlan.model_validate(plan)
        mapping = {}
        try:
            with closing(self.connect()) as conn:
                with conn:
                    for sub_question in plan.sub_questions:
                        cursor = conn.execute(
                            "INSERT INTO sub_question (question_id, plan_number, text) VALUES (?, ?, ?)",
                            (question_id, sub_question.sub_question_id, sub_question.text),
                        )
                        mapping[sub_question.sub_question_id] = cursor.lastrowid
        except sqlite3.IntegrityError as error:
            raise EvidenceStoreError(f"could not save sub-questions: {error}") from error
        return mapping

    # ----- evidence (SDD 6.2, SRS 3.3) -----

    # Stores one passage and returns its evidence_id. record.sub_question_id is the
    # plan number, which must already have been added for this question.
    def add_evidence(self, question_id, record):
        if not isinstance(record, EvidenceRecord):
            record = EvidenceRecord.model_validate(record)
        with closing(self.connect()) as conn:
            row = conn.execute(
                "SELECT sub_question_id FROM sub_question WHERE question_id = ? AND plan_number = ?",
                (question_id, record.sub_question_id),
            ).fetchone()
            if row is None:
                raise EvidenceStoreError(
                    f"question {question_id} has no sub-question numbered {record.sub_question_id}"
                )
            with conn:
                cursor = conn.execute(
                    "INSERT INTO evidence (sub_question_id, source_url, source_title, published_date, "
                    "passage, retrieved_at, source_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (row[0], record.source_url, record.source_title, record.published_date or None,
                     record.passage, record.retrieved_at, record.source_type.value),
                )
            return cursor.lastrowid

    # Resolves evidence ids to their rows. Ids that do not exist are left out; callers
    # that must have every id (the Synthesizer does) compare what comes back.
    def read_evidence(self, evidence_ids):
        ids = []
        for evidence_id in evidence_ids:
            ids.append(int(evidence_id))
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        query = (
            "SELECT evidence_id, source_type, source_title, source_url, published_date, passage "
            f"FROM evidence WHERE evidence_id IN ({marks}) ORDER BY evidence_id"
        )
        with closing(self.connect()) as conn:
            found = conn.execute(query, ids).fetchall()
        rows = []
        for item in found:
            rows.append(StoredEvidence(
                evidence_id=item[0],
                source_type=item[1],
                source_title=item[2],
                source_url=item[3],
                published_date=item[4] or "",
                passage=item[5],
            ))
        return rows

    # Groups a question's evidence ids by plan number, for assembling SearchResults.
    # A plan number with no evidence is absent; whether that means empty or failed is
    # known only to the Search agent.
    def evidence_ids_by_plan_number(self, question_id):
        grouped = {}
        with closing(self.connect()) as conn:
            found = conn.execute(
                "SELECT s.plan_number, e.evidence_id FROM evidence e "
                "JOIN sub_question s ON s.sub_question_id = e.sub_question_id "
                "WHERE s.question_id = ? ORDER BY s.plan_number, e.evidence_id",
                (question_id,),
            ).fetchall()
        for plan_number, evidence_id in found:
            grouped.setdefault(plan_number, []).append(evidence_id)
        return grouped

    # ----- claims and verdicts (SDD 6.3, 6.4; SRS 3.5.6) -----

    # Persists every claim of one loop iteration with its verdict, reason, restatement
    # and cited evidence, in one transaction. Returns the new claim_ids in claim order.
    # Refuses a draft the Fact-Checker did not fully check (req. 3.6.4) and any
    # citation of evidence that belongs to a different question.
    def save_claims(self, question_id, iteration, draft, verdict):
        if iteration < 1:
            raise EvidenceStoreError("iteration starts at 1")
        if not verdict.covers(draft):
            raise EvidenceStoreError("verdict does not cover every claim in the draft")
        checks = {}
        for check in verdict.checks:
            checks[check.claim_number] = check
        cited = sorted(set(draft.cited_evidence_ids()))
        marks = ",".join("?" * len(cited))
        claim_ids = []
        with closing(self.connect()) as conn:
            owned = conn.execute(
                "SELECT COUNT(*) FROM evidence e JOIN sub_question s "
                "ON s.sub_question_id = e.sub_question_id "
                f"WHERE s.question_id = ? AND e.evidence_id IN ({marks})",
                [question_id] + cited,
            ).fetchone()[0]
            if owned != len(cited):
                raise EvidenceStoreError("draft cites evidence that does not belong to this question")
            try:
                with conn:
                    for claim in draft.all_claims():
                        check = checks[claim.claim_number]
                        cursor = conn.execute(
                            "INSERT INTO claim (question_id, iteration, claim_number, sentence, "
                            "verdict, reason, restatement) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (question_id, iteration, claim.claim_number, claim.sentence,
                             check.verdict.value, check.reason, check.restatement or None),
                        )
                        claim_id = cursor.lastrowid
                        claim_ids.append(claim_id)
                        for evidence_id in sorted(set(claim.evidence_ids)):
                            conn.execute(
                                "INSERT INTO claim_evidence (claim_id, evidence_id) VALUES (?, ?)",
                                (claim_id, evidence_id),
                            )
            except sqlite3.IntegrityError as error:
                raise EvidenceStoreError(f"could not save claims: {error}") from error
        return claim_ids

    # The highest iteration with saved claims for a question, or None
    def latest_iteration(self, question_id):
        with closing(self.connect()) as conn:
            row = conn.execute(
                "SELECT MAX(iteration) FROM claim WHERE question_id = ?", (question_id,)
            ).fetchone()
        return row[0]

    # Reads saved claims back, with their cited evidence ids, so a verdict can be
    # traced after the run (req. 3.5.6). iteration=None means every iteration.
    def get_claims(self, question_id, iteration=None):
        query = (
            "SELECT claim_id, iteration, claim_number, sentence, verdict, reason, restatement "
            "FROM claim WHERE question_id = ?"
        )
        params = [question_id]
        if iteration is not None:
            query += " AND iteration = ?"
            params.append(iteration)
        query += " ORDER BY iteration, claim_number"
        claims = []
        with closing(self.connect()) as conn:
            for row in conn.execute(query, params).fetchall():
                cited = conn.execute(
                    "SELECT evidence_id FROM claim_evidence WHERE claim_id = ? ORDER BY evidence_id",
                    (row[0],),
                ).fetchall()
                evidence_ids = []
                for item in cited:
                    evidence_ids.append(item[0])
                claims.append({
                    "claim_id": row[0], "iteration": row[1], "claim_number": row[2],
                    "sentence": row[3], "verdict": row[4], "reason": row[5],
                    "restatement": row[6] or "", "evidence_ids": evidence_ids,
                })
        return claims

    # Recomputes the support score (req. 3.6.7) from saved claims alone, so the
    # evaluation harness never has to re-run the pipeline (req. 3.9.4). Defaults to
    # the latest iteration. None when nothing was verifiable.
    def support_score(self, question_id, iteration=None):
        if iteration is None:
            iteration = self.latest_iteration(question_id)
        if iteration is None:
            return None
        verifiable = 0
        supported = 0
        for claim in self.get_claims(question_id, iteration):
            if claim["verdict"] == Verdict.NOT_VERIFIABLE.value:
                continue
            verifiable += 1
            if claim["verdict"] == Verdict.SUPPORTED.value:
                supported += 1
        if verifiable == 0:
            return None
        return round(100.0 * supported / verifiable, 1)
