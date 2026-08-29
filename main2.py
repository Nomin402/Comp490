import json
import os
import re
from pathlib import Path

import jsonlines
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

ARTICLES_PATH = Path("LLM_classification_output_formatted.json")
ANSWERS_PATH = Path("w_q0_sparse_final_answers.json")
OUTPUT_PATH = Path("unused_citation_labels_v3.0_gemma.json")
MODEL_NAME_GEMMA = "ggml-org/gemma-4-26B-A4B-it-GGUF"
LLAMA_CPP_HOST = os.getenv("LLAMA_CPP_HOST", "localhost:11434")


class CitationLabel(BaseModel):
    evidence_rationale: str
    label: str


client = OpenAI(
    api_key="--- IGNORE ---",
    base_url=f"http://{LLAMA_CPP_HOST}/v1",
)


def load_articles_by_uri(path: Path) -> dict[str, str]:
    with path.open("r") as f:
        articles = json.load(f)
    return {str(article["uri"]): article["answer"] for article in articles}


def extract_final_answer_citations(final_answer_with_citations: str) -> set[str]:
    citation_ids: set[str] = set()
    for match in re.findall(r"\[(.*?)\]", final_answer_with_citations):
        for citation in match.split(","):
            citation_id = citation.strip()
            if citation_id:
                citation_ids.add(citation_id)
    return citation_ids


def extract_used_citations(item: dict) -> set[str]:
    used_citations = extract_final_answer_citations(item.get("final_answer_with_citations", ""))
    if used_citations:
        return used_citations

    fallback_citations: set[str] = set()
    for sentence_entry in item.get("response_sentences", []):
        for citation in sentence_entry.get("citations", []):
            citation_id = str(citation).strip()
            if citation_id:
                fallback_citations.add(citation_id)
    return fallback_citations


def get_label(answer: str, abstract: str) -> CitationLabel:
    prompt = f"""
System: You are an expert biomedical researcher and clinical NLP evaluator. Your task is to determine whether a retrieved abstract provides evidence relevant to the generated answer.

Task:
1. Read the GENERATED ANSWER and identify its factual claims.
2. Evaluate whether the ABSTRACT provides direct evidence for, directly contradicts, or is unrelated to any factual claim in the GENERATED ANSWER.
3. Classify the relationship into exactly ONE label:
   - "supporting": The abstract provides direct evidence supporting at least one factual claim in the answer.
   - "contradicting": The abstract directly contradicts at least one factual claim in the answer.
   - "irrelevant": The abstract does not directly support or contradict any factual claim in the answer.

Rules & Biomedical Nuance:
- Use only explicit evidence stated in the abstract.
- Do not infer support from general topical overlap.
- If the abstract is about the same topic but does not address a claim made in the answer, classify it as "irrelevant".

GENERATED ANSWER:
{answer}

ABSTRACT:
{abstract}

Output Format:
Output ONLY a valid JSON object matching the following structure:
{{
  "evidence_rationale": "<1-2 sentences explaining the relationship to the generated answer>",
  "label": "<supporting | contradicting | irrelevant>"
}}
"""
    response = client.chat.completions.parse(
        model=MODEL_NAME_GEMMA,
        messages=[{"role": "user", "content": prompt}],
        response_format=CitationLabel,
    )
    return response.choices[0].message.parsed


def build_unused_citation_labels(
    start_question: int = 0,
    end_question: int | None = None,
) -> list[dict]:
    articles_by_uri = load_articles_by_uri(ARTICLES_PATH)

    with ANSWERS_PATH.open("r") as f:
        items = json.load(f)

    rows: list[dict] = []
    total_references = 0
    used_references = 0
    unused_references = 0

    selected_items = items[start_question:end_question] if end_question is not None else items[start_question:]

    with jsonlines.open(OUTPUT_PATH, mode="a") as writer:
        for item in tqdm(
            selected_items,
            total=len(selected_items),
            desc="Questions",
        ):
            topic_id = item.get("topic_id", "")
            topic = item.get("topic", "")
            question = item.get("question", "")
            generated_answer = item.get("final_answer_text", "")
            used_citations = extract_used_citations(item)
            references = item.get("references", [])

            total_references += len(references)

            for citation in references:
                citation_uri = str(citation)
                if citation_uri in used_citations:
                    used_references += 1
                    continue

                abstract = articles_by_uri.get(citation_uri)
                if abstract is None:
                    continue

                unused_references += 1

                parsed = get_label(
                    answer=generated_answer,
                    abstract=abstract,
                )
                annotation = {
                    "topic_id": topic_id,
                    "topic": topic,
                    "question": question,
                    "answer": generated_answer,
                    "citation": citation_uri,
                    "abstract": abstract,
                    "evidence_rationale": parsed.evidence_rationale,
                    "label": parsed.label,
                    "used_in_final_answer": False,
                }
                writer.write(annotation)
                rows.append(annotation)

    print(
        f"Evaluated {total_references} retrieved citations; "
        f"{used_references} were already used in the final answer; "
        f"{unused_references} were labeled as unused citations."
    )

    return rows


def main() -> None:
    rows = build_unused_citation_labels()
    print(
        f"Wrote {len(rows)} unused-citation labels to {OUTPUT_PATH} "
        f"(unused citations are references not present in the final answer)"
    )


if __name__ == "__main__":
    main()
