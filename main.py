import json
import os
import jsonlines
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

ARTICLES_PATH = Path("LLM_classification_output_formatted.json")
ANSWERS_PATH = Path("w_q0_sparse_final_answers.json")
OUTPUT_PATH = Path("sentence_citation_labels_v2.2_10.json")
MODEL_NAME = "ggml-org/gemma-4-26B-A4B-it-GGUF"
LLAMA_CPP_HOST = os.getenv("LLAMA_CPP_HOST", "localhost:11434")
MAX_QUESTIONS = 10


class CitationLabel(BaseModel):
    justification: str
    label: int


client = OpenAI(
    api_key="--- IGNORE ---",
    base_url=f"http://{LLAMA_CPP_HOST}/v1",
)


def load_articles_by_uri(path: Path) -> dict[str, str]:
    with path.open("r") as f:
        articles = json.load(f)
    return {str(article["uri"]): article["answer"] for article in articles}


def get_label(topic: str, question: str, sentence: str, abstract: str) -> CitationLabel:
    prompt = f"""
Instruction: Identify the main factual claim expressed by the SENTENCE. Focus on the
meaning of the claim, not the specific wording used.
Determine whether the PASSAGE meaningfully addresses this specific claim.
If the passage does not address the claim, label it irrelevant.
If it is relevant, then classify the relationship between the SENTENCE and the PASSAGE as either
supporting or contradicting.

Use exactly ONE of these labels:
- supporting (1): the passage contains information that helps support the sentence and is relevant to the patient's concern in the narrative.
- contradicting (0): The passage provides information that conflicts with, disagrees with, refutes, or provides evidence against the sentence.
- irrelevant (2): The passage does not meaningfully address the claim
  made in the sentence. It may be related to the general topic, but it
  does not provide evidence either supporting or contradicting the sentence.

Important distinction:
Do NOT classify a passage as contradicting merely because it is irrelevant,
does not mention the sentence, or does not provide enough information to
support the sentence. Contradiction requires evidence in the passage that
conflicts with the claim made in the sentence.

Rules:
- Be strict and conservative.
- supporting requires information in the passage that is consistent with the
  statement expressed in the sentence.
- contradicting requires information in the passage that conflicts with the
  statement expressed in the sentence.
- If the passage is unrelated to the claim, classify it as irrelevant.
- Use only the provided information.
- Return a short justification explaining the label choice by referring to the
  statement expressed in the sentence.
- Output exactly the requested structured response.

Topic: {topic}
Original Question: {question}

Sentence:
{sentence}

PASSAGE:
{abstract}
"""
    response = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format=CitationLabel,
    )
    return response.choices[0].message.parsed


def build_sentence_citation_labels(max_questions: int = MAX_QUESTIONS) -> list[dict]:
    articles_by_uri = load_articles_by_uri(ARTICLES_PATH)

    with ANSWERS_PATH.open("r") as f:
        items = json.load(f)

    rows: list[dict] = []
    with jsonlines.open(OUTPUT_PATH, mode="a") as writer:
        for item in tqdm(items[:max_questions], total=max_questions):
            for sentence_entry in item.get("response_sentences", []):
                sentence_text = sentence_entry.get("text", "").strip()
                references = item.get("references", [])

                for citation in tqdm(references):
                    citation_uri = str(citation)
                    abstract = articles_by_uri.get(citation_uri)
                    if abstract is None:
                        continue

                    parsed = get_label(
                        topic=item.get("topic", ""),
                        question=item.get("question", ""),
                        sentence=sentence_text,
                        abstract=abstract,
                    )

                    writer.write(
                        {
                            "topic_id": item.get("topic_id"),
                            "topic": item.get("topic"),
                            "question": item.get("question"),
                            "sentence": sentence_text,
                            "citation": citation_uri,
                            "abstract": abstract,
                            "label": (
                                "supporting"
                                if parsed.label == 1
                                else "contradicting"
                                if parsed.label == 0
                                else "irrelevant"
                            ),
                            "justification": parsed.justification,
                        }
                    )

    return rows


def main() -> None:
    rows = build_sentence_citation_labels()
    with OUTPUT_PATH.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} sentence-citation labels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
