import json
import os
import jsonlines
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

ARTICLES_PATH = Path("LLM_classification_output_formatted.json")
ANSWERS_PATH = Path("w_q0_sparse_final_answers.json")
OUTPUT_PATH = Path("sentence_citation_labels_v3.0_qwen_sentence_citation.json")
MODEL_NAME_GEMMA = "ggml-org/gemma-4-26B-A4B-it-GGUF"
MODEL_NAME_QWEN = "Qwen/Qwen2.5-32B-Instruct-GGUF:Q5_K_M"
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


def get_label(sentence: str, abstract: str,) -> CitationLabel:
    prompt = f"""
System: You are an expert biomedical researcher and clinical NLP evaluator. Your task is to perform strict evidence verification for biomedical claims against reference scientific passages (e.g., PubMed/clinical abstracts).

Task:
1. Extract the core biomedical claim from the SENTENCE.
2. Evaluate whether the reference PASSAGE provides direct scientific evidence regarding this specific biomedical claim.
3. Classify the relationship into exactly ONE label:
   - "supporting": The passage contains explicit findings, experimental data, or clinical evidence that confirms the claim.
   - "contradicting": The passage explicitly refutes, disproves, or reports statistically significant/clinical findings conflicting with the claim.
   - "irrelevant": The passage does not directly test, confirm, or refute the claim (even if it shares medical keywords, diseases, or target genes), or the evidence is inconclusive/unreported.

Rules & Biomedical Nuance:
- Scientific Rigor: Base decisions strictly on the stated findings. Do not infer clinical efficacy from in vitro/animal findings unless the sentence specifies that scope.
- Absence of Proof != Disproof: If a passage fails to test or mention an effect, classify as "irrelevant", NOT "contradicting".
- Mechanism vs. Association: Differentiate between direct causation/mechanism and loose topical correlation.

Inputs:
SENTENCE:
{sentence}

PASSAGE:
{abstract}

Output Format:
Output ONLY a valid JSON object matching the following structure:
{{
  "evidence_rationale": "<1-2 sentences identifying explicit clinical/biological evidence from the passage>",
  "label": "<supporting | contradicting | irrelevant>"
}}

"""
    response = client.chat.completions.parse(
        model=MODEL_NAME_GEMMA,
        messages=[{"role": "user", "content": prompt}],
        response_format=CitationLabel,
    )
    return response.choices[0].message.parsed


def build_sentence_citation_labels(start_question: int,
    end_question: int) -> list[dict]:
    articles_by_uri = load_articles_by_uri(ARTICLES_PATH)

    with ANSWERS_PATH.open("r") as f:
        items = json.load(f)

    rows: list[dict] = []

    with jsonlines.open(OUTPUT_PATH, mode="a") as writer:
        for item in tqdm(items[start_question:end_question], total=end_question - start_question, desc="Questions"):
            topic_id = item.get("topic_id", "")
            topic = item.get("topic", "")
            question = item.get("question", "")
            references = item.get("references", [])

            #Every sentence in the response sentences
            for sentence_entry in tqdm(item.get("response_sentences", []), desc="Sentences"):
                sentence_text = sentence_entry.get("text", "").strip()

                #citations = sentence_entry.get("citations", [])

                for citation in references:
                    citation_uri = str(citation)
                    abstract = articles_by_uri.get(citation_uri)
                    if abstract is None:
                        continue

                    parsed = get_label(
                        sentence=sentence_text,
                        abstract=abstract,
                    )
                    annotation = {
                        "topic_id": topic_id,
                        "topic": topic,
                        "question": question,
                        "sentence": sentence_text,
                        "citation": citation_uri,
                        "abstract": abstract,
                        "evidence_rationale": parsed.evidence_rationale,
                        "label": parsed.label,
                    }
                    writer.write(annotation)
                    rows.append(annotation)

    return rows


def main() -> None:
    rows = build_sentence_citation_labels(0, 30)
    print(f"Wrote {len(rows)} sentence-citation labels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
