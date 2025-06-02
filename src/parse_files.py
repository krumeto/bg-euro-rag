import json
import re

from pypdf import PdfReader

from model2vec import StaticModel
import numpy as np

def parse_qa_file(filepath: str) -> list[str]:
    """
    Reads a Q&A file and returns a list of strings,
    where each element is formatted as:
        "{question}\n{answer}"

    Assumes that:
    - Questions are on their own line and end with a question mark '?'
    - The answer includes everything between this question and the next question
      (or the end of the file), including all lines and blank lines
    """
    qa_pairs = []
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    current_question = None
    current_answer_lines = []

    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.endswith('?'):
            if current_question is not None:
                answer = "\n".join(current_answer_lines).strip()
                qa_pairs.append(f"{current_question}\n{answer}")
                current_answer_lines = []
            current_question = stripped
        else:
            if current_question is not None:
                current_answer_lines.append(stripped)

    if current_question is not None:
        answer = "\n".join(current_answer_lines).strip()
        qa_pairs.append(f"{current_question}\n\n{answer}")

    return qa_pairs

def extract_bnb_law(pdf_path: str) -> list[str]:
    """
    Read the entire text from a PDF file, remove page headers, and split each
    separate 'Член' (Article), as well as the 'Преходни и заключителни разпоредби'
    and 'Допълнителни разпоредби', into separate list elements. Returns a list of
    strings, each containing the full text of one section.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines = text.splitlines()
        filtered_lines = [
            line for line in lines
            if not line.strip().isdigit() and line.strip() != "Закон за Българската народна банка"
        ]
        pages.append("\n".join(filtered_lines))
    combined_text = "\n".join(pages)
    pattern = (
        r'(?=Чл\.\s*\d+\.?|'
        r'Преходни и заключителни разпоредби|'
        r'Допълнителни разпоредби)'
    )
    raw_chunks = re.split(pattern, combined_text)
    sections = [
        chunk.strip()
        for chunk in raw_chunks
        if chunk.strip().startswith("Чл.")
           or chunk.strip().startswith("Преходни и заключителни разпоредби")
           or chunk.strip().startswith("Допълнителни разпоредби")
    ]
    return sections

def load_model(model_name: str) -> StaticModel: 
    """
    Load a static model for vector embeddings.
    """
    return StaticModel.from_pretrained(model_name)

def embed_texts(texts: list[str], model: StaticModel) -> np.ndarray:
    """
    Embed a list of texts using the provided static model.
    Returns a numpy array of embeddings.
    """
    return model.encode(texts)

def save_embeddings(embeddings: np.ndarray, filepath: str) -> None:
    """
    Save embeddings to a file.
    """
    np.save(filepath, embeddings)

if __name__ == "__main__":
    MODEL_NAME = "minishlab/potion-multilingual-128M"
    QA_FILEPATH = "src/resources/q_and_a.txt"
    BNB_FILEPATH = "src/resources/newbnblaw_bg.pdf"
    EMBEDDINGS_FILEPATH = "src/resources/"

    model = load_model(MODEL_NAME)

    qa_pairs = parse_qa_file(QA_FILEPATH)
    qa_embeddings = embed_texts(qa_pairs, model)
    save_embeddings(qa_embeddings, EMBEDDINGS_FILEPATH + "q_and_a_embeddings.npy")
    with open(EMBEDDINGS_FILEPATH+ "q_and_a_texts.json", "w", encoding="utf-8") as outf:
        json.dump(qa_pairs, outf, ensure_ascii=False, indent=2)

    print(f"Processed {len(qa_pairs)} Q&A pairs, saved embeddings to {EMBEDDINGS_FILEPATH}, "
          f"and saved texts to src/resources/q_and_a_texts.json.")

    bnb_law_sections = extract_bnb_law(BNB_FILEPATH)
    bnb_law_embeddings = embed_texts(bnb_law_sections, model)
    save_embeddings(bnb_law_embeddings, EMBEDDINGS_FILEPATH + "bnb_law_embeddings.npy")
    with open(EMBEDDINGS_FILEPATH + "bnb_law_texts.json", "w", encoding="utf-8") as outf:
        json.dump(bnb_law_sections, outf, ensure_ascii=False, indent=2)

    print(f"Processed {len(bnb_law_sections)} BNB law sections, saved embeddings to {EMBEDDINGS_FILEPATH}, "
          f"and saved texts to src/resources/bnb_law_texts.json.")
