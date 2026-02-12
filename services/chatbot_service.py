import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from services.rag_service import retrieve_context

MODEL_NAME = "google/flan-t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading RAG chatbot model (FLAN-T5 Small)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

print("Chatbot model loaded successfully.")


def generate_medical_reply(question: str) -> str:
    """
    Standard RAG flow:
    1. Retrieve factual context
    2. Ask LLM to answer ONLY from that context
    """

    context = retrieve_context(question)

    prompt = (
        "Answer the question using ONLY the information provided below.\n"
        "If the answer is not present, say 'Information not available.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.0   # fully deterministic
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return (
        answer.strip()
        + "\n\n📌 Source: medical_knowledge.txt"
        + "\n\n⚠️ Disclaimer: This is AI-assisted information, not a medical diagnosis."
    )
