"""Streamlit UI (requirement 1).

Flow: Upload Images / Take Photo -> Preview -> Add/Remove -> Process Documents
(explicit user action, nothing auto-processes) -> Q&A chat below, once
documents are indexed.

Run with: streamlit run app/ui/streamlit_app.py
"""
import sys
import uuid
from pathlib import Path

import streamlit as st

# Allow running via `streamlit run app/ui/streamlit_app.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.memory.session_memory import SessionMemory

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")


def _init_state():
    if "images" not in st.session_state:
        st.session_state.images = []  # list of {id, filename, bytes}
    if "camera_key" not in st.session_state:
        st.session_state.camera_key = 0
    if "documents_indexed" not in st.session_state:
        st.session_state.documents_indexed = False
    if "memory" not in st.session_state:
        st.session_state.memory = SessionMemory()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {question, answer, sources}
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


def _get_pipeline():
    if st.session_state.pipeline is None:
        from app.pipeline import DocumentQAPipeline
        with st.spinner("Starting up..."):
            st.session_state.pipeline = DocumentQAPipeline()
    return st.session_state.pipeline


def _add_image(file_bytes: bytes, filename: str):
    from app.config import settings
    if len(st.session_state.images) >= settings.MAX_IMAGES:
        st.warning(f"Maximum {settings.MAX_IMAGES} images reached.")
        return
    st.session_state.images.append({"id": str(uuid.uuid4()), "filename": filename, "bytes": file_bytes})


def _remove_image(image_id: str):
    st.session_state.images = [img for img in st.session_state.images if img["id"] != image_id]


def render_capture_section():
    st.subheader("1. Add Documents")
    tab_upload, tab_camera = st.tabs(["📁 Upload Images", "📷 Take Photo"])

    with tab_upload:
        from app.config import settings
        uploaded_files = st.file_uploader(
            "Upload up to 20 images",
            type=settings.ALLOWED_IMAGE_TYPES,
            accept_multiple_files=True,
            key="uploader",
        )
        if uploaded_files:
            existing_names = {img["filename"] for img in st.session_state.images}
            for f in uploaded_files:
                if f.name not in existing_names:
                    _add_image(f.getvalue(), f.name)

    with tab_camera:
        st.caption("On mobile this will request camera access and default to the rear camera.")
        photo = st.camera_input("Capture a document", key=f"camera_{st.session_state.camera_key}")
        if photo is not None:
            filename = f"capture_{st.session_state.camera_key}.jpg"
            _add_image(photo.getvalue(), filename)
            st.session_state.camera_key += 1  # reset widget so the next shot is a fresh capture
            st.rerun()


def render_preview_section():
    images = st.session_state.images
    if not images:
        return

    st.subheader(f"2. Preview ({len(images)} image{'s' if len(images) != 1 else ''})")
    cols = st.columns(3)
    for idx, img in enumerate(images):
        with cols[idx % 3]:
            st.image(img["bytes"], caption=img["filename"], use_container_width=True)
            if st.button("Remove", key=f"remove_{img['id']}"):
                _remove_image(img["id"])
                st.rerun()


def render_process_section():
    if not st.session_state.images:
        return

    st.subheader("3. Process")
    if st.button("Process Documents", type="primary"):
        pipeline = _get_pipeline()
        images_payload = [(img["bytes"], img["filename"]) for img in st.session_state.images]
        with st.spinner("Running OCR, chunking, and indexing..."):
            summary = pipeline.process_documents(images_payload)
        st.session_state.documents_indexed = True
        st.success(
            f"Indexed {summary['chunks_indexed']} chunks from {summary['documents_processed']} document(s)."
        )


def render_qa_section():
    if not st.session_state.documents_indexed:
        return

    st.subheader("4. Ask Questions")

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["sources"]:
                st.caption("Source:\n" + turn["sources"])

    question = st.chat_input("Ask a question about your documents...")
    if question:
        pipeline = _get_pipeline()
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.answer_question(question, st.session_state.memory)
            st.write(result.answer)
            if result.sources:
                st.caption("Source:\n" + result.sources)

        st.session_state.chat_history.append({
            "question": question,
            "answer": result.answer,
            "sources": result.sources,
        })


def main():
    _init_state()
    st.title("📄 Image Q&A")
    st.caption("Upload or capture documents, then ask questions grounded strictly in their content.")

    render_capture_section()
    render_preview_section()
    render_process_section()
    render_qa_section()


if __name__ == "__main__":
    main()
