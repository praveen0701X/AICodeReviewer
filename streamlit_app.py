#!/usr/bin/env python3
# streamlit_app.py

import os
import io
import re
import sys
import json
import math
import time
import shutil
import textwrap
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple

import streamlit as st

# Optional imports handled dynamically
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from pygments.lexers import guess_lexer
    from pygments.util import ClassNotFound
    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False

# -----------------------
# Utilities and constants
# -----------------------

EXT_LANG_MAP = {
    ".py": "python",
    ".c": "c",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".java": "java",
}

PY_FLAKE8_SUGGESTIONS = {
    "F401": "Imported but unused: remove the import or use it. If it's for side-effects, consider `# noqa: F401`.",
    "F821": "Undefined name: check for typos or define the variable/function before use.",
    "F841": "Local variable assigned but not used: remove it or use it.",
    "E225": "Missing whitespace around operator: add spaces for readability.",
    "E231": "Missing whitespace after comma/semicolon/colon.",
    "E302": "Expected 2 blank lines between top-level definitions.",
    "E305": "Expected 2 blank lines after class or function definition.",
    "E501": "Line too long: break long lines, wrap strings, or use implicit concatenation.",
    "W291": "Trailing whitespace: remove it.",
    "E701": "Multiple statements on one line: split into separate lines.",
    "E999": "Syntax error: fix the syntax on the reported line.",
}

C_CPP_SUGGESTIONS_PATTERNS = [
    (re.compile(r"unused variable|[-]Wunused-variable"), "Remove the variable or explicitly mark it as unused (e.g., (void)var or [[maybe_unused]])."),
    (re.compile(r"unused parameter|[-]Wunused-parameter"), "Omit the parameter name, mark it unused, or use it in the function."),
    (re.compile(r"implicit declaration of function"), "Include the proper header for the function or declare it before use."),
    (re.compile(r"control reaches end of non-void function"), "Ensure all code paths return a value."),
    (re.compile(r"comparison between signed and unsigned"), "Use consistent signedness or cast explicitly to avoid issues."),
    (re.compile(r"may be used uninitialized"), "Initialize the variable before use."),
    (re.compile(r"format specifies type .* but argument has type"), "Fix printf/scanf format specifiers to match argument types."),
    (re.compile(r"no return statement in function returning non-void"), "Add a proper return statement."),
    (re.compile(r"redefinition of|previous definition of"), "Avoid duplicate definitions; consolidate or change scope."),
]

JAVA_SUGGESTIONS_PATTERNS = [
    (re.compile(r"cannot find symbol"), "Check imports, spelling, or that the symbol is defined and visible."),
    (re.compile(r"incompatible types"), "Adjust variable types or cast safely; ensure generics are correct."),
    (re.compile(r"unchecked conversion|unchecked call"), "Use generics properly or add @SuppressWarnings(\"unchecked\") with justification."),
    (re.compile(r"missing return statement"), "Add a return statement for all code paths."),
    (re.compile(r"variable .* might not have been initialized"), "Initialize the variable before use."),
    (re.compile(r"reached end of file while parsing"), "Check for missing braces or syntax issues."),
]

def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run(cmd: List[str], input_text: Optional[str] = None, cwd: Optional[str] = None, timeout: int = 30) -> Tuple[int, str, str]:
    env = os.environ.copy()
    env["TERM"] = "dumb"
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    try:
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            cwd=cwd,
            timeout=timeout,
            env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, "", f"Timeout after {timeout}s running: {' '.join(cmd)}"
    except Exception as e:
        return 1, "", f"Failed to run {' '.join(cmd)}: {e}"

def detect_language(code: str, filename: Optional[str]) -> str:
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in EXT_LANG_MAP:
            return EXT_LANG_MAP[ext]

    code_snippet = (code or "")[:1000]

    if HAS_PYGMENTS and code_snippet.strip():
        try:
            lex = guess_lexer(code_snippet)
            alias = (lex.aliases[0] if getattr(lex, "aliases", []) else "").lower()
            if alias in {"python"}:
                return "python"
            if alias in {"c"}:
                return "c"
            if alias in {"cpp", "c++"}:
                return "cpp"
            if alias in {"java"}:
                return "java"
        except ClassNotFound:
            pass
        except Exception:
            pass

    if "#include" in code_snippet:
        if "std::" in code_snippet or "using namespace std" in code_snippet or ".hpp" in code_snippet:
            return "cpp"
        return "c"
    if "public class" in code_snippet or "System.out.println" in code_snippet:
        return "java"
    if "def " in code_snippet or "import " in code_snippet or "print(" in code_snippet:
        return "python"
    return "python"

def gcc_like_parse(stderr: str) -> List[Dict]:
    diags = []
    pat = re.compile(r"^(.*?):(\d+):(?:(\d+):)?\s*(fatal error|error|warning|note):\s*(.*)$")
    for line in stderr.splitlines():
        m = pat.match(line.strip())
        if m:
            _file, ln, col, sev, msg = m.groups()
            ln = int(ln)
            col = int(col) if col else None
            rule = None
            m2 = re.search(r"\[-W([^\]]+)\]", msg)
            if m2:
                rule = f"-W{m2.group(1)}"
            diags.append({
                "line": ln,
                "col": col,
                "severity": "error" if "error" in sev else ("warning" if "warning" in sev else "note"),
                "message": msg.strip(),
                "rule": rule,
            })
    return diags

def javac_parse(stderr: str) -> List[Dict]:
    diags = []
    lines = stderr.splitlines()
    pat = re.compile(r"^(.*?):(\d+):\s*(error|warning):\s*(.*)$")
    i = 0
    n = len(lines)
    while i < n:
        m = pat.match(lines[i].strip())
        if m:
            _file, ln, sev, msg = m.groups()
            ln = int(ln)
            col = None
            if i + 2 < n:
                caret_line = lines[i + 2]
                caret_pos = caret_line.find("^")
                if caret_pos >= 0:
                    col = caret_pos + 1
            diags.append({
                "line": ln,
                "col": col,
                "severity": "error" if sev == "error" else "warning",
                "message": msg.strip(),
                "rule": None,
            })
            i += 3
        else:
            i += 1
    return diags

def flake8_parse(stdout: str) -> List[Dict]:
    diags = []
    pat = re.compile(r"^(.*?):(\d+):(\d+):\s*([A-Z]\d{3})\s+(.*)$")
    for line in stdout.splitlines():
        m = pat.match(line.strip())
        if m:
            _f, ln, col, code, msg = m.groups()
            diags.append({
                "line": int(ln),
                "col": int(col),
                "severity": "error" if code[0] in {"E", "F"} else "warning",
                "message": f"{code} {msg.strip()}",
                "rule": code,
            })
    return diags

def pyflakes_parse(stdout: str) -> List[Dict]:
    diags = []
    pat = re.compile(r"^(.*?):(\d+):\s*(.*)$")
    for line in stdout.splitlines():
        m = pat.match(line.strip())
        if m:
            _f, ln, msg = m.groups()
            sev = "error" if "undefined name" in msg.lower() or "syntax" in msg.lower() else "warning"
            diags.append({
                "line": int(ln),
                "col": None,
                "severity": sev,
                "message": msg.strip(),
                "rule": None,
            })
    return diags

def suggest_for_python(rule: Optional[str], message: str) -> Optional[str]:
    if not rule:
        low = message.lower()
        if "undefined name" in low:
            return PY_FLAKE8_SUGGESTIONS["F821"]
        if "syntax error" in low or "E999" in message:
            return PY_FLAKE8_SUGGESTIONS["E999"]
        return None
    return PY_FLAKE8_SUGGESTIONS.get(rule)

def suggest_for_c_cpp(message: str, rule: Optional[str]) -> Optional[str]:
    for pat, hint in C_CPP_SUGGESTIONS_PATTERNS:
        if pat.search(message):
            return hint
    return None

def suggest_for_java(message: str) -> Optional[str]:
    for pat, hint in JAVA_SUGGESTIONS_PATTERNS:
        if pat.search(message):
            return hint
    return None

def make_annotated_code(code: str, diags: List[Dict]) -> str:
    lines = normalize_newlines(code).split("\n")
    by_line: Dict[int, List[Dict]] = {}
    for d in diags:
        if d.get("line"):
            by_line.setdefault(d["line"], []).append(d)

    width = len(str(len(lines)))
    out = []
    for i, line in enumerate(lines, start=1):
        marker = ""
        if i in by_line:
            severities = {d["severity"] for d in by_line[i]}
            if "error" in severities:
                marker = "⛔"
            elif "warning" in severities:
                marker = "⚠️"
            else:
                marker = "•"
        prefix = f"{str(i).rjust(width)} | {marker} "
        out.append(prefix + line)
    return "\n".join(out)

def analyze_python(code: str, display_name: str = "input.py") -> Tuple[List[Dict], str, str]:
    # Try flake8 with module syntax first
    try:
        cmd = [sys.executable, "-m", "flake8", "-", "--stdin-display-name", display_name]
        rc, out, err = run(cmd, input_text=code)
        diags = flake8_parse(out)
        for d in diags:
            d["suggestion"] = suggest_for_python(d.get("rule"), d.get("message", ""))
        tool = "flake8"
        raw = out or err
        return diags, tool, raw
    except Exception:
        pass

    # Try direct flake8 command
    if which("flake8"):
        cmd = ["flake8", "-", "--stdin-display-name", display_name]
        rc, out, err = run(cmd, input_text=code)
        diags = flake8_parse(out)
        for d in diags:
            d["suggestion"] = suggest_for_python(d.get("rule"), d.get("message", ""))
        tool = "flake8"
        raw = out or err
        return diags, tool, raw

    # Fallback to pyflakes
    try:
        cmd = [sys.executable, "-m", "pyflakes", "-"]
        rc, out, err = run(cmd, input_text=code)
        diags = pyflakes_parse(out or err)
        for d in diags:
            d["suggestion"] = suggest_for_python(None, d.get("message", ""))
        tool = "pyflakes"
        raw = out or err
        return diags, tool, raw
    except Exception:
        pass

    # Last resort: SyntaxError via compile
    diags = []
    try:
        compile(code, display_name, "exec")
    except SyntaxError as e:
        diags.append({
            "line": e.lineno or 1,
            "col": e.offset or None,
            "severity": "error",
            "message": f"SyntaxError: {e.msg}",
            "rule": "E999",
            "suggestion": PY_FLAKE8_SUGGESTIONS["E999"],
        })
    return diags, "python built-in (compile)", ""

def analyze_c_cpp(code: str, language: str) -> Tuple[List[Dict], str, str]:
    compiler = "gcc" if language == "c" else "g++"
    if not which(compiler):
        return [{
            "line": 1,
            "col": None,
            "severity": "error",
            "message": f"{compiler} not found on PATH; install a C/C++ toolchain.",
            "rule": None,
            "suggestion": "Install GCC/Clang (Linux/Mac) or MinGW-w64/LLVM (Windows), then retry.",
        }], compiler, f"{compiler} not available."

    flags = ["-Wall", "-Wextra", "-fsyntax-only", "-Wno-unknown-pragmas"]
    std = ["-std=c11"] if language == "c" else ["-std=c++17"]
    with tempfile.TemporaryDirectory() as tmpd:
        fname = os.path.join(tmpd, f"main.{'c' if language=='c' else 'cpp'}")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(code)
        rc, out, err = run([compiler] + std + flags + [fname], cwd=tmpd)
    diags = gcc_like_parse(err)
    for d in diags:
        d["suggestion"] = suggest_for_c_cpp(d["message"], d.get("rule"))
    tool = compiler
    raw = err
    return diags, tool, raw

def analyze_java(code: str, original_name: Optional[str]) -> Tuple[List[Dict], str, str]:
    if not which("javac"):
        return [{
            "line": 1,
            "col": None,
            "severity": "error",
            "message": "javac not found on PATH; install a JDK.",
            "rule": None,
            "suggestion": "Install OpenJDK/Oracle JDK and ensure javac is on PATH.",
        }], "javac", "javac not available."

    m = re.search(r"public\s+class\s+([A-Za-z_]\w*)", code)
    filename = f"{m.group(1)}.java" if m else (original_name or "Main.java")

    with tempfile.TemporaryDirectory() as tmpd:
        fpath = os.path.join(tmpd, filename)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(code)
        cmd = ["javac", "-Xlint:all", "-encoding", "UTF-8", fpath]
        rc, out, err = run(cmd, cwd=tmpd)
        diags = javac_parse(err)
        for d in diags:
            d["suggestion"] = suggest_for_java(d["message"])
    return diags, "javac -Xlint:all", err

def analyze_code(language: str, code: str, filename: Optional[str]) -> Dict:
    if language == "python":
        diags, tool, raw = analyze_python(code, (filename or "input.py"))
    elif language in {"c", "cpp"}:
        diags, tool, raw = analyze_c_cpp(code, language)
    elif language == "java":
        diags, tool, raw = analyze_java(code, filename)
    else:
        diags, tool, raw = ([{
            "line": 1, "col": None, "severity": "error",
            "message": f"Unsupported language: {language}", "rule": None,
            "suggestion": None,
        }], "n/a", "")

    errors = [d for d in diags if d["severity"] == "error"]
    warnings = [d for d in diags if d["severity"] == "warning"]
    notes = [d for d in diags if d["severity"] == "note"]

    annotated = make_annotated_code(code, diags)
    report_lines = [
        f"Code Review Report",
        f"Language: {language}",
        f"Analyzer: {tool}",
        f"Errors: {len(errors)}, Warnings: {len(warnings)}, Notes: {len(notes)}",
        "",
        "Diagnostics:"
    ]
    for d in diags:
        loc = f"{d.get('line')}:{d.get('col') or 1}"
        rule = f" [{d.get('rule')}]" if d.get("rule") else ""
        report_lines.append(f"- {d['severity'].upper()} at {loc}{rule}: {d['message']}")
        if d.get("suggestion"):
            report_lines.append(f"  Suggestion: {d['suggestion']}")
    report_lines += ["", "Annotated Code:", annotated]
    report_text = "\n".join(report_lines)

    return {
        "language": language,
        "tool": tool,
        "diagnostics": diags,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "note_count": len(notes),
        "annotated_code": annotated,
        "raw_output": raw,
        "report_text": report_text,
    }

def ai_suggestions_block(language: str, code: str) -> Optional[str]:
    use_ai = st.session_state.get("use_ai", False)
    if not use_ai:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OPENAI_API_KEY not set; skipping AI suggestions.")
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = f"Review the following {language} code. List potential bugs, edge cases, and improvements in bullets:\n\n{code}"
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"AI suggestions unavailable: {e}")
        return None

def extract_text_from_pdf(uploaded_file) -> Tuple[str, Dict]:
    data = uploaded_file.read()
    meta = {"pages": None, "bytes": len(data)}
    text = ""
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            meta["pages"] = doc.page_count
            parts = []
            for p in doc:
                parts.append(p.get_text("text"))
            text = "\n".join(parts)
            doc.close()
            return text, meta
        except Exception as e:
            pass
    if HAS_PYPDF2:
        try:
            reader = PdfReader(io.BytesIO(data))
            meta["pages"] = len(reader.pages)
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            text = "\n".join(parts)
            return text, meta
        except Exception as e:
            pass
    return "", meta

def naive_sentence_split(text: str) -> List[str]:
    s = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [x.strip() for x in s if x.strip()]

def naive_extractive_summary(text: str, max_sentences: int = 5) -> str:
    sentences = naive_sentence_split(text)
    if not sentences:
        return ""
    scored = sorted(sentences, key=lambda s: len(s), reverse=True)
    chosen = sorted(scored[:max_sentences], key=lambda s: sentences.index(s))
    return " ".join(chosen)

def sumy_lexrank_summary(text: str, language: str = "english", max_sentences: int = 5) -> Optional[str]:
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lex_rank import LexRankSummarizer
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        summarizer = LexRankSummarizer()
        sents = summarizer(parser.document, max_sentences)
        return " ".join(str(s) for s in sents)
    except Exception:
        return None

def load_abstractive_pipeline(model_name: str):
    try:
        from transformers import pipeline
        return pipeline("summarization", model=model_name)
    except Exception as e:
        st.error(f"Failed to load transformers pipeline: {e}")
        return None

def chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind(".", start, end)
            if cut == -1 or cut < start + max_chars * 0.6:
                cut = end
        else:
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
        if cut == end and end == len(text):
            break
        start = cut - overlap if cut - overlap > start else cut
    chunks = [c for c in chunks if c]
    return chunks

def abstractive_summarize(text: str, model_name: str = "sshleifer/distilbart-cnn-12-6", max_len: int = 200, min_len: int = 60) -> Tuple[str, List[str]]:
    try:
        pipe = load_abstractive_pipeline(model_name)
        if pipe is None:
            return "[Abstractive summarization unavailable]", []
    except Exception as e:
        return f"[Abstractive model not available: {e}]", []
    
    chunks = chunk_text(text, max_chars=3500, overlap=200)
    partials = []
    for ch in chunks:
        try:
            out = pipe(ch, max_length=max_len, min_length=min_len, do_sample=False)
            partials.append(out[0]["summary_text"])
        except Exception as e:
            partials.append(f"[Chunk summarization failed: {e}]")
    
    summary_text = " ".join(partials)
    if len(partials) > 3:
        try:
            out = pipe(summary_text, max_length=max_len, min_length=min_len, do_sample=False)
            summary_text = out[0]["summary_text"]
        except Exception:
            pass
    return summary_text, partials

# -----------------------
# Streamlit UI
# -----------------------
# --- MODIFIED LINE 1: Changed page_icon to 💻 ---
st.set_page_config(page_title="🛠️ Code Review + PDF Summarizer", page_icon="💻", layout="wide")

if "use_ai" not in st.session_state:
    st.session_state["use_ai"] = False

# --- MODIFIED LINE 2: Changed sidebar title ---
st.sidebar.title("HOW CAN I HELP YOU?") 
# --- MODIFIED LINE 3: Changed Code Review radio option icon ---
page = st.sidebar.radio("Go to", ["🛠️ Code Review", "📘 PDF Summarizer"])

st.sidebar.markdown("---")
st.sidebar.checkbox("Enable AI suggestions (OpenAI)", key="use_ai", help="Requires OPENAI_API_KEY and openai package.")

st.sidebar.markdown("Made with ❤️ in Python")

def render_code_review():
    st.header("🧾 Multi-language Code Review & Error Detection")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload source file (.py, .c, .cpp, .java)", type=["py", "c", "cc", "cxx", "cpp", "java"])
    with col2:
        st.caption("…or paste code below")
        code_text = st.text_area("Paste code here", height=240, placeholder="Paste your code…")

    filename = uploaded.name if uploaded else None
    code_content = ""
    if uploaded:
        try:
            code_content = uploaded.read().decode("utf-8", errors="replace")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
    elif code_text.strip():
        code_content = code_text

    if not code_content:
        st.info("Upload a file or paste code to analyze.")
        return

    detected_lang = detect_language(code_content, filename)
    lang = st.selectbox("Detected language", ["python", "c", "cpp", "java"], index=["python", "c", "cpp", "java"].index(detected_lang))

    if st.button("Analyze Code", type="primary"):
        with st.spinner("Analyzing…"):
            result = analyze_code(lang, code_content, filename)

        c1, c2, c3 = st.columns(3)
        c1.metric("Errors", result["error_count"])
        c2.metric("Warnings", result["warning_count"])
        c3.metric("Notes", result["note_count"])

        st.subheader("Diagnostics")
        if result["diagnostics"]:
            rows = []
            for d in result["diagnostics"]:
                rows.append({
                    "Line": d.get("line"),
                    "Col": d.get("col"),
                    "Type": d.get("severity"),
                    "Rule": d.get("rule"),
                    "Message": d.get("message"),
                    "Suggestion": d.get("suggestion"),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.success("No issues found by the selected analyzer.")

        st.subheader("Annotated Code")
        st.text(result["annotated_code"])

        with st.expander("Raw tool output"):
            st.code(result["raw_output"] or "(no output)")

        ai_txt = ai_suggestions_block(lang, code_content)
        if ai_txt:
            st.subheader("🤖 AI Suggestions")
            st.write(ai_txt)

        st.download_button(
            "Download Report",
            data=result["report_text"],
            file_name=f"code_review_report_{lang}.txt",
            mime="text/plain",
        )

def render_pdf_summarizer():
    st.header("📘 PDF Summarizer")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if not uploaded_pdf:
        st.info("Upload a PDF to summarize.")
        return

    st.subheader("Summarization Options")
    mode = st.radio("Method", ["Extractive (LexRank / sumy)", "Abstractive (transformers)"], horizontal=True)

    if mode.startswith("Extractive"):
        language = st.selectbox("Language (for tokenizer)", ["english", "spanish", "french", "german", "italian"], index=0)
        max_sents = st.slider("Sentences in summary", min_value=3, max_value=15, value=7)
    else:
        model_name = st.selectbox("Transformer model", [
            "sshleifer/distilbart-cnn-12-6",
            "facebook/bart-base", 
            "google/pegasus-xsum",
            "t5-small",
        ], index=0)
        max_len = st.slider("Max summary length (tokens)", 64, 512, 200, step=16)
        min_len = st.slider("Min summary length (tokens)", 16, 256, 60, step=8)

    if st.button("Summarize PDF", type="primary"):
        with st.spinner("Extracting text…"):
            text, meta = extract_text_from_pdf(uploaded_pdf)

        if not text.strip():
            st.error("No extractable text found. If this is a scanned PDF, try OCR (e.g., Tesseract) before summarization.")
            return

        char_count = len(text)
        word_count = len(text.split())
        st.metric("Original length", f"{word_count:,} words ({char_count:,} chars)")
        if meta.get("pages") is not None:
            st.caption(f"Pages: {meta['pages']}")

        with st.spinner("Summarizing…"):
            if mode.startswith("Extractive"):
                summary = sumy_lexrank_summary(text, language=language, max_sentences=max_sents)
                if not summary:
                    summary = naive_extractive_summary(text, max_sentences=max_sents)
                partials = []
            else:
                summary, partials = abstractive_summarize(text, model_name=model_name, max_len=max_len, min_len=min_len)

        st.subheader("Summary")
        st.write(summary)

        st.download_button(
            "Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain",
        )

        if mode.startswith("Abstractive") and partials:
            with st.expander("Chunked partial summaries (pre-merge)"):
                for i, p in enumerate(partials, 1):
                    st.markdown(f"Chunk {i}")
                    st.write(p)

if page == "🛠️ Code Review":
    render_code_review()
else:
    render_pdf_summarizer()
