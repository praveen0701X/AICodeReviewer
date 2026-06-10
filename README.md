# AICodeReviewer

An AI-powered code review tool built with Python and Streamlit.
Paste your code and get instant feedback on bugs, style issues,
and improvements — powered by Google Gemini.

## Features
- Supports Python, JavaScript, C/C++, and more
- Detects bugs, logic errors, and bad practices
- Suggests cleaner, more efficient rewrites
- Simple web UI — no account or setup beyond install

## Getting started

**Prerequisites:** Python 3.8+

```bash
git clone https://github.com/praveen0701X/AICodeReviewer
cd AICodeReviewer
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Usage
1. Paste any code snippet into the text area
2. Click **Review**
3. Read the structured feedback — issues, suggestions, and an
   improved version of your code

## How it works
The app sends your code to the Google Gemini API with a
carefully crafted system prompt that instructs it to behave
like a senior code reviewer. The response is parsed and
displayed in a clean Streamlit interface.

## Tech stack
| Layer | Tool |
|-------|------|
| UI | Streamlit |
| AI engine | Google Gemini API |
| Language | Python 3 |

## License
MIT © praveen0701X
