import gradio as gr
import requests
import re
import concurrent.futures
from urllib.parse import urlparse, urlunparse
import string
import unicodedata
import math

# Languages
languages = [
    "Latin",
    "English",
]


# Helper functions
def extract_chunks(text):
    parts = re.split(r"(\n+)", text)
    sentences, breaks = [], []
    for part in parts:
        if part.strip() == "":
            breaks.append(part)
        else:
            sentences.append(part.strip())
    return [s for s in sentences if s], breaks


def rmSpacesBeforePunct(text):
    noats = re.sub(r'@@( |$)', '', text) # rm @@
    nospacebeforepunct = re.sub(r" +([:,;!?.’“])", r'\1', noats) # rm spaces before punctuation
    nospacebeforeparens = re.sub(r'([({\[’]) +', r'\1', nospacebeforepunct)  # rm multiple spaces after opening parens/brackets
    return re.sub(r' +([)}\]])', r'\1', nospacebeforeparens) # rm multiple spaces before opening parens/brackets

def rebuild_text(translated_sentences, breaks):
    result = []
    s_idx = b_idx = 0
    while s_idx < len(translated_sentences) or b_idx < len(breaks):
        if s_idx < len(translated_sentences):
            result.append(translated_sentences[s_idx])
            s_idx += 1
        if b_idx < len(breaks):
            result.append(breaks[b_idx])
            b_idx += 1
    return rmSpacesBeforePunct("".join(result))


def get_parent_url(api_url):
    parsed = urlparse(api_url)
    path_parts = parsed.path.rstrip('/').split('/')  # Split path and remove last segment
    parent_path = '/'.join(path_parts[:-1]) + '/'
    return urlunparse((parsed.scheme, parsed.netloc, parent_path, '', '', ''))


def get_models(api_url):
    base_url = get_parent_url(api_url)
    try:
        response = requests.get(f"{base_url}/models")
        data = response.json()
        models = [m.get("id", str(m)) for m in data.get("models", [])]
        return models if models else ["No models found"]
    except Exception as e:
        return [f"Error fetching models: {e}"]


def update_models(api_url):
    models = get_models(api_url)
    return {"choices": models, "value": models[0]}


def to_ascii(text):  # ASCIIfy (convert Æ,æ,Œ,œ,á,é,í,ó,ú, etc. to ASCII):
    extras = { 'Æ': 'Ae', 'æ': 'ae', 'Œ': 'Oe', 'œ': 'oe',
               'Ǣ': 'Ae', 'ǣ': 'ae',
               'Ǽ': 'Ae', 'ǽ': 'ae', 'ǽ': 'ae' }
    text = ''.join(extras.get(c, c) for c in text)
    normalized = unicodedata.normalize('NFKD', text)
    return normalized.encode('ascii', 'ignore').decode('ascii')

def translate_text(api_url, model_id, source_lang, target_lang, source_text):
    source_text = to_ascii(source_text)
    # this is due to how my pre-tokenized 🇻🇦 source text was formatted:
    source_text = re.sub(f'([{re.escape(string.punctuation)}])', r' \1 ', source_text)   # add space before and after punctuation
    source_text = re.sub(r'\s+', ' ', source_text).strip()   # normalize whitespace

    paragraphs, breaks = extract_chunks(source_text)
    translated_sentences = []

    payloads = []
    for para in paragraphs:
        payloads.append(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": para}],
            }
        )

    def send_request(payload):
        try:
            response = requests.post(api_url, json=payload)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(paragraphs)) as executor:
        futures = [executor.submit(send_request, p) for p in payloads]
        results = [f.result() for f in futures]

    for data in results:
        if "choices" in data:
            translated_sentences.append(data["choices"][0]["message"]["content"])
        else:
            predictions = data.get("predictions", [])
            scores = data.get("scores", [])
            if predictions and predictions[0]:
                translated_sentences.append(predictions[0][0])
                score = scores[0][0] if scores else 0
            else:
                translated_sentences.append("Error: No translation returned by the server.")

    return rebuild_text(translated_sentences, breaks), score, math.exp(score)*100


custom_css = """
/* General layout */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f6fa;
}

/* Inputs and textboxes */
input, select, textarea {
    border-radius: 8px;
    border: 1px solid #ccc;
    padding: 8px;
    font-size: 14px;
    transition: all 0.2s;
}

input:focus, select:focus, textarea:focus {
    border-color: #4285F4;
    box-shadow: 0 0 5px rgba(66,133,244,0.5);
    outline: none;
}

/* Boxes */
.gr-box {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
}

/* Buttons */
button {
    width: 220px;
    background-color: #4285F4;
    color: white;
    font-size: 16px;
    border: none;
    border-radius: 6px;
    padding: 10px;
    cursor: pointer;
    margin-top: 10px;
    transition: all 0.2s;
}
button:hover {
    background-color: #357AE8;
    transform: translateY(-2px);
}

/* Scrollable settings column */
#settings-col {
    overflow-y: auto;
    max-height: 600px;
    padding: 10px;
}
"""
# Gradio Interface
with gr.Blocks(title='Eole Latin 🇻🇦 → Engish 🇬🇧 Translator') as iface:
    gr.Markdown('<h1 style="text-align: center; font-family: Arial;"><a href="https://eole-nlp.github.io/eole" target="_blank">Eole</a> Latin 🇻🇦 → English 🇬🇧 Translator</h1>')
    gr.Markdown('<a href="https://huggingface.co/Geremia23/AquinasLatinEnglishModel" target="_blank">AquinasLatinEnglish model</a> trained on the <a href="https://huggingface.co/datasets/Geremia23/AquinasLatinEnglish" target="_blank">AquinasLatinEnglish parallel corpus</a> using <a href="https://isidore.co/forum/index.php/topic,377.msg1327.html#msg1327" target="_blank">Transformers and Byte-Pair Encoding (BPE)</a>.')

    with gr.Row(equal_height=True):
        # Left Column: Source language + text
        with gr.Column(scale=4):
            source_lang = gr.Dropdown(languages[0:1], label="Source Language", value="Latin", interactive=False)
            source_text = gr.Textbox(placeholder="Enter text here…", lines=15, label="Source Text (1024 tokens max)", autofocus=True)

        # Right Column: Target language + translated text
        with gr.Column(scale=4):
            target_lang = gr.Dropdown(languages[1:], label="Target Language", value="English", interactive=False)
            translated_text = gr.Textbox(
                placeholder="Translation will appear here…", lines=15, label="Translated Text", interactive=False
            )

        # Settings Column (scrollable)
        with gr.Column(scale=2, elem_id="settings-col") as settings_col:
            api_url = gr.Dropdown(
                label="API URL",
                choices=["https://isidore.co/laen/infer"],
                value="https://isidore.co/laen/infer",
                interactive=False,
            )
            model_id = gr.Dropdown(choices=get_models(api_url.value), label="Model", value='aquinas-latin-english', interactive=False)
            score = gr.Textbox(label="Prediction Score: ln(prob. of best prediction)", value="", interactive=False)
            prob_of_best = gr.Textbox(label="Prob. of Best Prediction (%)", value="", interactive=False)

    # Update models when API URL changes
    api_url.change(update_models, inputs=[api_url], outputs=[model_id])

    # Translate button
    translate_button = gr.Button("Translate", elem_id="button-container")
    translate_button.click(
        translate_text,
        inputs=[api_url, model_id, source_lang, target_lang, source_text],
        outputs=[translated_text, score, prob_of_best],
    )

iface.launch(css=custom_css)
