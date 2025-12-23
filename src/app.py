import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime
import sys
import os

# Ensure we can import from src and the cloned repo
sys.path.append(os.path.abspath("ClinicalReport"))
from text2bioc import text2document
from ssplit import NegBioSSplitter

# Import local modules
from src.preprocessing import get_labels, clean, get_mention_keywords, CATEGORIES
from src.model import get_model_answer, load_bert_model

# Initialize model on startup
print("Initializing models...")
load_bert_model()

# --- Example Data ---
EXAMPLE_REPORTS = {
    "Example 1: Multiple Pathologies": """
1. mild pulmonary edema, and cardiomegaly. trace pleural fluid effusions.
2. low lung volumes with minimal basilar atelectasis.
3. no new focal consolidation.
4. interval placement of defibrillation pads.""",

    "Example 2: Negative Findings": """
1. unremarkable cardiomediastinal silhouette
2. diffuse reticular pattern, which can be seen with an atypical infection or chronic fibrotic change. no focal consolidation.
3. no pleural effusion or pneumothorax
4. mild degenerative changes in the lumbar spine and old right rib fractures.""",
}

EXAMPLE_QA_PAIRS = {
    "Clinical Case 1": {
        "passage": """Abnormal echocardiogram findings and followup. Shortness of breath, congestive heart failure, and valvular insufficiency. The patient complains of shortness of breath, which is worsening. The patient underwent an echocardiogram, which shows severe mitral regurgitation and also large pleural effusion. The patient is an 86-year-old female admitted for evaluation of abdominal pain and bloody stools.""",
        "question": "How old is the patient?"
    },
    "Clinical Case 2": {
        "passage": """The word pharmacy is derived from its root word pharma which was a term used since the 15th–17th centuries. However, the original Greek roots from pharmakos imply sorcery or even poison.""",
        "question": "What word is the word pharmacy taken from?"
    }
}

# --- Helper Functions for UI ---

def load_disease_example(example_name):
    return EXAMPLE_REPORTS.get(example_name, "")

def load_qa_example(example_name):
    if example_name in EXAMPLE_QA_PAIRS:
        return EXAMPLE_QA_PAIRS[example_name]["passage"], EXAMPLE_QA_PAIRS[example_name]["question"]
    return "", ""

def process_disease_extraction(report_text, use_cleanup, example_selector):
    if not report_text or report_text.strip() == "":
        return "Please enter a clinical report.", None, None, None, None

    try:
        start_time = time.time()
        
        # Split sentences
        splitter = NegBioSSplitter()
        document = text2document("temp", report_text)
        document = splitter.split_doc(document)
        sentences_array = [s.text for s in document.passages[0].sentences]

        if use_cleanup:
            sentences_array = [clean(s) for s in sentences_array]

        # Get Predictions
        predictions = get_labels(sentences_array)
        processing_time = time.time() - start_time

        # Format Results for Table
        results_data = []
        for category, is_present in predictions.items():
            status = "✓ Present" if is_present else "✗ Absent"
            color = "🔴" if is_present else "🟢"
            results_data.append([category, color + " " + status])

        # Create Plot
        disease_names = list(predictions.keys())
        disease_values = [1 if predictions[d] else 0 for d in disease_names]

        fig = go.Figure(data=[
            go.Bar(
                x=disease_values,
                y=disease_names,
                orientation='h',
                marker=dict(
                    color=['#e74c3c' if v == 1 else '#27ae60' for v in disease_values],
                    line=dict(color='#2c3e50', width=1.5)
                ),
                text=[f"{'Present' if v == 1 else 'Absent'}" for v in disease_values],
                textposition='auto',
            )
        ])
        fig.update_layout(title="Disease Detection Results", height=500)

        # Highlight Text Logic
        highlighted_text = report_text
        for category, is_present in predictions.items():
            if is_present:
                keywords = get_mention_keywords(category)
                for keyword in keywords:
                    import re
                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                    highlighted_text = pattern.sub(
                        lambda m: f'<mark style="background-color: #ffeb3b; padding: 2px 4px;">{m.group()}</mark>',
                        highlighted_text
                    )
        
        highlighted_html = f'<div style="background: white; padding: 20px;">{highlighted_text}</div>'
        
        # Metrics HTML
        metrics_html = f"<div>Processing Time: {processing_time:.3f} seconds</div>"

        return results_data, fig, metrics_html, highlighted_html, None

    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None

def process_question_answering(passage, question, example_selector):
    if not passage or not question:
        return "Please enter both passage and question.", None, None

    try:
        start_time = time.time()
        answer = get_model_answer(question, passage)
        processing_time = time.time() - start_time

        answer_html = f"""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 25px; border-radius: 12px; color: white;">
            <h3>Answer:</h3>
            <div style="background: rgba(255,255,255,0.9); color: black; padding: 10px; border-radius: 5px;">{answer}</div>
            <small>Time: {processing_time:.3f}s</small>
        </div>
        """
        return answer_html, "", "" # Returning empty strings for optional outputs
    except Exception as e:
        return f"Error: {str(e)}", None, None

def process_batch_reports(file, use_cleanup):
    if file is None:
        return "Please upload a CSV file.", None, None
    try:
        batch_df = pd.read_csv(file.name)
        if 'Report Impression' not in batch_df.columns:
            return "CSV must contain 'Report Impression' column.", None, None
            
        splitter = NegBioSSplitter()
        results_list = []
        
        for idx, row in batch_df.iterrows():
            report = row['Report Impression']
            document = text2document(str(idx), report)
            document = splitter.split_doc(document)
            sentences = [s.text for s in document.passages[0].sentences]
            
            if use_cleanup:
                sentences = [clean(s) for s in sentences]
                
            predictions = get_labels(sentences)
            row_res = {'Report_ID': idx}
            row_res.update(predictions)
            results_list.append(row_res)
            
        results_df = pd.DataFrame(results_list)
        output_path = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_path, index=False)
        
        return "Processing Complete!", None, output_path
    except Exception as e:
        return f"Error: {str(e)}", None, None

# --- Gradio UI Layout ---

custom_css = """
#header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; color: white; text-align: center; }
"""

with gr.Blocks(css=custom_css, title="Clinical Report Analyzer", theme=gr.themes.Soft()) as demo:
    gr.HTML('<div id="header"><h1>Clinical Report Analyzer</h1></div>')

    with gr.Tabs():
        # Tab 1: Disease Extraction
        with gr.Tab("🔍 Disease Label Extraction"):
            with gr.Row():
                with gr.Column():
                    ex_drop = gr.Dropdown(list(EXAMPLE_REPORTS.keys()), label="Load Example")
                    d_input = gr.Textbox(label="Report", lines=10)
                    d_clean = gr.Checkbox(label="Cleanup", value=True)
                    d_btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    d_table = gr.Dataframe(headers=["Disease", "Status"])
                    d_plot = gr.Plot()
                    d_metrics = gr.HTML()
                    d_high = gr.HTML()

            ex_drop.change(load_disease_example, inputs=[ex_drop], outputs=[d_input])
            d_btn.click(process_disease_extraction, inputs=[d_input, d_clean, ex_drop], outputs=[d_table, d_plot, d_metrics, d_high, ex_drop])

        # Tab 2: QA
        with gr.Tab("💬 Medical Q&A"):
            with gr.Row():
                with gr.Column():
                    q_drop = gr.Dropdown(list(EXAMPLE_QA_PAIRS.keys()), label="Load Case")
                    q_pass = gr.Textbox(label="Passage", lines=8)
                    q_ques = gr.Textbox(label="Question")
                    q_btn = gr.Button("Ask", variant="primary")
                with gr.Column():
                    q_ans = gr.HTML()
            
            q_drop.change(load_qa_example, inputs=[q_drop], outputs=[q_pass, q_ques])
            q_btn.click(process_question_answering, inputs=[q_pass, q_ques, q_drop], outputs=[q_ans, q_ans, q_ans])

        # Tab 3: Batch
        with gr.Tab("📊 Batch Processing"):
            b_file = gr.File(label="Upload CSV")
            b_clean = gr.Checkbox(label="Cleanup", value=True)
            b_btn = gr.Button("Process")
            b_out = gr.HTML()
            b_down = gr.File()
            
            b_btn.click(process_batch_reports, inputs=[b_file, b_clean], outputs=[b_out, b_out, b_down])

if __name__ == "__main__":
    demo.launch()
