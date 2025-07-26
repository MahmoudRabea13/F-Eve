import gradio as gr
from functions import load_models, extract_and_preprocess_faces, predict_age, predict_gender, match_faces
from PIL import Image
import tempfile
import math

# Load models
AGE_MODEL_PATH = "vgg_age.h5"
GENDER_MODEL_PATH = "squeeze_gender.h5"
age_model, gender_model, face_model = load_models(AGE_MODEL_PATH, GENDER_MODEL_PATH)

def analyze_faces(img1, img2, criteria_thresh, cos_thresh, eucl_thresh, angle_thresh):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp1, tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp2:
        img1.save(tmp1.name)
        img2.save(tmp2.name)

        face1, gray1 = extract_and_preprocess_faces(tmp1.name, face_model)
        face2, gray2 = extract_and_preprocess_faces(tmp2.name, face_model)

        age1 = predict_age(gray1, age_model)
        age2 = predict_age(gray2, age_model)
        gender1, _ = predict_gender(gray1, gender_model)
        gender2, _ = predict_gender(gray2, gender_model)

        match_result = match_faces(face1, face2, criteria_thresh, cos_thresh, eucl_thresh, angle_thresh)
        emoji1 = "👨" if gender1 == "Male" else "👩"
        emoji2 = "👨" if gender2 == "Male" else "👩"
        summary = f"""
        <div style='text-align:center; font-weight:bold; color:#0A4D68; font-size:16px;'>
         Face 1 → Age: {math.ceil(age1)}, Gender: {gender1} {emoji1} <br>
         Face 2 → Age: {math.ceil(age2)}, Gender: {gender2} {emoji2} <br><br>
        🧠 <u>Matching</u><br>
        • Cosine Similarity: {match_result['cos_sim']:.4f} (>{cos_thresh})<br>
        • Euclidean Distance: {match_result['eucl_dist']:.4f} (<{eucl_thresh})<br>
        • Angle: {match_result['angle']:.2f}° (<{angle_thresh})<br><br>
        ✅ Final: {'<span style="color:green;">MATCH</span>' if match_result['is_match'] else '<span style="color:red;">MISMATCH</span>'}
        ({match_result['votes']}/3 passed)
        </div>
        """
        return summary

# Gradio UI
theme = gr.themes.Soft(primary_hue="blue")

with gr.Blocks(theme=theme, title="F-Eve") as demo:
    gr.Markdown("## 🧬 Facial Evolution Verification Engine (F-Eve)")
    #gr.Markdown("`Upload two facial images to analyze age, gender, and identity match.`")

    with gr.Row():
        img1_input = gr.Image(label="Face 1", type="pil", height=240, width=240)
        img2_input = gr.Image(label="Face 2", type="pil", height=240, width=240)

    with gr.Row():
        # Left column - compact accordion settings
        with gr.Column(scale=1, min_width=260):
            with gr.Accordion("⚙️ Matching Settings", open=True):
                criteria_thresh = gr.Slider(1, 3, value=1, step=1, label="Min Matching Criteria (out of 3)", container=True)
                cos_thresh = gr.Slider(0.0, 1.0, value=0.4, step=0.01, label="Cosine Similarity Threshold", container=True)
                eucl_thresh = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Euclidean Distance Threshold", container=True)
                angle_thresh = gr.Slider(0, 90, value=50, step=1, label="Angular Distance Threshold (degrees)", container=True)

        # Right column - centered button + results
        with gr.Column(scale=1, min_width=260, elem_id="result-column"):
            analyze_button = gr.Button("🔍 Analyze", elem_id="analyze-btn")
            output_html = gr.HTML(label="", elem_id="results-box")

    analyze_button.click(
        analyze_faces,
        inputs=[img1_input, img2_input, criteria_thresh, cos_thresh, eucl_thresh, angle_thresh],
        outputs=output_html
    )

    # Inline CSS styling
    demo.stylesheets.append("""
        /* Custom styles for the Gradio app */
        body {
            background: linear-gradient(to right, #dceefb, #f0f4f8);
            font-family: 'Segoe UI', sans-serif;
        }

        #analyze-btn {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 60%;
            padding: 12px;
            font-weight: bold;
            font-size: 16px;
            background: linear-gradient(to right, #0077b6, #00b4d8);
            color: white;
            border: none;
            border-radius: 10px;
            transition: background 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        #analyze-btn:hover {
            background: linear-gradient(to right, #00b4d8, #0077b6);
            transform: scale(1.03);
        }

        #results-box {
            text-align: center;
            margin-top: 15px;
            background: rgba(255, 255, 255, 0.7);
            padding: 15px;
            border-radius: 15px;
            font-size: 16px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease-in-out;
        }

        .gr-accordion {
            max-height: 320px;
            overflow-y: auto;
            border-radius: 12px;
            background: rgba(255,255,255,0.8);
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            animation: fadeIn 0.8s ease-in-out;
        }

        #result-column {
            display: flex;
            flex-direction: column;
            justify-content: start;
            align-items: center;
        }

        .gr-image {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border-radius: 15px;
            animation: fadeIn 0.8s ease-in-out;
        }

        h1, h2, h3 {
            text-align: center !important;
            color: #0A4D68;
            animation: fadeInDown 1s ease-in-out;
            font-weight: 700;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    """)
demo.launch()
