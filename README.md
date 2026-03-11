# 🌾 AgroVision AI — Agriculture Image Analyzer

A **premium Streamlit application** that uses **Hugging Face AI models** to analyze plant and crop images, detect diseases, and provide comprehensive theoretical treatment solutions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **4 AI Models** | Plant disease classifier, crop identifier, ViT model, fruit/veg detector |
| 🦠 **38+ Disease Classes** | Tomato, Potato, Corn, Apple diseases and more |
| 💊 **Treatment Plans** | Chemical, organic, and biological treatment options |
| 🛡️ **Prevention Strategies** | Step-by-step crop protection strategies |
| 📊 **Confidence Charts** | Interactive Plotly bar charts and severity gauge |
| 📸 **Camera & Upload** | Upload images or take photos directly |
| 📥 **Report Download** | Download full diagnosis report as Markdown |
| 🌿 **Crop Care Tips** | General agricultural best practices |

---

## 🚀 Quick Start

### 1. Clone / Download

```bash
cd d:\Agro
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Hugging Face API Token

1. Sign up at [huggingface.co](https://huggingface.co)
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Create a **Read** token (free)
4. Copy the token

### 5. Configure Token

**Option A** — Enter it in the sidebar when the app launches.

**Option B** — Add it to `.env`:

```
HUGGINGFACE_API_TOKEN=hf_your_actual_token_here
```

### 6. Run the App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🤖 AI Models Used

| Model | HuggingFace ID | Best For |
|---|---|---|
| 🦠 Plant Disease Classifier | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` | Leaf disease detection |
| 🌿 Crop / Plant Identifier | `microsoft/resnet-50` | General plant ID |
| 🔬 General Agriculture Vision | `google/vit-base-patch16-224` | Scene understanding |
| 🍎 Fruit & Vegetable Detector | `jazzmacedo/fruits-and-vegetables-detector` | Produce detection |

---

## 🌱 Supported Crops & Diseases

- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot
- **Potato**: Early Blight, Late Blight
- **Corn**: Common Rust, Northern Leaf Blight
- **Apple**: Apple Scab, Black Rot
- **General**: Broad agricultural scene analysis

---

## 📁 Project Structure

```
d:\Agro\
├── app.py              # Main Streamlit application
├── config.py           # Models, disease knowledge base, settings
├── hf_utils.py         # Hugging Face API client
├── knowledge_base.py   # Disease lookup and label parsing
├── style.css           # Premium dark-theme CSS
├── requirements.txt    # Python dependencies
├── .env                # API token (do not commit)
├── .env.example        # Token template
└── README.md           # This file
```

---

## 📸 Tips for Best Results

- Upload **clear, well-lit** close-up photos of affected plant parts
- Focus on **leaves, stems, or fruit** showing disease symptoms
- Ensure the image is **not blurry or overexposed**
- Use the **Plant Disease Classifier** model for the most accurate leaf disease detection

---

## ⚠️ Disclaimer

AI predictions are for **educational and guidance purposes only**. Always consult a **certified agronomist or plant pathologist** before making critical agricultural decisions.

---

## 📄 License

MIT License — Free to use and modify.
