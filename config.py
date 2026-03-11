"""
Configuration file for Agriculture Image Analyzer
Contains model endpoints, disease knowledge base, and app settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Hugging Face API Settings
# ──────────────────────────────────────────────
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
HF_API_BASE  = "https://api-inference.huggingface.co/models"

# ──────────────────────────────────────────────
# Model Endpoints
# ──────────────────────────────────────────────
MODELS = {
    "Plant Disease Classifier": {
        "id":          "ozair23/mobilenet_v2_1.0_224-finetuned-plantdisease",
        "task":        "image-classification",
        "description": "MobileNetV2 fine-tuned on PlantVillage — 38 disease classes (tomato, potato, corn, apple, grape and more).",
        "icon":        "🦠",
        "labels":      "plant_disease",
    },
    "Plant Disease ViT": {
        "id":          "Diginsa/Plant-Disease-Detection-Project",
        "task":        "image-classification",
        "description": "Vision Transformer specifically trained to detect plant diseases from leaf photos.",
        "icon":        "🔬",
        "labels":      "plant_disease",
    },
    "Crop Health Inspector": {
        "id":          "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
        "task":        "image-classification",
        "description": "MobileNetV2 for identifying healthy vs diseased crops across 38 categories.",
        "icon":        "🌿",
        "labels":      "plant_disease",
    },
    "Fruit & Vegetable Detector": {
        "id":          "jazzmacedo/fruits-and-vegetables-detector",
        "task":        "image-classification",
        "description": "Classifies 36 types of fruits and vegetables — best for produce identification.",
        "icon":        "🍎",
        "labels":      "produce",
    },
    "General Vision (Validation)": {
        "id":          "microsoft/resnet-50",
        "task":        "image-classification",
        "description": "General-purpose model used for image validation and scene consistency.",
        "icon":        "🔍",
        "labels":      "general",
    },
}

# ──────────────────────────────────────────────
# Comprehensive Plant Disease Knowledge Base
# ──────────────────────────────────────────────
DISEASE_KNOWLEDGE = {
    # ── Tomato ──────────────────────────────────
    "tomato": {
        "Tomato___Bacterial_spot": {
            "display":     "Tomato Bacterial Spot",
            "severity":    "High",
            "pathogen":    "Xanthomonas campestris pv. vesicatoria",
            "symptoms":    "Small, dark water-soaked spots on leaves, stems, and fruit. Spots enlarge and turn brown with yellow halos.",
            "causes":      "Warm, wet, humid conditions; infected seed; overhead irrigation spreads bacteria.",
            "treatment":   [
                "Apply copper-based bactericides (copper hydroxide or copper sulfate) at first sign.",
                "Use fixed copper + mancozeb sprays every 7–10 days.",
                "Remove and destroy infected plant debris.",
                "Avoid overhead irrigation; use drip irrigation instead.",
            ],
            "prevention":  [
                "Use certified disease-free or resistant seed varieties.",
                "Practice 2–3 year crop rotation.",
                "Sanitize tools and equipment regularly.",
                "Maintain proper plant spacing for air circulation.",
            ],
            "organic":     ["Neem oil spray", "Copper sulfate (Bordeaux mixture)", "Bacillus subtilis biofungicide"],
            "fertilizer":  "Balanced NPK with micronutrients; avoid excess nitrogen.",
            "recovery":    "3–4 weeks with proper treatment; remove severely infected plants.",
        },
        "Tomato___Early_blight": {
            "display":     "Tomato Early Blight",
            "severity":    "Medium",
            "pathogen":    "Alternaria solani",
            "symptoms":    "Dark brown concentric ring spots (target-board pattern) on older leaves first. Yellowing around lesions.",
            "causes":      "Warm (24–29 °C), humid conditions; infected soil or debris; poor nutrition.",
            "treatment":   [
                "Apply fungicides: chlorothalonil, mancozeb, or azoxystrobin.",
                "Remove infected lower leaves immediately.",
                "Improve air circulation by pruning excess foliage.",
                "Apply foliar calcium spray to strengthen cell walls.",
            ],
            "prevention":  [
                "Rotate crops (avoid Solanaceae for 3 years).",
                "Use resistant tomato varieties.",
                "Mulch around plants to prevent soil splash.",
                "Water at the base of plants in the morning.",
            ],
            "organic":     ["Copper-based fungicides", "Neem oil", "Bacillus subtilis (Serenade)"],
            "fertilizer":  "Adequate phosphorus and potassium; calcium supplementation.",
            "recovery":    "2–3 weeks with fungicide + leaf removal.",
        },
        "Tomato___Late_blight": {
            "display":     "Tomato Late Blight",
            "severity":    "Critical",
            "pathogen":    "Phytophthora infestans",
            "symptoms":    "Large, irregular grayish-green water-soaked spots on leaves. White fuzzy mold on undersides. Dark lesions on stems and fruit.",
            "causes":      "Cool (10–25 °C) and wet weather; high humidity (>90%); airborne spores.",
            "treatment":   [
                "Apply systemic fungicides: metalaxyl, cymoxanil, dimethomorph.",
                "Use contact fungicides: mancozeb, chlorothalonil as protectants.",
                "Remove and destroy ALL infected plant material (do not compost).",
                "In severe cases, destroy the entire crop to prevent spread.",
            ],
            "prevention":  [
                "Plant resistant varieties (e.g., Mountain Magic, Defiant).",
                "Avoid planting in low-lying, poorly drained areas.",
                "Monitor weather forecasts; apply preventive fungicides before rainy periods.",
                "Destroy all plant debris at season end.",
            ],
            "organic":     ["Copper hydroxide", "Bordeaux mixture", "Phosphorous acid"],
            "fertilizer":  "Balanced fertilization; avoid over-irrigation.",
            "recovery":    "Difficult — act within 24–48 hours of detection.",
        },
        "Tomato___Leaf_Mold": {
            "display":     "Tomato Leaf Mold",
            "severity":    "Medium",
            "pathogen":    "Passalora fulva (Fulvia fulva)",
            "symptoms":    "Yellow patches on upper leaf surface; olive-green to grayish velvety mold on undersides.",
            "causes":      "High humidity (>85%); poor ventilation; temperatures 22–25 °C.",
            "treatment":   [
                "Apply fungicides: chlorothalonil, mancozeb, myclobutanil.",
                "Reduce humidity by improving greenhouse ventilation.",
                "Avoid overhead watering.",
                "Remove heavily infected leaves.",
            ],
            "prevention":  [
                "Use resistant varieties (e.g., Jasper, Betty).",
                "Ensure adequate plant spacing.",
                "Maintain humidity below 85% in greenhouses.",
            ],
            "organic":     ["Potassium bicarbonate", "Neem oil", "Copper fungicide"],
            "fertilizer":  "Balanced NPK; avoid excess nitrogen.",
            "recovery":    "2–3 weeks with environment control and fungicide.",
        },
        "Tomato___Septoria_leaf_spot": {
            "display":     "Tomato Septoria Leaf Spot",
            "severity":    "Medium",
            "pathogen":    "Septoria lycopersici",
            "symptoms":    "Many small, circular spots with dark borders and tan/gray centers. Tiny black specks (pycnidia) in spot centers.",
            "causes":      "Warm wet weather; infected debris in soil; spreads by rain splash.",
            "treatment":   [
                "Apply fungicides: chlorothalonil, mancozeb, copper-based.",
                "Remove infected lower leaves promptly.",
                "Avoid working with wet plants.",
            ],
            "prevention":  [
                "3-year crop rotation.",
                "Mulch to reduce soil splash.",
                "Use drip irrigation.",
                "Destroy infected crop residues.",
            ],
            "organic":     ["Copper hydroxide", "Bacillus subtilis", "Neem oil"],
            "fertilizer":  "Maintain soil fertility; potassium strengthens resistance.",
            "recovery":    "3–4 weeks with consistent fungicide program.",
        },
        "Tomato___healthy": {
            "display":     "Healthy Tomato Plant",
            "severity":    "None",
            "pathogen":    "N/A",
            "symptoms":    "No visible disease symptoms. Leaves are vibrant green, firm, and well-formed.",
            "causes":      "N/A — plant is healthy.",
            "treatment":   ["No treatment needed. Maintain current care routine."],
            "prevention":  [
                "Continue regular monitoring every 3–5 days.",
                "Maintain consistent watering and fertilization schedule.",
                "Practice integrated pest management (IPM).",
            ],
            "organic":     ["Compost tea as foliar feed", "Neem oil as preventive spray"],
            "fertilizer":  "Balanced NPK (10-10-10) + calcium and magnesium.",
            "recovery":    "N/A — plant is healthy!",
        },
    },

    # ── Potato ──────────────────────────────────
    "potato": {
        "Potato___Early_blight": {
            "display":     "Potato Early Blight",
            "severity":    "Medium",
            "pathogen":    "Alternaria solani",
            "symptoms":    "Dark brown circular lesions with concentric rings on lower older leaves. Yellowing of surrounding tissue.",
            "causes":      "Warm temperatures (24–29 °C); humid conditions; nutrient-deficient plants.",
            "treatment":   [
                "Apply fungicides: chlorothalonil, mancozeb, azoxystrobin.",
                "Remove infected leaves and destroy them.",
                "Ensure adequate plant nutrition (especially nitrogen and potassium).",
            ],
            "prevention":  [
                "Plant certified disease-free seed potatoes.",
                "Rotate crops with non-solanaceous plants.",
                "Maintain optimal fertility to reduce plant stress.",
            ],
            "organic":     ["Copper fungicides", "Neem oil", "Bacillus subtilis"],
            "fertilizer":  "Adequate nitrogen and potassium; balanced micronutrients.",
            "recovery":    "2–4 weeks with prompt treatment.",
        },
        "Potato___Late_blight": {
            "display":     "Potato Late Blight",
            "severity":    "Critical",
            "pathogen":    "Phytophthora infestans",
            "symptoms":    "Water-soaked pale green/dark brown lesions on leaves and stems. White fluffy sporulation on undersides in humid conditions. Tuber rot.",
            "causes":      "Cool (10–20 °C), wet weather; high humidity; infected seed potatoes.",
            "treatment":   [
                "Apply systemic fungicides: metalaxyl-M + mancozeb (Ridomil Gold MZ).",
                "Apply contact fungicides: chlorothalonil, copper hydroxide.",
                "Destroy infected plant material — do not compost.",
                "Hill soil over tubers to prevent infection.",
            ],
            "prevention":  [
                "Use certified disease-free seed potatoes.",
                "Plant resistant varieties (e.g., Sarpo Mira, Defender).",
                "Apply preventive fungicides every 7 days in high-risk weather.",
                "Harvest promptly before wet season.",
            ],
            "organic":     ["Bordeaux mixture", "Copper hydroxide", "Phosphorous acid"],
            "fertilizer":  "Balanced K and P; avoid excess nitrogen.",
            "recovery":    "Immediate action required — can destroy entire crop in days.",
        },
        "Potato___healthy": {
            "display":     "Healthy Potato Plant",
            "severity":    "None",
            "pathogen":    "N/A",
            "symptoms":    "Vigorous green foliage, no lesions or discoloration.",
            "causes":      "N/A",
            "treatment":   ["No treatment necessary. Maintain regular monitoring."],
            "prevention":  [
                "Monitor regularly for early disease signs.",
                "Maintain optimal soil moisture and fertility.",
                "Implement IPM strategies proactively.",
            ],
            "organic":     ["Compost application", "Preventive neem oil spray"],
            "fertilizer":  "Balanced NPK with emphasis on potassium for tuber quality.",
            "recovery":    "N/A — plant is thriving!",
        },
    },

    # ── Corn / Maize ────────────────────────────
    "corn": {
        "Corn___Common_rust": {
            "display":     "Corn Common Rust",
            "severity":    "Medium",
            "pathogen":    "Puccinia sorghi",
            "symptoms":    "Brick-red to brown oval pustules (uredia) scattered on both leaf surfaces. Pustules release powdery reddish spores.",
            "causes":      "Cool (16–23 °C), humid conditions; airborne urediniospores from alternate hosts.",
            "treatment":   [
                "Apply fungicides: propiconazole, azoxystrobin, trifloxystrobin.",
                "Time applications at first sign of disease before tasseling.",
                "Scout fields regularly during cool, wet periods.",
            ],
            "prevention":  [
                "Plant rust-resistant hybrid varieties.",
                "Avoid planting in areas with history of rust outbreaks.",
                "Monitor fields regularly from silking onwards.",
            ],
            "organic":     ["Sulfur-based fungicides", "Neem oil", "Potassium bicarbonate"],
            "fertilizer":  "Adequate potassium and phosphorus improves plant resistance.",
            "recovery":    "3–4 weeks with early fungicide application.",
        },
        "Corn___Northern_Leaf_Blight": {
            "display":     "Corn Northern Leaf Blight",
            "severity":    "High",
            "pathogen":    "Exserohilum turcicum",
            "symptoms":    "Large (2.5–15 cm), long, grayish-green to tan cigar-shaped lesions on leaves. Lesions have wavy margins.",
            "causes":      "Moderate temperatures (18–27 °C); high humidity and long periods of leaf wetness.",
            "treatment":   [
                "Apply fungicides: azoxystrobin, propiconazole, tebuconazole.",
                "Apply at VT (tasseling) growth stage for best results.",
                "Remove heavily infected leaves when possible.",
            ],
            "prevention":  [
                "Plant resistant hybrids (Ht gene resistance).",
                "Rotate with non-host crops (soybean, wheat).",
                "Manage crop residue by deep tillage.",
            ],
            "organic":     ["Copper-based fungicides", "Bacillus amyloliquefaciens"],
            "fertilizer":  "Balanced nitrogen; high potassium improves resistance.",
            "recovery":    "3–5 weeks; yield impact depends on timing of infection.",
        },
        "Corn___healthy": {
            "display":     "Healthy Corn Plant",
            "severity":    "None",
            "pathogen":    "N/A",
            "symptoms":    "Deep green, uniform leaves; no lesions, pustules, or discoloration.",
            "causes":      "N/A",
            "treatment":   ["No treatment needed."],
            "prevention":  [
                "Continue routine scouting every 7–10 days.",
                "Maintain balanced fertilization program.",
                "Plan crop rotation for next season.",
            ],
            "organic":     ["Foliar compost tea", "Mycorrhizal inoculants"],
            "fertilizer":  "Side-dress nitrogen at V6; balanced K and P.",
            "recovery":    "N/A — crop is healthy!",
        },
    },

    # ── Apple ────────────────────────────────────
    "apple": {
        "Apple___Apple_scab": {
            "display":     "Apple Scab",
            "severity":    "High",
            "pathogen":    "Venturia inaequalis",
            "symptoms":    "Olive-green to black velvety lesions on leaves and fruit. Infected fruit becomes distorted and cracked.",
            "causes":      "Cool (16–24 °C), wet spring weather; infected leaf litter provides inoculum.",
            "treatment":   [
                "Apply fungicides: myclobutanil, captan, sulfur, or mancozeb.",
                "Begin sprays at green tip and continue on 7–10 day schedule.",
                "Apply during extended wet periods (infection periods).",
            ],
            "prevention":  [
                "Plant scab-resistant apple varieties (Liberty, Enterprise, GoldRush).",
                "Rake and destroy fallen leaves in autumn.",
                "Prune trees to improve air circulation.",
                "Use a scab prediction model (RIMpro or similar) for spray timing.",
            ],
            "organic":     ["Sulfur sprays", "Copper fungicide", "Lime sulfur (dormant)"],
            "fertilizer":  "Balanced NPK; avoid excess nitrogen (increases susceptibility).",
            "recovery":    "Early season treatment prevents fruit infection; 4–6 weeks.",
        },
        "Apple___Black_rot": {
            "display":     "Apple Black Rot",
            "severity":    "High",
            "pathogen":    "Botryosphaeria obtusa",
            "symptoms":    "Circular brown leaf spots with purple margins ('frog-eye' spots). Black mummified fruit. Bark cankers on limbs.",
            "causes":      "Warm (25–30 °C) wet weather; fungus overwinters in mummified fruit and cankers.",
            "treatment":   [
                "Apply fungicides: captan, thiophanate-methyl, myclobutanil.",
                "Remove mummified fruit and cankered wood immediately.",
                "Prune out infected limbs 15 cm below visible infection.",
            ],
            "prevention":  [
                "Compost or dispose of all infected plant material.",
                "Prune to improve air circulation.",
                "Apply protective fungicide sprays during bloom and petal fall.",
                "Maintain tree vigor through proper nutrition.",
            ],
            "organic":     ["Copper-based fungicides", "Kaolin clay as physical barrier"],
            "fertilizer":  "Balanced fertilization; avoid drought stress.",
            "recovery":    "3–6 weeks; remove all inoculum sources.",
        },
        "Apple___healthy": {
            "display":     "Healthy Apple Tree",
            "severity":    "None",
            "pathogen":    "N/A",
            "symptoms":    "Glossy green leaves, no lesions or discoloration; well-formed developing fruit.",
            "causes":      "N/A",
            "treatment":   ["Maintain current orchard management practices."],
            "prevention":  [
                "Apply dormant oil spray before bud break.",
                "Maintain regular pruning schedule.",
                "Monitor for pests (codling moth, aphids) proactively.",
            ],
            "organic":     ["Dormant neem oil", "Kaolin clay", "Pheromone traps"],
            "fertilizer":  "Annual soil test; apply targeted NPK + micronutrients.",
            "recovery":    "N/A — tree is healthy!",
        },
    },

    # ── General / Unknown ────────────────────────
    "general": {
        "default": {
            "display":     "Agricultural Plant Analysis",
            "severity":    "Unknown",
            "pathogen":    "To be determined by expert",
            "symptoms":    "Various symptoms detected — see AI confidence scores above.",
            "causes":      "Multiple potential causes including pathogens, nutrient deficiency, pests, or environmental stress.",
            "treatment":   [
                "Consult a certified agronomist or plant pathologist.",
                "Collect samples and submit to a diagnostic laboratory.",
                "Apply broad-spectrum protective measures while awaiting diagnosis.",
                "Isolate affected plants from healthy ones.",
            ],
            "prevention":  [
                "Implement Integrated Pest Management (IPM) practices.",
                "Maintain regular field scouting every 5–7 days.",
                "Keep detailed records of observed symptoms and weather.",
                "Maintain crop rotation plans.",
            ],
            "organic":     ["Neem oil general spray", "Compost to boost soil health", "Beneficial microbe inoculants"],
            "fertilizer":  "Perform soil and foliar tissue tests to identify deficiencies.",
            "recovery":    "Dependent on diagnosis — seek professional advice promptly.",
        },
    },
}

# ──────────────────────────────────────────────
# Crop Care Tips (shown in sidebar)
# ──────────────────────────────────────────────
GENERAL_TIPS = [
    "🌱 **Regular Monitoring**: Scout your crops every 5–7 days for early detection.",
    "💧 **Smart Irrigation**: Use drip irrigation to keep foliage dry and reduce disease spread.",
    "🔄 **Crop Rotation**: Rotate crops every 2–3 years to break disease cycles.",
    "🌿 **Organic Matter**: Incorporate compost to improve soil structure and beneficial microbes.",
    "🧪 **Soil Testing**: Test soil every season to guide precise fertilizer application.",
    "✂️ **Pruning**: Remove dead or diseased plant material promptly.",
    "☀️ **Plant Spacing**: Ensure adequate spacing for air circulation and light penetration.",
    "🐛 **IPM Strategy**: Combine biological, cultural, and chemical control methods.",
]

# ──────────────────────────────────────────────
# Severity colour mapping
# ──────────────────────────────────────────────
SEVERITY_COLOURS = {
    "None":     "#22c55e",   # green
    "Low":      "#84cc16",   # lime
    "Medium":   "#f59e0b",   # amber
    "High":     "#ef4444",   # red
    "Critical": "#7c3aed",   # purple
    "Unknown":  "#6b7280",   # gray
}
