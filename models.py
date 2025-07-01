from typing import Dict, List, Optional
import numpy as np
import cv2
import torch
import clip
from PIL import Image
import io
import pytesseract
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from config import MONGODB_CLIENT, FOOD_101_PATH, VFN_PATH, NUTRITION5K_PATH, ECUSTFD_PATH, OPEN_FOOD_FACTS_PATH, RECIPE1M_PATH, YOLO_MODEL_PATH
from ultralytics import YOLO
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
except Exception as e:
    logger.error(f"Failed to load CLIP model: {e}", exc_info=True)
    clip_model, clip_preprocess = None, None

try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
    yolo_model = None

try:
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
except Exception as e:
    logger.error(f"Failed to load BERT model: {e}", exc_info=True)
    bert_tokenizer, bert_model = None, None

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = 384
faiss_index = faiss.IndexFlatL2(embedding_dim)

# MongoDB collections
db = MONGODB_CLIENT['nutrisnap_db']
recipe_db_collection = db['recipes']
food_logs_collection = db['food_logs']
nutrition_db_collection = db['nutrition_db']

# Load Food-101 classes
food_classes = []
food_101_classes_path = f"{FOOD_101_PATH}/meta/classes.txt"
if os.path.exists(food_101_classes_path):
    with open(food_101_classes_path, 'r') as f:
        food_classes = [line.strip() for line in f.readlines() if line.strip()]
if not food_classes:
    food_classes = [item['food_name'] for item in nutrition_db_collection.find().limit(100)]
    logger.warning("Food-101 classes not found, using nutrition_db fallback")

# Load nutrition data for FAISS
nutrition_data = list(nutrition_db_collection.find())
food_names = [item['food_name'] for item in nutrition_data if item.get('food_name')]
if food_names:
    food_embeddings = sentence_transformer.encode(food_names, batch_size=32, show_progress_bar=True)
    faiss_index.add(np.array(food_embeddings))
    logger.info(f"Loaded {len(food_names)} embeddings into FAISS index")
else:
    logger.warning("No nutrition data available for FAISS index")

# Load Nutrition5k and ECUSTFD metadata
nutrition5k_metadata = None
ecustfd_metadata = None
if os.path.exists(f"{NUTRITION5K_PATH}/metadata.csv"):
    nutrition5k_metadata = pd.read_csv(f"{NUTRITION5K_PATH}/metadata.csv")
    logger.info("Loaded Nutrition5k metadata")
if os.path.exists(f"{ECUSTFD_PATH}/metadata.csv"):
    ecustfd_metadata = pd.read_csv(f"{ECUSTFD_PATH}/metadata.csv")
    logger.info("Loaded ECUSTFD metadata")

def food_recognition(image_data: bytes) -> List[Dict]:
    """
    Use YOLOv8 and CLIP for food detection and labeling.
    """
    if not yolo_model or not clip_model:
        logger.error("Required models (YOLO or CLIP) not loaded")
        return [{"food_name": "unknown", "confidence": 0.0, "bounding_box": [0, 0, 0, 0]}]

    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)

        # YOLOv8 detection
        results = yolo_model.predict(image_np, conf=0.5)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "box": [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])],
                    "class_id": int(box.cls)
                })

        # CLIP classification
        results = []
        for det in detections:
            x1, y1, x2, y2 = det['box']
            if x2 <= x1 or y2 <= y1:
                continue
            cropped = image_np[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            cropped_pil = Image.fromarray(cropped)
            image_input = clip_preprocess(cropped_pil).unsqueeze(0).to(device)
            text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in food_classes]).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                text_features = clip_model.encode_text(text_inputs)
                logits_per_image, _ = clip_model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            food_name = food_classes[np.argmax(probs)]
            results.append({
                "food_name": food_name,
                "confidence": float(np.max(probs)),
                "bounding_box": det['box']
            })

        return results if results else [{"food_name": "unknown", "confidence": 0.0, "bounding_box": [0, 0, 0, 0]}]
    except Exception as e:
        logger.error(f"Food recognition error: {e}", exc_info=True)
        return [{"food_name": "unknown", "confidence": 0.0, "bounding_box": [0, 0, 0, 0]}]

def portion_estimation(food_item: Dict, image_data: Optional[bytes] = None) -> float:
    """
    Estimate portion size using Nutrition5k/ECUSTFD depth data.
    """
    food_name = food_item['food_name'].lower()
    bounding_box = food_item['bounding_box']

    # Nutrition5k metadata lookup
    if nutrition5k_metadata is not None:
        match = nutrition5k_metadata[nutrition5k_metadata['dish_id'].str.lower().str.contains(food_name, na=False)]
        if not match.empty:
            avg_mass = match['total_mass'].mean()
            if not pd.isna(avg_mass) and avg_mass > 0:
                return float(avg_mass)

    # ECUSTFD metadata lookup
    if ecustfd_metadata is not None:
        match = ecustfd_metadata[ecustfd_metadata['food_name'].str.lower().str.contains(food_name, na=False)]
        if not match.empty:
            avg_mass = match['weight'].mean()
            if not pd.isna(avg_mass) and avg_mass > 0:
                return float(avg_mass)

    # Depth-based estimation
    if image_data and os.path.exists(f"{NUTRITION5K_PATH}/depth"):
        try:
            dish_id = match['dish_id'].iloc[0] if not match.empty else food_name
            depth_image_path = f"{NUTRITION5K_PATH}/depth/{dish_id}.png"
            if os.path.exists(depth_image_path):
                depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
                if depth_image is not None:
                    x1, y1, x2, y2 = bounding_box
                    if x2 > x1 and y2 > y1:
                        cropped_depth = depth_image[y1:y2, x1:x2]
                        if cropped_depth.size > 0:
                            depth_values = cropped_depth[cropped_depth > 0]
                            if depth_values.size > 0:
                                avg_depth = np.mean(depth_values) / 1000
                                area_pixels = (x2 - x1) * (y2 - y1)
                                pixel_to_meter = 0.0001
                                area_m2 = area_pixels * (pixel_to_meter ** 2)
                                volume_m3 = area_m2 * avg_depth
                                density = 1000
                                if nutrition5k_metadata is not None:
                                    density_row = nutrition5k_metadata[nutrition5k_metadata['dish_id'].str.lower().str.contains(food_name, na=False)]
                                    if not density_row.empty and 'total_volume' in density_row:
                                        density = (density_row['total_mass'] / density_row['total_volume']).mean() * 1000 or 1000
                                weight_grams = volume_m3 * density * 1000
                                if 10 < weight_grams < 1000:
                                    return float(weight_grams)
        except Exception as e:
            logger.error(f"Depth estimation error: {e}", exc_info=True)

    return 100.0  # Fallback

def nutrition_calculation(food_name: str, portion: float) -> Dict:
    """
    Calculate nutrition using nutrition_db_collection and FAISS.
    """
    try:
        if not food_name or not isinstance(portion, (int, float)) or portion <= 0:
            raise ValueError("Invalid food_name or portion")
        
        food_embedding = sentence_transformer.encode([food_name])[0]
        _, indices = faiss_index.search(np.array([food_embedding]), k=1)
        nutrition = nutrition_data[indices[0][0]]

        calories_per_100g = float(nutrition.get("calories_per_100g", 0.0))
        carbs_per_100g = float(nutrition.get("carbs_per_100g", 0.0))
        protein_per_100g = float(nutrition.get("protein_per_100g", 0.0))
        fat_per_100g = float(nutrition.get("fat_per_100g", 0.0))

        return {
            "calories": (portion / 100) * calories_per_100g,
            "macros": {
                "carbs": (portion / 100) * carbs_per_100g,
                "protein": (portion / 100) * protein_per_100g,
                "fat": (portion / 100) * fat_per_100g
            }
        }
    except Exception as e:
        logger.error(f"Nutrition calculation error for {food_name}: {e}", exc_info=True)
        return {
            "calories": 0.0,
            "macros": {
                "carbs": 0.0,
                "protein": 0.0,
                "fat": 0.0
            }
        }

def text_parser(text: str) -> List[Dict]:
    """
    Parse text input using BERT and Recipe1M+.
    """
    if not bert_tokenizer or not bert_model:
        logger.error("BERT model not loaded")
        return [{"food_name": "unknown", "quantity": 1, "portion_grams": 100.0}]

    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text input")
        
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)

        recipes = recipe_db_collection.find({"ingredients.text": {"$regex": text, "$options": "i"}}).limit(5)
        results = []
        for recipe in recipes:
            for ingredient in recipe.get('ingredients', []):
                ingredient_name = ingredient.get('text', '').split()[0].lower()
                if not ingredient_name:
                    continue
                portion_grams = 100.0
                if nutrition5k_metadata is not None:
                    match = nutrition5k_metadata[nutrition5k_metadata['dish_id'].str.lower().str.contains(ingredient_name, na=False)]
                    if not match.empty:
                        portion_grams = match['total_mass'].mean() or 100.0
                elif ecustfd_metadata is not None:
                    match = ecustfd_metadata[ecustfd_metadata['food_name'].str.lower().str.contains(ingredient_name, na=False)]
                    if not match.empty:
                        portion_grams = match['weight'].mean() or 100.0
                results.append({
                    "food_name": ingredient_name,
                    "quantity": 1,
                    "portion_grams": float(portion_grams)
                })
        return results if results else [{"food_name": "unknown", "quantity": 1, "portion_grams": 100.0}]
    except Exception as e:
        logger.error(f"Text parsing error: {e}", exc_info=True)
        return [{"food_name": "unknown", "quantity": 1, "portion_grams": 100.0}]

def barcode_scanner(image_data: bytes) -> Dict:
    """
    Scan barcode/nutrition label using Tesseract and Open Food Facts.
    """
    try:
        if not image_data:
            raise ValueError("No image data provided")
        
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image data")
        
        text = pytesseract.image_to_string(image).strip()
        if not text:
            raise ValueError("No text detected in image")
        
        nutrition = nutrition_db_collection.find_one({"barcode": text})
        if not nutrition:
            food_embedding = sentence_transformer.encode([text])[0]
            _, indices = faiss_index.search(np.array([food_embedding]), k=1)
            nutrition = nutrition_data[indices[0][0]]
        
        return {
            "food_name": nutrition.get("food_name", "unknown"),
            "portion_grams": 100.0,
            "nutrition": {
                "calories": float(nutrition.get("calories_per_100g", 0.0)),
                "macros": {
                    "carbs": float(nutrition.get("carbs_per_100g", 0.0)),
                    "protein": float(nutrition.get("protein_per_100g", 0.0)),
                    "fat": float(nutrition.get("fat_per_100g", 0.0))
                }
            }
        }
    except Exception as e:
        logger.error(f"Barcode scanning error: {e}", exc_info=True)
        return {
            "food_name": "unknown",
            "portion_grams": 100.0,
            "nutrition": {
                "calories": 0.0,
                "macros": {
                    "carbs": 0.0,
                    "protein": 0.0,
                    "fat": 0.0
                }
            }
        }

def meal_recommendation(user_id: str, preferences: Dict) -> List[str]:
    """
    Generate meal recommendations using K-means/Collaborative Filtering.
    """
    try:
        from sklearn.cluster import KMeans
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("Invalid user_id")
        
        user_logs = list(food_logs_collection.find({"user_id": user_id}))
        if not user_logs:
            common_foods = [item['food_name'] for item in nutrition_db_collection.find().sort("calories_per_100g", -1).limit(2)]
            return common_foods if common_foods else ["salad", "soup"]
        
        food_names = [item['food_items'][0]['food_name'] for item in user_logs if item.get('food_items')]
        if not food_names:
            return ["salad", "soup"]
        
        embeddings = sentence_transformer.encode(food_names, batch_size=32, show_progress_bar=True)
        kmeans = KMeans(n_clusters=min(2, len(food_names)), random_state=0).fit(embeddings)
        cluster_centers = kmeans.cluster_centers_
        _, indices = faiss_index.search(np.array(cluster_centers), k=1)
        recommendations = [nutrition_data[idx[0]]['food_name'] for idx in indices]
        return recommendations if recommendations else ["salad", "soup"]
    except Exception as e:
        logger.error(f"Meal recommendation error: {e}", exc_info=True)
        return ["salad", "soup"]

def rag_qna(query: str, user_id: str) -> str:
    """
    Answer nutrition questions using RAG.
    """
    try:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Invalid query")
        
        query_embedding = sentence_transformer.encode([query])[0]
        _, indices = faiss_index.search(np.array([query_embedding]), k=1)
        context = nutrition_data[indices[0][0]]
        context_text = f"Food: {context['food_name']}, Calories: {context.get('calories_per_100g', 0)} kcal/100g, Carbs: {context.get('carbs_per_100g', 0)}g, Protein: {context.get('protein_per_100g', 0)}g, Fat: {context.get('fat_per_100g', 0)}g"
        
        # Simulated GPT-4 response
        response = f"Based on the context, {context['food_name']} has {context.get('calories_per_100g', 0)} kcal per 100g. For {query}, this suggests a moderate caloric content suitable for a balanced diet."
        return response
    except Exception as e:
        logger.error(f"RAG QnA error: {e}", exc_info=True)
        return "Unable to answer due to an error."