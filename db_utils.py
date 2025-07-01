from pymongo import MongoClient
from typing import Dict, List
from config import MONGODB_CLIENT, MONGODB_URI, USDA_FOODDATA_PATH, OPEN_FOOD_FACTS_PATH, NUTRITION5K_PATH, ECUSTFD_PATH, RECIPE1M_PATH
import pandas as pd
import json
import logging
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB collections
db = MONGODB_CLIENT['nutrisnap_db']
users_collection = db['users']
food_logs_collection = db['food_logs']
feedback_collection = db['feedback']
nutrition_db_collection = db['nutrition_db']
recipe_db_collection = db['recipes']

# Initialize Sentence Transformers and FAISS
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = 384
faiss_index = faiss.IndexFlatL2(embedding_dim)

def load_nutrition_db():
    """
    Load Nutrition5k, ECUSTFD, USDA FoodData Central, Open Food Facts into MongoDB and FAISS.
    """
    try:
        nutrition_db_collection.delete_many({})
        faiss_index.reset()
        all_records = []

        # Load Nutrition5k
        nutrition5k_path = f"{NUTRITION5K_PATH}/metadata.csv"
        if os.path.exists(nutrition5k_path):
            nutrition5k_data = pd.read_csv(nutrition5k_path)
            records = nutrition5k_data[['dish_id', 'total_calories', 'total_mass', 'total_carb', 'total_protein', 'total_fat']].dropna().to_dict('records')
            formatted_records = [
                {
                    'food_name': str(r['dish_id']).strip(),
                    'calories_per_100g': float((r['total_calories'] / r['total_mass']) * 100) if r['total_mass'] > 0 else 0.0,
                    'carbs_per_100g': float((r['total_carb'] / r['total_mass']) * 100) if r['total_mass'] > 0 else 0.0,
                    'protein_per_100g': float((r['total_protein'] / r['total_mass']) * 100) if r['total_mass'] > 0 else 0.0,
                    'fat_per_100g': float((r['total_fat'] / r['total_mass']) * 100) if r['total_mass'] > 0 else 0.0,
                    'barcode': '',
                    'source': 'Nutrition5k'
                } for r in records if str(r['dish_id']).strip() and all(r.get(k, 0) >= 0 for k in ['total_calories', 'total_carb', 'total_protein', 'total_fat'])
            ]
            all_records.extend(formatted_records)
            logger.info(f"Loaded {len(formatted_records)} records from Nutrition5k")

        # Load ECUSTFD
        ecustfd_path = f"{ECUSTFD_PATH}/metadata.csv"
        if os.path.exists(ecustfd_path):
            ecustfd_data = pd.read_csv(ecustfd_path)
            records = ecustfd_data[['food_name', 'calories', 'carbs', 'protein', 'fat']].dropna().to_dict('records')
            formatted_records = [
                {
                    'food_name': str(r['food_name']).strip(),
                    'calories_per_100g': float(r['calories']) if pd.notna(r['calories']) and r['calories'] >= 0 else 0.0,
                    'carbs_per_100g': float(r['carbs']) if pd.notna(r['carbs']) and r['carbs'] >= 0 else 0.0,
                    'protein_per_100g': float(r['protein']) if pd.notna(r['protein']) and r['protein'] >= 0 else 0.0,
                    'fat_per_100g': float(r['fat']) if pd.notna(r['fat']) and r['fat'] >= 0 else 0.0,
                    'barcode': '',
                    'source': 'ECUSTFD'
                } for r in records if str(r['food_name']).strip()
            ]
            all_records.extend(formatted_records)
            logger.info(f"Loaded {len(formatted_records)} records from ECUSTFD")

        # Load Open Food Facts
        if os.path.exists(OPEN_FOOD_FACTS_PATH):
            nutrition_data = pd.read_csv(OPEN_FOOD_FACTS_PATH, low_memory=False)
            records = nutrition_data[['product_name', 'energy_100g', 'carbohydrates_100g', 'proteins_100g', 'fat_100g', 'code']].dropna().to_dict('records')
            formatted_records = [
                {
                    'food_name': str(r['product_name']).strip(),
                    'calories_per_100g': float(r['energy_100g']) / 4.184 if pd.notna(r['energy_100g']) and r['energy_100g'] >= 0 else 0.0,  # Convert kJ to kcal
                    'carbs_per_100g': float(r['carbohydrates_100g']) if pd.notna(r['carbohydrates_100g']) and r['carbohydrates_100g'] >= 0 else 0.0,
                    'protein_per_100g': float(r['proteins_100g']) if pd.notna(r['proteins_100g']) and r['proteins_100g'] >= 0 else 0.0,
                    'fat_per_100g': float(r['fat_100g']) if pd.notna(r['fat_100g']) and r['fat_100g'] >= 0 else 0.0,
                    'barcode': str(r['code']).strip(),
                    'source': 'Open Food Facts'
                } for r in records if str(r['product_name']).strip()
            ]
            all_records.extend(formatted_records)
            logger.info(f"Loaded {len(formatted_records)} records from Open Food Facts")

        # Load USDA FoodData Central
        food_path = f"{USDA_FOODDATA_PATH}/food.csv"
        nutrient_path = f"{USDA_FOODDATA_PATH}/food_nutrient.csv"
        if os.path.exists(food_path) and os.path.exists(nutrient_path):
            food_data = pd.read_csv(food_path)
            nutrient_data = pd.read_csv(nutrient_path)
            records = []
            for _, food in food_data.iterrows():
                nutrients = nutrient_data[nutrient_data['fdc_id'] == food['fdc_id']]
                record = {
                    'food_name': str(food['description']).strip(),
                    'calories_per_100g': float(next((n['amount'] for n in nutrients.to_dict('records') if n['nutrient_name'] == 'Energy'), 0.0)),
                    'carbs_per_100g': float(next((n['amount'] for n in nutrients.to_dict('records') if n['nutrient_name'] == 'Carbohydrate, by difference'), 0.0)),
                    'protein_per_100g': float(next((n['amount'] for n in nutrients.to_dict('records') if n['nutrient_name'] == 'Protein'), 0.0)),
                    'fat_per_100g': float(next((n['amount'] for n in nutrients.to_dict('records') if n['nutrient_name'] == 'Total lipid (fat)'), 0.0)),
                    'barcode': '',
                    'source': 'USDA'
                }
                if record['food_name'] and all(record[k] >= 0 for k in ['calories_per_100g', 'carbs_per_100g', 'protein_per_100g', 'fat_per_100g']):
                    records.append(record)
            all_records.extend(records)
            logger.info(f"Loaded {len(records)} records from USDA FoodData Central")

        # Insert into MongoDB
        if all_records:
            nutrition_db_collection.insert_many(all_records)
            logger.info(f"Inserted {len(all_records)} records into nutrition_db")

            # Update FAISS index
            food_names = [r['food_name'] for r in all_records if r['food_name']]
            if food_names:
                food_embeddings = sentence_transformer.encode(food_names, batch_size=32, show_progress_bar=True)
                faiss_index.add(np.array(food_embeddings))
                logger.info(f"Added {len(food_names)} embeddings to FAISS index")
            else:
                logger.warning("No valid food names found for FAISS index")

    except Exception as e:
        logger.error(f"Error loading nutrition data: {e}", exc_info=True)
        raise

    return nutrition_db_collection

def load_recipe_db():
    """
    Load Recipe1M+ into MongoDB for text parsing.
    """
    try:
        recipe_path = f"{RECIPE1M_PATH}/layer1.json"
        if os.path.exists(recipe_path):
            with open(recipe_path, 'r') as f:
                recipes = json.load(f)
            recipe_db_collection.delete_many({})
            formatted_recipes = [
                {
                    'recipe_id': str(r.get('id', '')).strip(),
                    'title': str(r.get('title', '')).strip(),
                    'ingredients': [
                        {'text': str(ing).strip()} for ing in r.get('ingredients', [])
                        if str(ing).strip()
                    ],
                    'instructions': r.get('instructions', []),
                    'source': 'Recipe1M+'
                } for r in recipes if str(r.get('title', '')).strip() and r.get('ingredients')
            ]
            if formatted_recipes:
                recipe_db_collection.insert_many(formatted_recipes)
                logger.info(f"Loaded {len(formatted_recipes)} recipes into recipes collection")
            else:
                logger.warning("No valid recipes found in Recipe1M+")
    except Exception as e:
        logger.error(f"Error loading recipe data: {e}", exc_info=True)
        raise

    return recipe_db_collection

def save_food_log(food_log: Dict):
    """
    Save food log to MongoDB.
    """
    try:
        if not all(key in food_log for key in ['_id', 'user_id', 'timestamp', 'food_items']):
            raise ValueError("Invalid food log format")
        food_logs_collection.insert_one(food_log)
        logger.info(f"Saved food log: {food_log['_id']}")
    except Exception as e:
        logger.error(f"Error saving food log: {e}", exc_info=True)
        raise

def save_feedback(feedback: Dict, food_log_id: str):
    """
    Save user feedback and update food log.
    """
    try:
        if not all(key in feedback for key in ['_id', 'user_id', 'food_log_id', 'corrections', 'timestamp']):
            raise ValueError("Invalid feedback format")
        feedback_collection.insert_one(feedback)
        food_logs_collection.update_one(
            {"_id": food_log_id},
            {"$set": {"food_items": feedback.get('corrections', [])}}
        )
        logger.info(f"Saved feedback and updated food log: {food_log_id}")
    except Exception as e:
        logger.error(f"Error saving feedback: {e}", exc_info=True)
        raise

def get_user_preferences(user_id: str) -> Dict:
    """
    Fetch user preferences from MongoDB.
    """
    try:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("Invalid user_id")
        user = users_collection.find_one({"_id": user_id})
        return user.get('preferences', {}) if user else {}
    except Exception as e:
        logger.error(f"Error fetching user preferences: {e}", exc_info=True)
        return {}

def get_food_logs(user_id: str) -> List[Dict]:
    """
    Fetch food logs for a user, sorted by timestamp.
    """
    try:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("Invalid user_id")
        return list(food_logs_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(50))
    except Exception as e:
        logger.error(f"Error fetching food logs: {e}", exc_info=True)
        return []