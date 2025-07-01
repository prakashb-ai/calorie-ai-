import os
import logging
from pymongo import MongoClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
try:
    MONGODB_CLIENT = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    MONGODB_CLIENT.admin.command('ping')
    logger.info("MongoDB connection verified for config")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
    raise SystemExit("MongoDB connection failure")

# Dataset paths
DATASET_PATHS = {
    'FOOD_101_PATH': {
        'path': os.getenv('FOOD_101_PATH', '/data/food-101'),
        'required_files': ['meta/classes.txt'],
        'description': 'Food-101 dataset (images + classes.txt for food recognition)'
    },
    'UECFOOD_256_PATH': {
        'path': os.getenv('UECFOOD_256_PATH', '/data/uecfood-256'),
        'required_files': ['category.txt'],
        'description': 'UECFOOD-256 dataset (images + category.txt for food recognition)'
    },
    'VFN_PATH': {
        'path': os.getenv('VFN_PATH', '/data/vfn'),
        'required_files': [],
        'description': 'VFN dataset (images + annotations for YOLOv8 training)'
    },
    'FOODSEG103_PATH': {
        'path': os.getenv('FOODSEG103_PATH', '/data/foodseg103'),
        'required_files': [],
        'description': 'FoodSeg103 dataset (images + annotations for YOLOv8 training)'
    },
    'NUTRITION5K_PATH': {
        'path': os.getenv('NUTRITION5K_PATH', '/data/nutrition5k'),
        'required_files': ['metadata.csv'],
        'description': 'Nutrition5k dataset (images + metadata.csv for portion estimation)'
    },
    'ECUSTFD_PATH': {
        'path': os.getenv('ECUSTFD_PATH', '/data/ecustfd'),
        'required_files': ['metadata.csv'],
        'description': 'ECUSTFD dataset (images + metadata.csv for portion estimation)'
    },
    'USDA_FOODDATA_PATH': {
        'path': os.getenv('USDA_FOODDATA_PATH', '/data/usda_fooddata'),
        'required_files': ['food.csv', 'food_nutrient.csv'],
        'description': 'USDA FoodData Central (food.csv, food_nutrient.csv for nutrition data)'
    },
    'OPEN_FOOD_FACTS_PATH': {
        'path': os.getenv('OPEN_FOOD_FACTS_PATH', '/data/open_food_facts/products.csv'),
        'required_files': ['products.csv'],
        'description': 'Open Food Facts (products.csv for barcode and nutrition data)'
    },
    'RECIPE1M_PATH': {
        'path': os.getenv('RECIPE1M_PATH', '/data/recipe1m'),
        'required_files': ['layer1.json'],
        'description': 'Recipe1M+ dataset (layer1.json for text parsing)'
    },
    'YOLO_MODEL_PATH': {
        'path': os.getenv('YOLO_MODEL_PATH', '/models/yolov8n.pt'),
        'required_files': ['yolov8n.pt'],
        'description': 'YOLOv8 model (yolov8n.pt for food recognition)'
    }
}

# Validate dataset paths
for key, config in DATASET_PATHS.items():
    path = config['path']
    required_files = config['required_files']
    if path.startswith('/data/') and not os.path.exists(path):
        logger.warning(f"{key} path not found: {path}. Using placeholder path.")
    for file in required_files:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            logger.warning(f"Required file not found: {full_path} for {key}. Some features may be limited.")

# Export paths for backward compatibility
FOOD_101_PATH = DATASET_PATHS['FOOD_101_PATH']['path']
UECFOOD_256_PATH = DATASET_PATHS['UECFOOD_256_PATH']['path']
VFN_PATH = DATASET_PATHS['VFN_PATH']['path']
FOODSEG103_PATH = DATASET_PATHS['FOODSEG103_PATH']['path']
NUTRITION5K_PATH = DATASET_PATHS['NUTRITION5K_PATH']['path']
ECUSTFD_PATH = DATASET_PATHS['ECUSTFD_PATH']['path']
USDA_FOODDATA_PATH = DATASET_PATHS['USDA_FOODDATA_PATH']['path']
OPEN_FOOD_FACTS_PATH = DATASET_PATHS['OPEN_FOOD_FACTS_PATH']['path']
RECIPE1M_PATH = DATASET_PATHS['RECIPE1M_PATH']['path']
YOLO_MODEL_PATH = DATASET_PATHS['YOLO_MODEL_PATH']['path']