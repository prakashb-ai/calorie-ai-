from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from models import food_recognition, portion_estimation, nutrition_calculation, text_parser, barcode_scanner, meal_recommendation, rag_qna
from db_utils import save_food_log, save_feedback, get_user_preferences, get_food_logs
from config import MONGODB_CLIENT, MONGODB_URI
import jwt
import uuid
import datetime
import logging
import os

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

# MongoDB client
db = MONGODB_CLIENT['nutrisnap_db']

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
app.config['JWT_SECRET'] = os.getenv('JWT_SECRET', 'your-secret-key')  # Set in environment
HOST = os.getenv('FLASK_HOST', '0.0.0.0')
PORT = int(os.getenv('FLASK_PORT', 5000))
DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

def validate_user_id(user_id: str) -> bool:
    """Validate user_id format."""
    return isinstance(user_id, str) and len(user_id.strip()) > 0 and len(user_id) <= 50

def authenticate_request(request):
    """Authenticate request using JWT."""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        logger.warning("Missing or invalid Authorization header")
        return False
    
    token = auth_header.split(' ')[1]
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
        request.user_id = payload.get('user_id')
        return True
    except jwt.ExpiredSignatureError:
        logger.error("Token expired")
        return False
    except jwt.InvalidTokenError:
        logger.error("Invalid token")
        return False

def standardize_response(status: str, data: dict = None, error: str = None) -> dict:
    """Standardize API response format."""
    response = {
        "status": status,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
    }
    if data:
        response["data"] = data
    if error:
        response["error"] = error
    return response

@app.route('/api/food/scan', methods=['POST'])
@limiter.limit("10 per minute")
def scan_food():
    """Process food image using Food-101, VFN, Nutrition5k datasets."""
    try:
        if not authenticate_request(request):
            return jsonify(standardize_response("error", error="Unauthorized")), 401

        if 'image' not in request.files:
            return jsonify(standardize_response("error", error="No image provided")), 400

        image = request.files['image']
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify(standardize_response("error", error="Invalid image format")), 400

        user_id = request.form.get('user_id')
        if not user_id or not validate_user_id(user_id):
            return jsonify(standardize_response("error", error="Valid user_id is required")), 400

        image_data = image.read()
        detected_foods = food_recognition(image_data)
        if not detected_foods:
            return jsonify(standardize_response("error", error="No foods detected")), 400

        food_log = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "timestamp": datetime.datetime.now(datetime.UTC),
            "food_items": []
        }

        for food in detected_foods:
            portion = portion_estimation(food)
            nutrition = nutrition_calculation(food['food_name'], portion)
            food_log['food_items'].append({
                "food_name": food['food_name'],
                "portion_grams": portion,
                "nutrition": nutrition,
                "confidence": food['confidence'],
                "bounding_box": food['bounding_box']
            })

        save_food_log(food_log)
        logger.info(f"Food log saved for user {user_id}, log_id: {food_log['_id']}")
        return jsonify(standardize_response("success", data={"food_log_id": food_log['_id'], "food_items": food_log['food_items']})), 201

    except Exception as e:
        logger.error(f"Error in scan_food: {str(e)}", exc_info=True)
        return jsonify(standardize_response("error", error="Internal server error")), 500

@app.route('/api/food/text', methods=['POST'])
@limiter.limit("10 per minute")
def parse_text_input():
    """Parse text input using Recipe1M+ dataset."""
    try:
        if not authenticate_request(request):
            return jsonify(standardize_response("error", error="Unauthorized")), 401

        data = request.get_json()
        if not data or 'user_id' not in data or 'text' not in data:
            return jsonify(standardize_response("error", error="user_id and text are required")), 400

        user_id = data['user_id']
        text_input = data['text']
        if not validate_user_id(user_id) or not isinstance(text_input, str) or len(text_input.strip()) > 1000:
            return jsonify(standardize_response("error", error="Invalid user_id or text (max 1000 chars)")), 400

        parsed_items = text_parser(text_input)
        if not parsed_items:
            return jsonify(standardize_response("error", error="No valid food items parsed")), 400

        food_log = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "timestamp": datetime.datetime.now(datetime.UTC),
            "food_items": []
        }

        for item in parsed_items:
            nutrition = nutrition_calculation(item['food_name'], item['portion_grams'])
            food_log['food_items'].append({
                "food_name": item['food_name'],
                "portion_grams": item['portion_grams'],
                "nutrition": nutrition
            })

        save_food_log(food_log)
        logger.info(f"Text food log saved for user {user_id}, log_id: {food_log['_id']}")
        return jsonify(standardize_response("success", data={"food_log_id": food_log['_id'], "food_items": food_log['food_items']})), 201

    except Exception as e:
        logger.error(f"Error in parse_text_input: {str(e)}", exc_info=True)
        return jsonify(standardize_response("error", error="Internal server error")), 500

@app.route('/api/food/barcode', methods=['POST'])
@limiter.limit("10 per minute")
def scan_barcode():
    """Scan barcode/nutrition label using Open Food Facts."""
    try:
        if not authenticate_request(request):
            return jsonify(standardize_response("error", error="Unauthorized")), 401

        if 'image'  not in request.files:
            return jsonify(standardize_response("error", error="No image provided")), 400

        image = request.files['image']
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify(standardize_response("error", error="Invalid image format")), 400

        user_id = request.form.get('user_id')
        if not user_id or not validate_user_id(user_id):
            return jsonify(standardize_response("error", error="Valid user_id is required")), 400

        image_data = image.read()
        nutrition_data = barcode_scanner(image_data)
        if nutrition_data['food_name'] == "unknown":
            return jsonify(standardize_response("error", error="Unable to identify food from barcode")), 400

        food_log = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "timestamp": datetime.datetime.now(datetime.UTC),
            "food_items": [{
                "food_name": nutrition_data['food_name'],
                "portion_grams": nutrition_data.get('portion_grams', 100.0),
                "nutrition": nutrition_data['nutrition']
            }]
        }

        save_food_log(food_log)
        logger.info(f"Barcode food log saved for user {user_id}, log_id: {food_log['_id']}")
        return jsonify(standardize_response("success", data={"food_log_id": food_log['_id'], "food_items": food_log['food_items']})), 201

    except Exception as e:
        logger.error(f"Error in scan_barcode: {str(e)}", exc_info=True)
        return jsonify(standardize_response("error", error="Internal server error")), 500

@app.route('/api/food/qna', methods=['POST'])
@limiter.limit("20 per minute")
def food_qna():
    """Answer nutrition-related questions using RAG."""
    try:
        if not authenticate_request(request):
            return jsonify(standardize_response("error", error="Unauthorized")), 401

        data = request.get_json()
        if not data or 'user_id' not in data or 'query' not in data:
            return jsonify(standardize_response("error", error="user_id and query are required")), 400

        user_id = data['user_id']
        query = data['query']
        if not validate_user_id(user_id) or not isinstance(query, str) or len(query.strip()) > 500:
            return jsonify(standardize_response("error", error="Invalid user_id or query (max 500 chars)")), 400

        answer = rag_qna(query, user_id)
        return jsonify(standardize_response("success", data={"user_id": user_id, "query": query, "answer": answer})), 200

    except Exception as e:
        logger.error(f"Error in food_qna: {str(e)}", exc_info=True)
        return jsonify(standardize_response("error", error="Internal server error")), 500

@app.route('/api/food/feedback', methods=['POST'])
@limiter.limit("10 per minute")
def submit_feedback():
    """Store user feedback for online learning."""
    try:
        if not authenticate_request(request):
            return jsonify(standardize_response("error", error="Unauthorized")), 401

        data = request.get_json()
        if not data or 'user_id' not in data or 'food_log_id' not in data or 'corrections' not in data:
            return jsonify(standardize_response("error", error="user_id, food_log_id, and corrections are required")), 400

        user_id = data['user_id']
        food_log_id = data['food_log_id']
        corrections = data['corrections']
        if not validate_user_id(user_id) or not isinstance(food_log_id, str) or not isinstance(corrections, list):
            return jsonify(standardize_response("error", error="Invalid user_id, food_log_id, or corrections")), 400

        feedback = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "food_log_id": food_log_id,
            "corrections": corrections,
            "timestamp": datetime.datetime.now(datetime.UTC)
        }

        save_feedback(feedback, food_log_id)
        logger.info(f"Feedback saved for food_log {food_log_id}, feedback_id: {feedback['_id']}")
        return jsonify(standardize_response("success", data={"message": "Feedback submitted successfully", "feedback_id": feedback['_id']})), 201

    except Exception as e:
        logger.error(f"Error in submit_feedback: {str(e)}", exc_info=True)
        return jsonify(standardize_response("error", error="Internal server error")), 500

@app.route('/api/food/recommendations', methods=['GET'])
@limiter.limit("20 per minute")
def get_recommendations():
    """Get personalized meal recommendations."""
    try:
        if not authenticate_request(request):
            return jsonify(standardize_response("error", error="Unauthorized")), 401

        user_id = request.args.get('user_id')
        if not user_id or not validate_user_id(user_id):
            return jsonify(standardize_response("error", error="Valid user_id is required")), 400

        preferences = get_user_preferences(user_id)
        recommendations = meal_recommendation(user_id, preferences)
        return jsonify(standardize_response("success", data={"user_id": user_id, "recommendations": recommendations})), 200

    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
        return jsonify(standardize_response("error", error="Internal server error")), 500

@app.route('/api/food/tracker', methods=['GET'])
@limiter.limit("20 per minute")
def get_food_tracker():
    """Get food log history and insights."""
    try:
        if not authenticate_request(request):
            return jsonify(standardize_response("error", error="Unauthorized")), 401

        user_id = request.args.get('user_id')
        if not user_id or not validate_user_id(user_id):
            return jsonify(standardize_response("error", error="Valid user_id is required")), 400

        logs = get_food_logs(user_id)
        logs_list = []
        total_calories = 0

        for log in logs:
            log['_id'] = str(log['_id'])
            logs_list.append(log)
            for item in log['food_items']:
                total_calories += item['nutrition']['calories']

        return jsonify(standardize_response("success", data={
            "user_id": user_id,
            "food_logs": logs_list,
            "total_calories": round(total_calories, 2)
        })), 200

    except Exception as e:
        logger.error(f"Error in get_food_tracker: {str(e)}", exc_info=True)
        return jsonify(standardize_response("error", error="Internal server error")), 500

if __name__ == '__main__':
    app.run(debug=DEBUG, host=HOST, port=PORT)