import io
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from PIL import Image
from ultralytics import YOLO

# ============ CLASS NAMES ============
FOOD_CLASS_NAMES = [
    "Apple pie", "Baby back ribs", "Baklava", "Beef carpaccio", "Beef tartare",
    "Beet salad", "Beignets", "Bibimbap", "Bread pudding", "Breakfast burrito",
    "Bruschetta", "Caesar salad", "Cannoli", "Caprese salad", "Carrot cake",
    "Ceviche", "Cheesecake", "Cheese plate", "Chicken curry", "Chicken quesadilla",
    "Chicken wings", "Chocolate cake", "Chocolate mousse", "Churros", "Clam chowder",
    "Club sandwich", "Crab cakes", "Creme brulee", "Croque madame", "Cup cakes",
    "Deviled eggs", "Donuts", "Dumplings", "Edamame", "Eggs benedict",
    "Escargots", "Falafel", "Filet mignon", "Fish and chips", "Foie gras",
    "French fries", "French onion soup", "French toast", "Fried calamari", "Fried rice",
    "Frozen yogurt", "Garlic bread", "Gnocchi", "Greek salad", "Grilled cheese sandwich",
    "Grilled salmon", "Guacamole", "Gyoza", "Hamburger", "Hot and sour soup",
    "Hot dog", "Huevos rancheros", "Hummus", "Ice cream", "Lasagna",
    "Lobster bisque", "Lobster roll sandwich", "Macaroni and cheese", "Macarons", "Miso soup",
    "Mussels", "Nachos", "Omelette", "Onion rings", "Oysters",
    "Pad thai", "Paella", "Pancakes", "Panna cotta", "Peking duck",
    "Pho", "Pizza", "Pork chop", "Poutine", "Prime rib",
    "Pulled pork sandwich", "Ramen", "Ravioli", "Red velvet cake", "Risotto",
    "Samosa", "Sashimi", "Scallops", "Seaweed salad", "Shrimp and grits",
    "Spaghetti bolognese", "Spaghetti carbonara", "Spring rolls", "Steak", "Strawberry shortcake",
    "Sushi", "Tacos", "Takoyaki", "Tiramisu", "Tuna tartare",
    "Waffles",
]

FRUIT_CLASS_NAMES = [
    "Apple", "Apricot", "Avocado", "Banana", "Black Berry", "Blueberry", "Cherry", "Coconut",
    "Cranberry", "Dragonfruit", "Durian", "Grape", "Grapefruit", "Guava", "Jackfruit", "Kiwi",
    "Lemon", "Lime", "Lychee", "Mango", "Mangosteen", "Melon Pear", "Olive", "Orange", "Papaya",
    "Passion Fruit", "Raspberry", "Salak", "Sapodilla", "Strawberry", "Tomato", "Watermelon"
]

YOLO_CLASS_NAMES = [
    "Ayam bakar", "Ayam goreng", "Bakso", "Bakwan", "Batagor", "Bihun", "Capcay", "Gado-gado",
    "Ikan goreng", "Kerupuk", "Martabak telor", "Mie", "Nasi goreng", "Nasi putih", "Nugget",
    "Opor ayam", "Pempek", "Rendang", "Roti", "Soto", "Steak", "Tahu", "Telur", "Tempe",
    "Terong balado", "Tumis kangkung", "Udang", "Sate", "Sosis"
]

# ============ FASTAPI INIT ============
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ LOAD MODELS ============
FOOD_MODEL_PATH = "mobilenetv3_food41.keras"
FRUIT_MODEL_PATH = "final_fruit_mobilenetv3.keras"
YOLO_MODEL_PATH = "indo_yolo.pt"

# ======== CEK FILE MODEL (DEBUG) =========
def debug_file(path):
    print(f"== Debug info for {path} ==")
    try:
        size = os.path.getsize(path)
        print(f"File size: {size/1024/1024:.2f} MB")
    except Exception as e:
        print(f"File not found or error reading size: {e}")
        return
    try:
        with open(path, "rb") as f:
            head = f.read(16)
        print(f"First 16 bytes: {head}")
    except Exception as e:
        print(f"Error reading file head: {e}")
    print("=================================")

print("Loading food model...")
debug_file(FOOD_MODEL_PATH)
food_model = load_model(FOOD_MODEL_PATH, compile=False)

print("Loading fruit model...")
debug_file(FRUIT_MODEL_PATH)
fruit_model = load_model(FRUIT_MODEL_PATH, compile=False)

print("Loading YOLO model...")
debug_file(YOLO_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)
print("All models loaded.")

# ============ PREDICT FUNCTIONS ============

def predict_image(image_bytes, model, class_names):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image).astype("float32")
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        idx = int(np.argmax(pred[0]))
        conf = float(np.max(pred[0]))
        label = class_names[idx]
        return label, conf
    except Exception as e:
        print(f"Error in predict_image: {e}")
        return None, None

def predict_yolo(image_bytes, yolo_model):
    try:
        import cv2
        img_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        results = yolo_model(img)
        result = results[0]
        out_boxes = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = result.names[class_id]
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            out_boxes.append({
                "class": name,
                "confidence": conf,
                "box": bbox
            })
        return out_boxes
    except Exception as e:
        print(f"Error in predict_yolo: {e}")
        return []

# ============ ENDPOINTS ============

@app.post("/predict/food")
async def predict_food(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        label, conf = predict_image(image_bytes, food_model, FOOD_CLASS_NAMES)
        if label is None:
            return {"error": "Failed to process image."}
        return {"class": label, "confidence": conf}
    except Exception as e:
        print(f"Error in predict_food endpoint: {e}")
        return {"error": str(e)}

@app.post("/predict/fruit")
async def predict_fruit(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        label, conf = predict_image(image_bytes, fruit_model, FRUIT_CLASS_NAMES)
        if label is None:
            return {"error": "Failed to process image."}
        return {"class": label, "confidence": conf}
    except Exception as e:
        print(f"Error in predict_fruit endpoint: {e}")
        return {"error": str(e)}

@app.post("/predict/yolo")
async def predict_yolo_api(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        results = predict_yolo(image_bytes, yolo_model)
        return {"results": results}
    except Exception as e:
        print(f"Error in predict_yolo endpoint: {e}")
        return {"error": str(e)}

@app.get("/")
def root():
    return {
        "message": "MealMind API running (Keras & YOLO)!",
        "food_model": FOOD_MODEL_PATH,
        "fruit_model": FRUIT_MODEL_PATH,
        "yolo_model": YOLO_MODEL_PATH,
        "endpoints": ["/predict/food", "/predict/fruit", "/predict/yolo"]
    }
    