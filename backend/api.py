"""
FastAPI App for Image Classification
runs on: http://127.0.0.1:8000/
start with (from backend): fastapi dev api.py 
"""
import project_config

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet18, resnet50, convnext_tiny
import base64
import io
from PIL import Image

from computerVision.basemodels import Config, load_config
from LLM.chat_service import build_followup_with_images_prompt, get_initial_answer, get_follow_up_answer

config : Config= load_config(project_config.SELECTED_CNN_MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


# response basemodels
class FollowUpRequest(BaseModel):
    question: str | None
    images: list[str] | None

class RecipeResponse(BaseModel):
    recipe: str
    food: list[str]
    chat_history: list

class InitialAnswerRequest(BaseModel):
    images: list[str]

chat = {
    "messages": []
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular-Frontend Port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def build_model(num_classes, model):
    if model == "efficientnet_b0":
        model = efficientnet_b0(weights="DEFAULT")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )
        return model
    elif model == "resnet18":
        model = resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model == "resnet50":
        model = resnet50(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model == "convnext_tiny":
        model = convnext_tiny(weights="DEFAULT")

        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

        return model
    else:
        return

@app.on_event("startup")
def load_model():
    model = build_model(len(config.meta.classes), config.settings.model)

    model.load_state_dict(
        torch.load(config.settings.model_path, map_location=device)
    )

    model.to(device)
    model.eval()

    app.state.model = model

def decode_image(base64_string: str) -> Image.Image:
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data)).convert("RGB")

def classify_image(base64_string: str) -> dict[str]:
    """
    Multilabel image classification using PyTorch.
    Returns probabilities per class.
    """
    class_names = config.meta.classes

    model = app.state.model
    image = decode_image(base64_string)

    img_tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # build output JSON
    result = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }

    return result

def uploadFile(image_data) -> dict:
    """
    Handles image upload and classification.
    """

    # Remove base64 prefix from uploaded images
    prefixes = [
        "data:image/jpeg;base64,",
        "data:image/jpg;base64,",
        "data:image/png;base64,",
        "image='data:image/jpeg;base64,"
    ]
    for prefix in prefixes:
        if image_data.startswith(prefix):
            image_data = image_data[len(prefix):]
        

    print("Classifying Image...")        
    result = classify_image(image_data)

    # for debugging uncomment these lines
    print(result)
    return result

@app.post("/api/recipe/initial")
async def api_initial_answer(request: InitialAnswerRequest):
    """Generate initial recipe based on fridge contents"""
    fridge_contents = set()
    try:
        for image in request.images:
            result = uploadFile(image)

            for key, value in result.items():
                if value > 0.7:
                    fridge_contents.add(key)

    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    
    print(fridge_contents)
    try:
        recipe = get_initial_answer(fridge_contents)
        print(recipe)
        return RecipeResponse(recipe=recipe, food=fridge_contents, chat_history=chat["messages"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/api/recipe/followup")
async def api_follow_up_answer(request: FollowUpRequest):
    try:
        has_additional_images = bool(request.images)
        
        additional_contents = set()
        if has_additional_images:
            
            for image in request.images:
                result = uploadFile(image)

                for key, value in result.items():
                    if value > 0.7:
                        additional_contents.add(key)

                prompt = build_followup_with_images_prompt(
                    request.question,
                    additional_contents
                )
        else:
            prompt = request.question

        answer = get_follow_up_answer(prompt)
        return RecipeResponse(recipe=answer, food=additional_contents, chat_history=chat["messages"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/chat-history")
async def get_chat_history():
    """Retrieve current chat history"""
    return {"messages": chat["messages"]}


@app.post("/api/chat-history/clear")
async def clear_chat_history():
    """Clear chat history and start fresh"""
    chat["messages"].clear()
    return {"message": "Chat history cleared"}
    
