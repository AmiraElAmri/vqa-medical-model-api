import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import BertTokenizer
import json

# Charger label_map.json
with open("label_map.json", "r") as f:
    label_map_raw = json.load(f)

# Convertir les clés en entiers uniquement si elles sont numériques
label_index = {}
for k, v in label_map_raw.items():
    if k.isdigit():
        label_index[int(k)] = v
    else:
        print(f"⚠️ Clé ignorée (non numérique) : {k}")

# Charger le modèle IA médical
class VQAModel(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # MobileNetV2 sans couche finale
        self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
        self.mobilenet.classifier[1] = torch.nn.Identity()

        # BioBERT sans mise à jour de poids
        from transformers import BertModel
        self.bert = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
        self.bert.requires_grad_(False)

        # MLP final
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280 + 768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, len(label_index))
        )

    def forward(self, image, tokens):
        image_features = self.mobilenet(image)
        text_features = self.bert(tokens['input_ids'], attention_mask=tokens['attention_mask']).pooler_output
        combined = torch.cat((image_features, text_features), dim=1)
        return self.classifier(combined)

# Charger le modèle entraîné
model_path = "saved_models/best_vqa_model.pth"
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
model = VQAModel(num_labels=len(label_index)).eval()
model.load_state_dict(checkpoint["model_state_dict"])

# Charger le tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prétraitement des images
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Fonction de prédiction
def predict(image, question):
    if image is None or not question.strip():
        return "❌ Image ou question manquante"

    # Prétraiter l'image
    image_tensor = transform(image).unsqueeze(0).to(torch.float32)

    # Encoder la question
    tokens = tokenizer(question, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

    # Prédiction
    with torch.no_grad():
        output = model(image_tensor, tokens)
        _, pred = torch.max(output, 1)

    return label_index.get(pred.item(), "❓ Inconnu")

# Interface Gradio
iface = gr.Interface(fn=predict,
                     inputs=["image", "text"],
                     outputs="text",
                     title="🧠 Modèle Multimodal Médical",
                     description="Analyse d'images endoscopiques + Questions médicales")

if __name__ == "__main__":
    iface.launch()