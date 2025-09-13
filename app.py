from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertModel, AutoTokenizer
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

# Configurações
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 200

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 8)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

# Função para carregar o melhor modelo salvo
def load_best_model(model, best_model_path):
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # Verificar se é um checkpoint completo ou apenas state_dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Modelo carregado com sucesso! Época: {checkpoint.get('epoch', 'N/A')}")
        else:
            # Se for apenas o state_dict
            model.load_state_dict(checkpoint)
            print("State dict carregado com sucesso!")
            
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None

# Inicializar tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Labels das emoções (ajuste conforme seu dataset)
EMOTION_LABELS = [
    'Alegria',      # 0
    'Tristeza',     # 1
    'Raiva',        # 2
    'Medo',         # 3
    'Surpresa',     # 4
    'Nojo',         # 5
    'Confiança',    # 6
    'Antecipação',  # 7
]

# Carregar modelo global
print("Carregando modelo...")
best_model_path = 'weights.pt'  # ou "modelo_emocoes.pt"
model = BERTClass()
model = load_best_model(model, best_model_path)

if model is not None:
    model.to(device)
    model.eval()
    print(f"Modelo carregado e movido para: {device}")
else:
    print("ERRO: Não foi possível carregar o modelo!")

# Função de pré-processamento
def preprocess_text(text):
    """Limpa e prepara o texto para o modelo"""
    if not isinstance(text, str):
        return ""
    
    # Remover caracteres especiais e normalizar
    text = re.sub(r'http\S+', '', text)  # URLs
    text = re.sub(r'@\w+', '', text)     # Mentions
    text = re.sub(r'#\w+', '', text)     # Hashtags
    text = re.sub(r'\d+', '', text)      # Números
    text = re.sub(r'[^\w\s]', '', text)  # Pontuação
    text = text.lower().strip()
    
    return text

def tokenize_text(text):
    """Tokeniza o texto para o modelo BERT"""
    text = preprocess_text(text)
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    return {
        'input_ids': encoding['input_ids'].to(device),
        'attention_mask': encoding['attention_mask'].to(device),
        'token_type_ids': encoding['token_type_ids'].to(device)
    }

@app.route('/')
def home():
    """Endpoint de teste"""
    return jsonify({
        "status": "API de Classificação de Emoções",
        "modelo_carregado": model is not None,
        "device": str(device),
        "emocoes_disponiveis": EMOTION_LABELS
    })

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Endpoint principal para predição de emoções"""
    try:
        # Verificar se o modelo foi carregado
        if model is None:
            return jsonify({"erro": "Modelo não carregado"}), 500
        
        # Obter dados da requisição
        data = request.get_json()
        
        if not data or 'texto' not in data:
            return jsonify({"erro": "Campo 'texto' é obrigatório"}), 400
        
        texto = data['texto']
        
        if not texto or len(texto.strip()) == 0:
            return jsonify({"erro": "Texto não pode estar vazio"}), 400
        
        # Tokenizar o texto
        inputs = tokenize_text(texto)
        
        # Fazer predição
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attn_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids']
            )
            
            # Aplicar softmax para obter probabilidades
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Obter todas as probabilidades
            all_probs = probabilities[0].cpu().numpy()
            
            # Criar resposta detalhada
            emocoes_probabilidades = {
                EMOTION_LABELS[i]: float(all_probs[i]) 
                for i in range(len(EMOTION_LABELS))
            }
            
            # Ordenar por probabilidade
            emocoes_ordenadas = sorted(
                emocoes_probabilidades.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        
        return jsonify({
            "texto_original": texto,
            "emocao_principal": EMOTION_LABELS[predicted_class],
            "confianca": round(confidence, 4),
            "todas_emocoes": emocoes_probabilidades,
            "ranking_emocoes": emocoes_ordenadas[:3],  # Top 3
            "texto_processado": preprocess_text(texto)
        })
        
    except Exception as e:
        return jsonify({"erro": f"Erro na predição: {str(e)}"}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("INICIANDO API DE CLASSIFICAÇÃO DE EMOÇÕES")
    print("="*50)
    print(f"Device: {device}")
    print(f"Modelo carregado: {'✅' if model is not None else '❌'}")
    print(f"Emoções: {len(EMOTION_LABELS)}")
    print("="*50)
    
    app.run(host='0.0.0.0', port=3030, debug=True)