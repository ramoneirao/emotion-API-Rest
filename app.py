from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertModel, AutoTokenizer
import re
from flask_cors import CORS
from joblib import load as joblib_load
import logging

app = Flask(__name__)
CORS(app)  

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            logger.info(f"Modelo carregado com sucesso! Época: {checkpoint.get('epoch', 'N/A')}")
        else:
            # Se for apenas o state_dict
            model.load_state_dict(checkpoint)
            logger.info("State dict carregado com sucesso!")
            
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return None

# Inicializar tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Labels das emoções (ajuste conforme seu dataset)
EMOTION_LABELS = [
    'Alegria',     # 0
    'Tristeza',    # 1
    'Raiva',       # 2
    'Medo',        # 3
    'Surpresa',    # 4
    'Nojo',        # 5
    'Confiança',   # 6
    'Antecipação', # 7
]

# Labels para SVM 
SVM_LABELS = ['alegria', 'tristeza', 'raiva', 'medo', 'nojo', 'surpresa', 'confianca', 'antecipacao']

# Carregar modelo BERT global
logger.info("Carregando modelo BERT...")
best_model_path = 'models/weights.pt'  
model = BERTClass()
model = load_best_model(model, best_model_path)

if model is not None:
    model.to(device)
    model.eval()
    logger.info(f"Modelo BERT carregado e movido para: {device}")
else:
    logger.error("ERRO: Não foi possível carregar o modelo BERT!")

# Carregar modelo SVM e vetorizador
svm_compatible = False
try:
    modelo_svm = joblib_load('models/svm.pkl')
    vetorizador = joblib_load('vec/vetorizador.pkl')
    
    # Testar compatibilidade na inicialização
    try:
        test_text = ["texto de teste"]
        test_vector = vetorizador.transform(test_text)
        n_features_vetorizador = test_vector.shape[1]
        
        # Verificar quantas features o modelo SVM espera
        if hasattr(modelo_svm, 'coef_'):
            n_features_modelo = modelo_svm.coef_.shape[1]
        elif hasattr(modelo_svm, 'n_features_in_'):
            n_features_modelo = modelo_svm.n_features_in_
        else:
            n_features_modelo = None
            
        if n_features_modelo is not None:
            if n_features_vetorizador == n_features_modelo:
                svm_compatible = True
                logger.info(f"SVM e vetorizador compatíveis: {n_features_modelo} features")
            else:
                logger.error(f"INCOMPATIBILIDADE: Vetorizador produz {n_features_vetorizador} features, mas SVM espera {n_features_modelo}")
        else:
            logger.warning("Não foi possível determinar o número de features esperado pelo modelo SVM")
            
    except Exception as e:
        logger.error(f"Erro ao testar compatibilidade SVM: {e}")
        
    logger.info("Modelo SVM e vetorizador carregados com sucesso!")
    
except Exception as e:
    modelo_svm = None
    vetorizador = None
    logger.error(f"Erro ao carregar modelo SVM ou vetorizador: {e}")

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
        "modelo_bert_carregado": model is not None,
        "modelo_svm_carregado": modelo_svm is not None and vetorizador is not None,
        "svm_compativel": svm_compatible,
        "device": str(device),
        "emocoes_disponiveis": EMOTION_LABELS
    })

@app.route('/svm', methods=['POST'])
def prever_emocao_svm():
    """Endpoint para predição de emoções usando SVM"""
    try:
        # Verificar se o modelo SVM e vetorizador foram carregados
        if modelo_svm is None or vetorizador is None:
            return jsonify({
                "erro": "Modelo SVM ou vetorizador não carregado",
                "solucao": "Verifique se os arquivos 'models/svm.pkl' e 'vec/vetorizador.pkl' existem"
            }), 500

        # Verificar compatibilidade
        if not svm_compatible:
            return jsonify({
                "erro": "Modelo SVM e vetorizador incompatíveis",
                "detalhes": "O vetorizador e o modelo SVM foram treinados separadamente e têm dimensões diferentes",
                "solucao": "Você precisa treinar e salvar o vetorizador e o modelo SVM juntos no mesmo experimento. Não é possível corrigir isso apenas alterando o código - é necessário retreinar os modelos."
            }), 500

        data = request.get_json()
        frase = data.get('frase') if data else None

        if not frase or len(frase.strip()) == 0:
            return jsonify({"erro": "Por favor, forneça uma frase."}), 400

        # Pré-processar a frase
        frase_processada = preprocess_text(frase)
        logger.info(f"Texto processado: '{frase_processada}'")
        
        # Transformar a frase usando o vetorizador treinado
        frase_vetorizada = vetorizador.transform([frase_processada])
        logger.info(f"Shape do vetor: {frase_vetorizada.shape}")

        # Fazer predição
        # Verificar se o modelo tem predict_proba (para probabilidades) ou apenas predict
        if hasattr(modelo_svm, 'predict_proba'):
            # Se tiver predict_proba, usar para obter probabilidades
            probabilidades = modelo_svm.predict_proba(frase_vetorizada)[0]
            indice_emocao = np.argmax(probabilidades)
            confianca = probabilidades[indice_emocao]
            
            # Criar dicionário com todas as probabilidades
            todas_probabilidades = {
                SVM_LABELS[i]: float(probabilidades[i]) 
                for i in range(min(len(SVM_LABELS), len(probabilidades)))
            }
            
        else:
            # Se não tiver, usar predict normal
            predicao = modelo_svm.predict(frase_vetorizada)[0]
            logger.info(f"Predição SVM: {predicao}")
            
            # Se a predição for um índice numérico
            if isinstance(predicao, (int, np.integer)):
                indice_emocao = predicao
                confianca = 1.0  # Não temos informação de confiança
            else:
                # Se a predição for uma string/label, encontrar o índice
                try:
                    indice_emocao = SVM_LABELS.index(predicao)
                    confianca = 1.0
                except ValueError:
                    return jsonify({"erro": f"Label '{predicao}' não encontrado nos labels conhecidos: {SVM_LABELS}"}), 500
            
            todas_probabilidades = None

        # Garantir que o índice está dentro do range válido
        if indice_emocao >= len(SVM_LABELS):
            return jsonify({"erro": f"Índice de emoção inválido: {indice_emocao}. Labels disponíveis: {SVM_LABELS}"}), 500

        emocao_principal = SVM_LABELS[indice_emocao]

        resultado = {
            "emocao": emocao_principal,
            "confianca": float(confianca) if hasattr(modelo_svm, 'predict_proba') else None,
            "texto_original": frase,
            "texto_processado": frase_processada
        }

        # Se temos probabilidades, incluir todas
        if todas_probabilidades:
            resultado["todas_emocoes"] = todas_probabilidades
            # Ranking das top 3 emoções
            ranking = sorted(todas_probabilidades.items(), key=lambda x: x[1], reverse=True)
            resultado["ranking_emocoes"] = ranking[:3]

        return jsonify(resultado)

    except Exception as e:
        logger.error(f"Erro na predição SVM: {str(e)}")
        return jsonify({
            "erro": "Erro interno do servidor",
            "detalhes": str(e),
            "tipo_erro": type(e).__name__
        }), 500

@app.route('/bert', methods=['POST'])
def predict_emotion():
    """Endpoint principal para predição de emoções usando BERT"""
    try:
        # Verificar se o modelo foi carregado
        if model is None:
            return jsonify({"erro": "Modelo BERT não carregado"}), 500
        
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
        logger.error(f"Erro na predição BERT: {str(e)}")
        return jsonify({"erro": f"Erro na predição: {str(e)}"}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("INICIANDO API DE CLASSIFICAÇÃO DE EMOÇÕES")
    print("="*50)
    print(f"Device: {device}")
    print(f"Modelo BERT carregado: {'✅' if model is not None else '❌'}")
    print(f"Modelo SVM carregado: {'✅' if modelo_svm is not None else '❌'}")
    print(f"SVM compatível: {'✅' if svm_compatible else '❌'}")
    print(f"Emoções BERT: {len(EMOTION_LABELS)}")
    print(f"Emoções SVM: {len(SVM_LABELS)}")
    print("="*50)
    
    if not svm_compatible and modelo_svm is not None:
        print("⚠️  ATENÇÃO: Modelo SVM e vetorizador são incompatíveis!")
        print("   Para corrigir: retreine e salve os modelos juntos.")
    
    app.run(host='0.0.0.0', port=3030, debug=True)
