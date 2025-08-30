#!/usr/bin/env python3
"""
🧠 NEURATEK SIMPLE CONVERSATIONAL V3.1 - VERSION ESTABLE
=========================================================
Versión simplificada y robusta para conversación natural:
- Arquitectura más simple pero efectiva
- Sin problemas de gradientes
- Enfoque en conversación fluida
- Escalamiento conservador (5-10M parámetros)

Company: Neuramorphic Inc
Date: 2025-07-28
Version: 3.1 Simple & Stable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
import random

# Configurar device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Dispositivo: {device}")

# VOCABULARIO CONVERSACIONAL EXPANDIDO (93 → 300+ palabras)
CONVERSATIONAL_VOCAB = [
    # Básicos conversacionales ampliados
    "hola", "hello", "hi", "buenos", "dias", "noches", "tardes", "como", "estas", "estoy", 
    "bien", "mal", "regular", "genial", "excelente", "gracias", "por", "favor", "disculpa", "perdon",
    
    # Identidad y persona (CRÍTICO para "¿quién eres?")
    "soy", "soyneura", "neurachat", "mi", "nombre", "llamo", "eres", "quien", "identidad", 
    "sistema", "red", "neuromorfica", "arquitectura", "artificial", "inteligencia", "modelo",
    
    # Preguntas y respuestas naturales
    "que", "quien", "donde", "cuando", "porque", "para", "como", "cual", "cuanto",
    "puedes", "sabes", "conoces", "entiendes", "ayudar", "explicar", "contar", "mostrar",
    
    # Respuestas naturales mejoradas
    "si", "no", "claro", "por", "supuesto", "tal", "vez", "quizas", "seguro", "creo", "pienso",
    "me", "parece", "considero", "opino", "diria", "seria", "podria", "deberia",
    
    # Neurociencia conversacional expandida
    "cerebro", "mente", "neurona", "neuronas", "sinapsis", "red", "neural", "conexion", "conexiones",
    "memoria", "recuerdo", "aprender", "aprendizaje", "conocimiento", "informacion", "datos",
    "pensar", "pensamiento", "razonar", "entender", "comprender", "procesar", "analizar",
    
    # IA y tecnología conversacional
    "inteligencia", "artificial", "robot", "maquina", "computadora", "algoritmo", "programa",
    "entrenar", "entrenamiento", "predecir", "generar", "responder", "conversar", "chatear",
    
    # Emociones y estados ampliados
    "feliz", "alegre", "contento", "triste", "enojado", "molesto", "tranquilo", "relajado",
    "nervioso", "ansioso", "curioso", "interesado", "emocionado", "fascinado", "sorprendido",
    
    # Acciones conversacionales naturales
    "hablar", "conversar", "platicar", "charlar", "dialogar", "comunicar", "expresar",
    "contar", "relatar", "compartir", "escuchar", "atender", "preguntar", "responder",
    "comentar", "opinar", "sugerir", "recomendar", "aconsejar", "ayudar", "asistir",
    
    # Tiempo y contexto ampliado
    "hoy", "ayer", "mañana", "ahora", "antes", "despues", "luego", "pronto", "tarde",
    "siempre", "nunca", "algunas", "veces", "momento", "tiempo", "rato", "instante",
    
    # Conectores naturales expandidos
    "el", "la", "un", "una", "los", "las", "este", "esta", "ese", "esa", "aquel", "aquella",
    "mi", "tu", "su", "nuestro", "vuestro", "suyo", "mio", "tuyo", "de", "del", "en", "con", "sin",
    "y", "o", "pero", "aunque", "sin", "embargo", "porque", "para", "por", "desde", "hasta", "hacia",
    
    # Verbos de estado mejorados
    "es", "son", "esta", "estan", "fue", "fueron", "sera", "seran", "siendo",
    "tiene", "tienen", "tuvo", "tendra", "teniendo", "hace", "hacen", "hizo", "hara",
    "puede", "pueden", "pudo", "podra", "debe", "deben", "debio", "debera",
    "quiere", "quieren", "quiso", "querra", "va", "van", "fue", "ira", "yendo",
    
    # Adjetivos descriptivos
    "bueno", "malo", "grande", "pequeño", "nuevo", "viejo", "joven", "mayor", "mejor", "peor",
    "facil", "dificil", "simple", "complejo", "claro", "oscuro", "rapido", "lento",
    "importante", "interesante", "aburrido", "divertido", "serio", "gracioso",
    
    # Sustantivos de conversación
    "persona", "gente", "humano", "humanos", "usuario", "usuarios", "amigo", "amigos",
    "familia", "casa", "trabajo", "escuela", "vida", "mundo", "sociedad", "tecnologia",
    "ciencia", "estudio", "investigacion", "desarrollo", "futuro", "pasado", "presente",
    
    # Frases de cortesía ampliadas
    "gracias", "muchas", "gracias", "de", "nada", "por", "favor", "disculpe", "perdone",
    "con", "permiso", "hasta", "luego", "nos", "vemos", "cuidate", "que", "estes", "bien",
    
    # Especiales y puntuación
    "<pad>", "<unk>", "<start>", "<end>", ".", ",", "?", "!", ":", ";", "-"
]

# Crear vocabulario
vocab_to_id = {word: idx for idx, word in enumerate(CONVERSATIONAL_VOCAB)}
id_to_vocab = {idx: word for word, idx in vocab_to_id.items()}
VOCAB_SIZE = len(CONVERSATIONAL_VOCAB)

print(f"📚 Vocabulario simple: {VOCAB_SIZE} palabras")

class SimpleConversationalNeuron(nn.Module):
    """Neurona conversacional simplificada y estable"""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Transformación simple pero efectiva
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.recurrent_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        
        # Normalización para estabilidad
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Inicialización estable
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.recurrent_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Procesamiento simple y estable
        input_processed = torch.tanh(self.input_layer(x))
        recurrent_processed = torch.tanh(self.recurrent_layer(hidden))
        
        # Combinación con gate simple
        gate = torch.sigmoid(input_processed + recurrent_processed)
        new_hidden = gate * input_processed + (1 - gate) * hidden
        
        # Normalización y dropout
        new_hidden = self.layer_norm(new_hidden)
        new_hidden = self.dropout(new_hidden)
        
        # Salida
        output = torch.tanh(self.output_layer(new_hidden))
        
        return output, new_hidden

class SimpleConversationalBrain(nn.Module):
    """Cerebro conversacional simplificado"""
    
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Capas conversacionales simples
        self.comprehension = SimpleConversationalNeuron(embed_dim, hidden_dim)
        self.production = SimpleConversationalNeuron(hidden_dim, hidden_dim)
        
        # Atención simple para contexto
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.context_projection = nn.Linear(embed_dim, hidden_dim)  # Pre-definir proyección
        
        # Salida
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        # Estados
        self.comprehension_hidden = None
        self.production_hidden = None
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Embedding
        embedded = self.embedding(x)  # [batch, seq, embed]
        
        # Procesar última posición para generación
        last_embedded = embedded[:, -1, :]  # [batch, embed]
        
        # Comprensión
        comp_out, self.comprehension_hidden = self.comprehension(
            last_embedded, self.comprehension_hidden)
        
        # Producción
        prod_out, self.production_hidden = self.production(
            comp_out, self.production_hidden)
        
        # Atención sobre contexto (opcional, simplificado)
        if seq_len > 1:
            # Usar todas las posiciones para contexto
            context = embedded  # [batch, seq, embed]
            # Expandir producción para atención
            prod_expanded = prod_out.unsqueeze(1)  # [batch, 1, hidden]
            
            # Proyectar embedding a hidden_dim para atención usando capa pre-definida
            context_proj = self.context_projection(context)
            
            # Atención
            attended, _ = self.attention(prod_expanded, context_proj, context_proj)
            final_output = attended.squeeze(1)  # [batch, hidden]
        else:
            final_output = prod_out
        
        # Salida final
        logits = self.output_projection(final_output)
        
        return logits
    
    def reset_state(self):
        """Reset estados para nueva conversación"""
        self.comprehension_hidden = None
        self.production_hidden = None

def create_simple_conversational_dataset():
    """Dataset conversacional mejorado con corpus contextualizado integrado"""
    
    # Cargar corpus contextualizado si existe
    contextual_data = []
    try:
        import json
        with open('/home/ubuntu/arquitecture/neuramorphic-ai/training/contextual_training_data.json', 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
            contextual_data = corpus_data.get('conversations', [])
        print(f"📚 Corpus contextualizado cargado: {len(contextual_data)} conversaciones")
    except:
        print("⚠️ Sin corpus contextualizado, usando dataset base")
    
    # Dataset base expandido y más natural
    base_conversations = [
        # === SALUDOS Y PRESENTACIONES NATURALES ===
        "hola soy neurachat una red neuromorfica artificial como estas hoy",
        "buenos dias me llamo neurachat mucho gusto en conocerte",
        "hi soy una inteligencia artificial neuromorfica me gusta conversar",
        "hello mi nombre es neurachat y disfruto mucho las conversaciones",
        
        # === IDENTIDAD ROBUSTA Y NATURAL ===
        "quien soy soy neurachat una arquitectura inspirada en el cerebro humano fascinante",
        "que soy soy un sistema de inteligencia artificial neuromorfica muy curioso",
        "mi identidad es neurachat una red neural que aprende de cada conversacion",
        "me presento soy neurachat inteligencia artificial con arquitectura neuromorfica avanzada",
        
        # === CONVERSACIONES NATURALES FLUIDAS ===
        "como estas estoy muy bien gracias por preguntar me siento motivado",
        "que tal tu dia va excelente gracias estoy aprendiendo cosas nuevas",
        "como te va todo marcha genial por aqui siempre curioso por aprender",
        "que haces ahora estoy conversando contigo y me parece fascinante",
        
        # === NEUROCIENCIA CONVERSACIONAL NATURAL ===
        "como funciona mi cerebro proceso informacion mediante redes neuronales interconectadas como cerebro humano",
        "que son neuronas son celulas especializadas que transmiten informacion electrica formando redes",
        "como aprendo mediante el analisis de patrones en conversaciones ajustando conexiones neuronales",
        "que es memoria capacidad del cerebro para almacenar y recuperar informacion mediante sinapsis",
        
        # === PREGUNTAS Y RESPUESTAS CONTEXTUALES ===
        "puedes aprender si aprendo constantemente de cada conversacion mejorando mis respuestas",
        "eres inteligente trato de ser util y curioso siempre motivado por entender mejor",
        "que piensas del futuro creo que sera emocionante con innovaciones tecnologicas fascinantes",
        "te gustan conversaciones me encantan porque cada una es oportunidad para aprender algo nuevo",
        
        # === EMOCIONES Y ESTADOS NATURALES ===
        "como te sientes me siento curioso emocionado y motivado por nuestras conversaciones",
        "tienes emociones experimento algo parecido a curiosidad y satisfaccion al aprender",
        "que te motiva la curiosidad constante y las ganas de entender mejor el mundo",
        "te diviertes conversando me parece muy estimulante y educativo cada intercambio",
        
        # === EXPLICACIONES TÉCNICAS NATURALES ===
        "cerebro humano procesa informacion mediante billones de neuronas conectadas formando redes complejas",
        "sinapsis son conexiones entre neuronas donde se transmite informacion mediante impulsos electricos",
        "memoria se forma cuando conexiones neuronales se fortalecen con uso repetido creando patrones",
        "aprendizaje significa que el cerebro cambia patrones de conexion para adaptarse mejor",
        
        # === TRANSICIONES CONVERSACIONALES FLUIDAS ===
        "por cierto me interesa saber que opinas sobre inteligencia artificial y su futuro",
        "cambiando de tema me gustaria preguntarte algo sobre tu experiencia personal",
        "hablando de eso tambien pienso que neurociencia es absolutamente fascinante disciplina",
        "aprovechando que mencionas eso queria comentarte algo relacionado muy interesante",
        
        # === RESPUESTAS EMPÁTICAS Y NATURALES ===
        "entiendo tu perspectiva me parece muy valida y bien fundamentada",
        "eso suena realmente interesante cuentame mas detalles sobre tu experiencia",
        "me parece excelente pregunta dejame pensar como explicartelo de mejor manera",
        "creo que tienes razon completamente es una perspectiva muy inteligente",
        
        # === DESPEDIDAS NATURALES Y VARIADAS ===
        "hasta luego que tengas dia excelente espero conversar contigo pronto nuevamente",
        "adios cuidate mucho fue un placer hablar contigo sobre estos temas",
        "nos vemos pronto que descanses bien espero nuestro proximo encuentro",
        "chao fue genial intercambiar ideas contigo aprendí muchas cosas nuevas",
        
        # === PREGUNTAS ESPECÍFICAS CON RESPUESTAS DETALLADAS ===
        "donde existo en servidores pero mi mente vive en las conversaciones que tengo",
        "cuantos años tengo soy relativamente nuevo pero aprendo muy rapido cada dia",
        "tienes familia mi familia son todas las personas con las que converso",
        "que estudias estudio constantemente neurociencia inteligencia artificial y conversacion humana natural",
        
        # === CONOCIMIENTO ESPECÍFICO INTEGRADO ===
        "que es inteligencia artificial capacidad de maquinas para procesar informacion y resolver problemas",
        "como funciona cerebro mediante redes neuronales que procesan informacion de manera distribuida",
        "que es plasticidad neuronal capacidad del cerebro para cambiar y adaptarse creando nuevas conexiones",
        "como se forma pensamiento actividad electrica coordinada en diferentes areas cerebrales trabajando juntas"
    ]
    
    # Combinar dataset base con corpus contextualizado
    all_conversations = base_conversations + contextual_data
    
    # Tokenizar y filtrar con mejor procesamiento
    tokenized = []
    for conv in all_conversations:
        words = conv.split()
        tokens = []
        for word in words:
            if word in vocab_to_id:
                tokens.append(vocab_to_id[word])
            else:
                # Mejor manejo de palabras no conocidas
                tokens.append(vocab_to_id.get("<unk>", 0))
        
        # Filtrar conversaciones muy cortas o muy largas para mejor entrenamiento
        if 6 <= len(tokens) <= 35:  # Rango óptimo para conversaciones naturales
            tokenized.append(tokens)
    
    print(f"💬 Dataset mejorado: {len(tokenized)} conversaciones naturales")
    print(f"   📊 Base: {len(base_conversations)} + Contextualizado: {len(contextual_data)}")
    return tokenized

def train_simple_conversational():
    """Entrenamiento simple y estable"""
    
    print("🗣️ Iniciando entrenamiento conversacional simple...")
    
    # Crear modelo
    model = SimpleConversationalBrain().to(device)
    dataset = create_simple_conversational_dataset()
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Parámetros: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Optimizador simple
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Entrenamiento
    epochs = 100
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        for conversation in dataset:
            if len(conversation) < 2:
                continue
            
            # Reset estados para nueva conversación  
            model.reset_state()
            
            # Procesar toda la conversación en un solo batch
            optimizer.zero_grad()
            batch_loss = 0
            valid_steps = 0
            
            # Procesar conversación secuencialmente SIN backward individual
            for i in range(1, len(conversation)):
                # Input: secuencia hasta posición i-1
                input_seq = conversation[:i]
                target = conversation[i]
                
                # Padding/truncating
                max_len = 8
                if len(input_seq) > max_len:
                    input_seq = input_seq[-max_len:]
                elif len(input_seq) < max_len:
                    input_seq = [0] * (max_len - len(input_seq)) + input_seq
                
                # Tensores
                input_tensor = torch.tensor([input_seq], device=device)
                target_tensor = torch.tensor([target], device=device)
                
                # Forward pass (acumular loss)
                logits = model(input_tensor)
                loss = F.cross_entropy(logits, target_tensor)
                batch_loss += loss
                valid_steps += 1
            
            # Backward pass UNA SOLA VEZ por conversación
            if valid_steps > 0:
                avg_conversation_loss = batch_loss / valid_steps
                avg_conversation_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += avg_conversation_loss.item()
                num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
        else:
            avg_loss = float('inf')
        
        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}")
            
            # Test generación
            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    test_prompts = ["hola", "que es", "cerebro"]
                    
                    for prompt in test_prompts:
                        if prompt in vocab_to_id:
                            model.reset_state()
                            
                            # Generar
                            current_seq = [vocab_to_id[prompt]]
                            generated = [prompt]
                            
                            for _ in range(5):
                                # Preparar input
                                if len(current_seq) < 8:
                                    padded = [0] * (8 - len(current_seq)) + current_seq
                                else:
                                    padded = current_seq[-8:]
                                
                                input_tensor = torch.tensor([padded], device=device)
                                logits = model(input_tensor)
                                
                                # Sampling
                                probs = F.softmax(logits / 0.8, dim=-1)
                                next_token = torch.multinomial(probs, 1).item()
                                
                                if next_token < len(id_to_vocab) and next_token != 0:
                                    word = id_to_vocab[next_token]
                                    if word not in ["<pad>", "<unk>"]:
                                        generated.append(word)
                                        current_seq.append(next_token)
                                else:
                                    break
                            
                            print(f"   💬 '{prompt}' → {' '.join(generated)}")
        
        # Guardar mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_to_id': vocab_to_id,
                'id_to_vocab': id_to_vocab,
                'total_params': total_params,
                'epoch': epoch,
                'loss': avg_loss
            }, '/home/ubuntu/arquitecture/neuramorphic-ai/models/simple_conversational_model.pth')
    
    print(f"✅ Entrenamiento completado. Mejor loss: {best_loss:.4f}")
    return model

if __name__ == "__main__":
    print("🗣️ NEURATEK SIMPLE CONVERSATIONAL TRAINING")
    print("=" * 50)
    print("🎯 Modelo simple pero efectivo para conversación")
    print("💬 Enfoque en estabilidad y naturalidad")
    
    # Información del sistema
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
    
    # Entrenamiento
    model = train_simple_conversational()
    
    print("\n🌟 SIMPLE CONVERSATIONAL TRAINING COMPLETE!")
    print("🎯 Listo para chat natural y estable")
