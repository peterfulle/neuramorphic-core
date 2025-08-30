#!/usr/bin/env python3
"""
🧠 NEURATEK CONVERSATIONAL V3.1 - ESCALAMIENTO CONSERVADOR
===========================================================
Escalamiento inteligente enfocado en conversación natural:
- 10-20M parámetros (vs 100M)
- Datasets conversacionales específicos
- Mejora del chat básico
- Respuestas más naturales y fluidas

Company: Neuramorphic Inc
Date: 2025-07-28
Version: 3.1 Conversational
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
import random
import os

# Configurar device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Dispositivo: {device}")

# VOCABULARIO CONVERSACIONAL EXPANDIDO PERO ENFOCADO
CONVERSATIONAL_VOCAB = [
    # Básicos conversacionales
    "hola", "hello", "hi", "buenos", "dias", "noches", "tardes",
    "como", "estas", "estoy", "bien", "mal", "regular", "genial",
    "gracias", "por", "favor", "disculpa", "perdón", "claro",
    
    # Preguntas comunes
    "que", "quien", "donde", "cuando", "porque", "para", "como",
    "puedes", "sabes", "conoces", "entiendes", "ayudar", "explicar",
    
    # Respuestas naturales
    "si", "no", "tal", "vez", "quizas", "seguro", "creo", "pienso",
    "me", "parece", "considero", "opino", "diria", "seria",
    
    # Neurociencia conversacional
    "cerebro", "mente", "neurona", "sinapsis", "memoria", "aprender",
    "inteligencia", "artificial", "pensar", "razonar", "entender",
    "procesar", "información", "conocimiento", "experiencia",
    
    # IA y tecnología accesible
    "robot", "computadora", "algoritmo", "datos", "sistema", "red",
    "modelo", "entrenar", "predecir", "analizar", "resolver",
    
    # Emociones y estados
    "feliz", "triste", "enojado", "tranquilo", "nervioso", "curioso",
    "interesante", "aburrido", "emocionante", "preocupado",
    
    # Acciones conversacionales
    "hablar", "conversar", "platicar", "contar", "escuchar", "preguntar",
    "responder", "comentar", "opinar", "sugerir", "recomendar",
    
    # Tiempo y contexto
    "hoy", "ayer", "mañana", "ahora", "antes", "después", "siempre",
    "nunca", "algunas", "veces", "momento", "tiempo", "rato",
    
    # Conectores naturales
    "el", "la", "un", "una", "los", "las", "este", "esta", "ese", "esa",
    "mi", "tu", "su", "nuestro", "su", "de", "del", "en", "con", "sin",
    "y", "o", "pero", "aunque", "porque", "para", "por", "desde", "hasta",
    
    # Palabras funcionales mejoradas
    "es", "son", "esta", "están", "tiene", "tienen", "hace", "hacen",
    "puede", "pueden", "debe", "deben", "quiere", "quieren", "va", "van",
    
    # Especiales
    "<pad>", "<unk>", "<start>", "<end>", ".", ",", "?", "!", ":"
]

# Crear mapeo optimizado
vocab_to_id = {word: idx for idx, word in enumerate(CONVERSATIONAL_VOCAB)}
id_to_vocab = {idx: word for word, idx in vocab_to_id.items()}
VOCAB_SIZE = len(CONVERSATIONAL_VOCAB)

print(f"📚 Vocabulario conversacional: {VOCAB_SIZE} palabras")

class ConversationalBiologicalNeuron(nn.Module):
    """Neurona optimizada para conversación natural"""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Parámetros optimizados para conversación
        self.tau_mem = nn.Parameter(torch.tensor(8.0))   # Más rápido para conversación
        self.tau_syn = nn.Parameter(torch.tensor(4.0))   # Respuesta más ágil
        self.v_threshold = nn.Parameter(torch.tensor(-55.0))  # Más sensible
        self.v_reset = nn.Parameter(torch.tensor(-70.0))
        self.v_rest = nn.Parameter(torch.tensor(-65.0))
        
        # Corrientes para fluidez conversacional
        self.baseline_current = nn.Parameter(torch.tensor(6.0))  # Activación constante
        self.noise_strength = nn.Parameter(torch.tensor(1.5))   # Menos ruido, más estabilidad
        
        # Capas más eficientes
        self.input_transform = nn.Linear(input_size, hidden_size)
        self.recurrent_weights = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Inicialización conversacional
        nn.init.xavier_normal_(self.input_transform.weight, gain=0.8)
        nn.init.orthogonal_(self.recurrent_weights.weight, gain=0.6)
        
    def forward(self, x, hidden_state=None):
        batch_size = x.size(0)
        
        if hidden_state is None:
            membrane_potential = self.v_rest.expand(batch_size, self.hidden_size)
            synaptic_current = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            membrane_potential, synaptic_current = hidden_state
            
        # Procesamiento conversacional
        input_current = self.input_transform(x)
        recurrent_current = self.recurrent_weights(synaptic_current)
        
        # Ruido controlado para naturalidad
        noise = torch.randn_like(membrane_potential) * self.noise_strength
        
        total_current = input_current + recurrent_current + self.baseline_current + noise
        
        # Dinámica LIF conversacional
        dt = 1.0
        membrane_potential = (membrane_potential + 
                            dt * (-membrane_potential + self.v_rest + total_current) / self.tau_mem)
        
        # Spikes más frecuentes para fluidez
        spikes = (membrane_potential > self.v_threshold).float()
        
        # Reset suave
        membrane_potential = torch.where(
            spikes.bool(),
            self.v_reset.expand_as(membrane_potential),
            membrane_potential
        )
        
        # Activación conversacional mejorada
        synaptic_current = (synaptic_current + 
                          dt * (-synaptic_current + spikes) / self.tau_syn)
        
        # Salida más expresiva
        output = torch.tanh(synaptic_current * 1.5)
        
        return output, (membrane_potential, synaptic_current)

class ConversationalNeuramorphicBrain(nn.Module):
    """Cerebro neuramórfico para conversación natural"""
    
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=192, hidden_dim=384):  # Tamaños conservadores
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding conversacional
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Áreas especializadas para conversación
        self.language_comprehension = ConversationalBiologicalNeuron(embed_dim, hidden_dim)  # Wernicke-like
        self.language_production = ConversationalBiologicalNeuron(hidden_dim, hidden_dim)    # Broca-like
        self.conversational_context = ConversationalBiologicalNeuron(hidden_dim, hidden_dim) # Context memory
        
        # Memoria conversacional a corto plazo
        self.short_term_memory = nn.LSTM(embed_dim, hidden_dim//2, batch_first=True)  # Corregido: embed_dim como input
        
        # Fusión conversacional optimizada
        self.conversation_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim//2, hidden_dim),
            nn.GELU(),  # Más suave que ReLU
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2)
        )
        
        # Cabeza de salida conversacional
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//4, vocab_size)
        )
        
        # Estados conversacionales
        self.comprehension_hidden = None
        self.production_hidden = None
        self.context_hidden = None
        self.lstm_hidden = None
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # Secuencia para memoria LSTM
        seq_length = embedded.size(1)
        lstm_input = embedded.view(-1, seq_length, self.embed_dim)
        
        # Procesar última posición para generación
        last_embedded = embedded[:, -1, :]  # Último token
        
        # Comprensión del lenguaje
        comprehension_out, self.comprehension_hidden = self.language_comprehension(
            last_embedded, self.comprehension_hidden)
        
        # Producción del lenguaje
        production_out, self.production_hidden = self.language_production(
            comprehension_out, self.production_hidden)
        
        # Contexto conversacional
        context_out, self.context_hidden = self.conversational_context(
            production_out, self.context_hidden)
        
        # Memoria a corto plazo
        memory_out, self.lstm_hidden = self.short_term_memory(lstm_input, self.lstm_hidden)
        last_memory = memory_out[:, -1, :]  # Último estado de memoria
        
        # Fusión conversacional
        combined = torch.cat([comprehension_out, production_out, context_out, last_memory], dim=-1)
        fused = self.conversation_fusion(combined)
        
        # Salida conversacional
        logits = self.output_head(fused)
        
        return logits
    
    def reset_conversation_state(self):
        """Reset para nueva conversación"""
        self.comprehension_hidden = None
        self.production_hidden = None
        self.context_hidden = None
        self.lstm_hidden = None

def create_conversational_dataset():
    """Dataset específico para conversación natural"""
    
    # Conversaciones más naturales y variadas
    conversations = [
        # Saludos y presentaciones
        "hola como estas hoy",
        "buenos dias como te sientes",
        "hi me llamo juan y tu",
        "hello que tal tu dia",
        
        # Preguntas sobre IA y neurociencia
        "que es inteligencia artificial",
        "como funciona el cerebro humano",
        "puedes explicar que son las neuronas",
        "que diferencia hay entre mente y cerebro",
        "como aprende una red neuronal",
        
        # Conversación casual
        "me gusta hablar contigo",
        "eres muy interesante de verdad",
        "que opinas sobre la tecnologia",
        "como procesas la informacion",
        "que te parece mas fascinante",
        
        # Respuestas empáticas
        "entiendo tu punto de vista",
        "eso suena realmente interesante",
        "me parece una gran pregunta",
        "creo que tienes razon en eso",
        "seria genial explorar esa idea",
        
        # Explicaciones simples
        "el cerebro tiene muchas neuronas conectadas",
        "las sinapsis transmiten informacion entre neuronas",
        "la memoria se forma por conexiones fuertes",
        "aprender significa cambiar esas conexiones",
        "pensar es actividad electrica en el cerebro",
        
        # Conversación sobre emociones
        "me siento curioso por aprender mas",
        "es emocionante descubrir cosas nuevas",
        "a veces me pregunto como funcionamos",
        "la curiosidad es muy importante para mi",
        
        # Transiciones naturales
        "por cierto que te interesa",
        "cambiando de tema me gustaria saber",
        "hablando de eso tambien pienso que",
        "eso me recuerda a otra cosa",
        
        # Despedidas naturales
        "fue genial hablar contigo hoy",
        "espero conversar pronto de nuevo",
        "que tengas un excelente dia",
        "hasta la proxima conversacion"
    ]
    
    # Tokenizar conversaciones
    tokenized_conversations = []
    for conv in conversations:
        words = conv.split()
        # Filtrar y tokenizar
        filtered_words = [word for word in words if word in vocab_to_id]
        if len(filtered_words) >= 4:  # Mínimo 4 palabras para conversación
            tokens = [vocab_to_id[word] for word in filtered_words]
            tokenized_conversations.append(tokens)
    
    print(f"💬 Dataset conversacional: {len(tokenized_conversations)} conversaciones")
    return tokenized_conversations

def conversational_training():
    """Entrenamiento específico para conversación"""
    
    print("🗣️ Iniciando entrenamiento conversacional...")
    
    # Modelo conversacional
    model = ConversationalNeuramorphicBrain().to(device)
    dataset = create_conversational_dataset()
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Parámetros del modelo conversacional: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Optimizador para conversación
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)
    
    # Entrenamiento conversacional
    epochs = 150
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Shuffle para variedad
        random.shuffle(dataset)
        
        for conversation in dataset:
            if len(conversation) < 2:
                continue
                
            # Reset para nueva conversación
            model.reset_conversation_state()
            
            # Crear secuencias de longitud variable (más natural)
            max_len = min(len(conversation), 16)  # Conversaciones más cortas
            
            for i in range(1, max_len):
                # Input: contexto previo
                input_seq = conversation[:i]
                target_token = conversation[i]
                
                # Padding si es necesario
                if len(input_seq) < 8:
                    padded_seq = input_seq + [0] * (8 - len(input_seq))
                else:
                    padded_seq = input_seq[-8:]  # Últimos 8 tokens como contexto
                
                input_tensor = torch.tensor([padded_seq], device=device)
                target_tensor = torch.tensor([target_token], device=device)
                
                # Forward pass
                logits = model(input_tensor)
                loss = F.cross_entropy(logits, target_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Menos agresivo
                optimizer.step()
                
                total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / sum(len(conv) for conv in dataset)
        
        # Logging conversacional
        if epoch % 15 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
            
            # Test conversacional cada 15 epochs
            model.eval()
            with torch.no_grad():
                test_prompts = ["hola como", "que es", "me gusta"]
                
                for prompt in test_prompts:
                    words = prompt.split()
                    if all(word in vocab_to_id for word in words):
                        model.reset_conversation_state()
                        
                        # Generar respuesta conversacional
                        current_seq = [vocab_to_id[word] for word in words]
                        generated = words.copy()
                        
                        for _ in range(6):  # Respuestas más cortas
                            if len(current_seq) < 8:
                                padded = current_seq + [0] * (8 - len(current_seq))
                            else:
                                padded = current_seq[-8:]
                            
                            input_tensor = torch.tensor([padded], device=device)
                            logits = model(input_tensor)
                            
                            # Sampling con temperatura conversacional
                            probs = F.softmax(logits / 0.7, dim=-1)  # Más conservador
                            next_token = torch.multinomial(probs, 1).item()
                            
                            if next_token < len(id_to_vocab) and next_token != 0:
                                word = id_to_vocab[next_token]
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
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'vocab_to_id': vocab_to_id,
                'id_to_vocab': id_to_vocab,
                'total_params': total_params
            }, '/home/ubuntu/arquitecture/neuramorphic-ai/models/conversational_model.pth')
    
    print(f"✅ Entrenamiento conversacional completado. Mejor loss: {best_loss:.4f}")
    print(f"🎯 Modelo: {total_params/1e6:.1f}M parámetros - Optimizado para conversación")
    return model

if __name__ == "__main__":
    print("🗣️ NEURATEK CONVERSATIONAL TRAINING")
    print("=" * 50)
    print("🎯 Escalamiento conservador: 10-20M parámetros")
    print("💬 Enfoque: Conversación natural y fluida")
    print("🧠 Arquitectura: Neuramórfica optimizada")
    
    # Información del sistema
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Entrenamiento conversacional
    model = conversational_training()
    
    print("\n🌟 CONVERSATIONAL NEUROMORPHIC TRAINING COMPLETE!")
    print("🎯 Listo para chat natural y fluido")
