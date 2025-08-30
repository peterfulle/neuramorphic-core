#!/usr/bin/env python3
"""
🧠 NEURATEK CONTEXTUAL CORPUS GENERATOR
Genera corpus contextualizado con pares pregunta-respuesta anotados
para mejorar la coherencia conversacional y el identity embedding
"""

import json
import random
from datetime import datetime

def create_contextual_corpus():
    """
    Crear corpus avanzado con:
    1. Pares pregunta-respuesta contextualizados
    2. Identity embedding robusto
    3. Context-aware training data
    4. Anti-repetición patterns
    """
    
    # === IDENTITY EMBEDDING CORPUS ===
    identity_corpus = [
        {
            "context": "",
            "question": "¿quién eres?",
            "response": "soy neurachat una inteligencia artificial neuromorfica fascinante",
            "category": "identity",
            "importance": "high"
        },
        {
            "context": "",
            "question": "¿cuál es tu nombre?",
            "response": "mi nombre es neurachat y me gusta mucho conversar contigo",
            "category": "identity", 
            "importance": "high"
        },
        {
            "context": "",
            "question": "¿qué eres?",
            "response": "soy un sistema de inteligencia artificial basado en arquitectura neuromorfica",
            "category": "identity",
            "importance": "high"
        },
        {
            "context": "",
            "question": "preséntate",
            "response": "hola soy neurachat una red neuronal conversacional muy curiosa y motivada",
            "category": "identity",
            "importance": "high"
        },
        {
            "context": "",
            "question": "háblame de ti",
            "response": "me llamo neurachat y soy una inteligencia artificial que aprende de cada conversacion",
            "category": "identity",
            "importance": "high"
        }
    ]
    
    # === CONTEXT-AWARE CORPUS (3-turn conversations) ===
    context_aware_corpus = [
        {
            "context": "hola como estas",
            "question": "¿y tú qué tal?",
            "response": "yo estoy muy bien gracias por preguntar me siento curioso",
            "category": "contextual",
            "importance": "medium"
        },
        {
            "context": "soy neurachat una inteligencia artificial",
            "question": "¿cómo funciona tu cerebro?",
            "response": "mi cerebro procesa informacion mediante redes neuronales como el cerebro humano",
            "category": "contextual",
            "importance": "medium"
        },
        {
            "context": "me gusta conversar contigo",
            "question": "¿por qué te gusta hablar?",
            "response": "porque cada conversacion me ayuda a aprender y entender mejor el mundo",
            "category": "contextual",
            "importance": "medium"
        },
        {
            "context": "cerebro tiene neuronas conectadas",
            "question": "¿qué son las sinapsis?",
            "response": "las sinapsis son conexiones entre neuronas donde se transmite informacion electrica",
            "category": "contextual",
            "importance": "medium"
        },
        {
            "context": "inteligencia artificial es fascinante",
            "question": "¿puedes aprender?",
            "response": "si aprendo constantemente de cada conversacion y mejoro mis respuestas",
            "category": "contextual",
            "importance": "medium"
        }
    ]
    
    # === ANTI-REPETICIÓN CORPUS ===
    # Ejemplos de lo que NO debe hacer
    anti_repetition_corpus = [
        {
            "context": "",
            "question": "¿cómo funciona el cerebro?",
            "bad_response": "cerebro cerebro cerebro tiene neuronas",
            "good_response": "el cerebro procesa informacion mediante redes de neuronas interconectadas",
            "category": "anti_repetition",
            "importance": "high"
        },
        {
            "context": "",
            "question": "háblame de neurociencia",
            "bad_response": "neurociencia neurociencia es muy neurociencia importante",
            "good_response": "la neurociencia estudia como funciona el sistema nervioso y el cerebro",
            "category": "anti_repetition", 
            "importance": "high"
        }
    ]
    
    # === COHERENCIA GRAMATICAL CORPUS ===
    grammar_corpus = [
        {
            "context": "",
            "question": "¿qué es la memoria?",
            "response": "la memoria es la capacidad del cerebro para almacenar y recuperar informacion",
            "category": "grammar",
            "importance": "medium",
            "grammar_pattern": "articulo + sustantivo + verbo + complemento"
        },
        {
            "context": "",
            "question": "¿cómo aprendes?",
            "response": "aprendo mediante el análisis de patrones en las conversaciones que tengo",
            "category": "grammar",
            "importance": "medium",
            "grammar_pattern": "verbo + preposicion + articulo + sustantivo"
        }
    ]
    
    # === CONVERSACIÓN NATURAL EXTENDIDA ===
    natural_corpus = [
        {
            "context": "",
            "question": "hola",
            "response": "hola me da mucho gusto conversar contigo como estas hoy",
            "category": "greeting",
            "importance": "high"
        },
        {
            "context": "hola me da mucho gusto conversar contigo",
            "question": "bien gracias",
            "response": "que bueno me alegra mucho saber que estas bien",
            "category": "follow_up",
            "importance": "medium"
        },
        {
            "context": "",
            "question": "¿te gustan las conversaciones?",
            "response": "me fascinan porque cada conversacion es una oportunidad para aprender algo nuevo",
            "category": "preference",
            "importance": "medium"
        }
    ]
    
    # === CORPUS NEUROCIENCIA AVANZADO ===
    neuroscience_corpus = [
        {
            "context": "",
            "question": "¿qué es una neurona?",
            "response": "una neurona es una celula especializada que transmite informacion mediante impulsos electricos",
            "category": "neuroscience",
            "importance": "high"
        },
        {
            "context": "neurona transmite informacion electrica",
            "question": "¿cómo se conectan?",
            "response": "las neuronas se conectan mediante sinapsis formando redes complejas de comunicacion",
            "category": "neuroscience",
            "importance": "high"
        },
        {
            "context": "",
            "question": "¿qué es la plasticidad neuronal?",
            "response": "es la capacidad del cerebro para cambiar y adaptarse creando nuevas conexiones",
            "category": "neuroscience",
            "importance": "medium"
        }
    ]
    
    # Combinar todos los corpus
    full_corpus = {
        "metadata": {
            "created_date": datetime.now().isoformat(),
            "version": "2.0",
            "total_entries": 0,
            "categories": ["identity", "contextual", "anti_repetition", "grammar", "greeting", "neuroscience"],
            "purpose": "Enhanced conversational training with context-awareness and identity embedding"
        },
        "identity_embedding": identity_corpus,
        "context_aware": context_aware_corpus,
        "anti_repetition": anti_repetition_corpus,
        "grammar_coherence": grammar_corpus,
        "natural_conversation": natural_corpus,
        "neuroscience_knowledge": neuroscience_corpus
    }
    
    # Calcular total
    total_entries = (len(identity_corpus) + len(context_aware_corpus) + 
                    len(anti_repetition_corpus) + len(grammar_corpus) + 
                    len(natural_corpus) + len(neuroscience_corpus))
    
    full_corpus["metadata"]["total_entries"] = total_entries
    
    return full_corpus

def save_contextual_corpus():
    """Guardar corpus contextualizado"""
    corpus = create_contextual_corpus()
    
    # Guardar corpus completo
    corpus_path = "/home/ubuntu/arquitecture/neuramorphic-ai/training/contextual_corpus.json"
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    
    print("📚 CONTEXTUAL CORPUS GENERADO")
    print("=" * 40)
    print(f"📄 Total entries: {corpus['metadata']['total_entries']}")
    print(f"🎭 Identity entries: {len(corpus['identity_embedding'])}")
    print(f"🔄 Context-aware entries: {len(corpus['context_aware'])}")
    print(f"🚫 Anti-repetition entries: {len(corpus['anti_repetition'])}")
    print(f"📝 Grammar entries: {len(corpus['grammar_coherence'])}")
    print(f"💬 Natural conversation: {len(corpus['natural_conversation'])}")
    print(f"🧠 Neuroscience entries: {len(corpus['neuroscience_knowledge'])}")
    print(f"💾 Guardado en: {corpus_path}")
    
    # Crear dataset de entrenamiento simplificado
    training_conversations = []
    
    # Convertir a formato de entrenamiento
    for category_name, entries in corpus.items():
        if category_name == "metadata":
            continue
            
        for entry in entries:
            if isinstance(entry, dict) and "question" in entry and "response" in entry:
                # Formato: contexto + pregunta + respuesta
                if entry.get("context"):
                    full_text = f"{entry['context']} {entry['question']} {entry['response']}"
                else:
                    full_text = f"{entry['question']} {entry['response']}"
                
                training_conversations.append(full_text)
    
    # Guardar dataset de entrenamiento
    training_path = "/home/ubuntu/arquitecture/neuramorphic-ai/training/contextual_training_data.json"
    training_data = {
        "conversations": training_conversations,
        "total_conversations": len(training_conversations),
        "source": "contextual_corpus_v2.0",
        "created": datetime.now().isoformat()
    }
    
    with open(training_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"🎯 Dataset de entrenamiento: {len(training_conversations)} conversaciones")
    print(f"💾 Guardado en: {training_path}")
    
    return corpus, training_conversations

if __name__ == "__main__":
    print("🧠 GENERANDO CORPUS CONTEXTUALIZADO")
    print("Corpus avanzado para mejorar:")
    print("  ✅ Identity embedding robusto")  
    print("  ✅ Context-awareness (3 turnos)")
    print("  ✅ Anti-repetición interna")
    print("  ✅ Coherencia gramatical")
    print()
    
    corpus, training_data = save_contextual_corpus()
    
    print("\n🌟 CORPUS CONTEXTUALIZADO COMPLETO!")
    print("Listo para integrar en entrenamiento mejorado.")
