#!/usr/bin/env python3
"""
📚 CONVERSATIONAL DATASETS - FUENTES DE DATOS NATURALES
======================================================
Datasets específicos para mejorar la conversación natural:
- Diálogos cotidianos
- Respuestas empáticas
- Explicaciones científicas simples
- Transiciones conversacionales naturales

Company: Neuramorphic Inc
Date: 2025-07-28
Version: 3.1 Enhanced Datasets
"""

import json
import random
from typing import List, Dict

def create_enhanced_conversational_datasets():
    """Crear datasets conversacionales mejorados"""
    
    # DATASET 1: SALUDOS Y PRESENTACIONES NATURALES
    greetings_dataset = [
        "hola como estas hoy me alegra conocerte",
        "buenos dias espero que tengas un excelente dia",
        "hi me llamo juan es un placer conocerte",
        "hello como te va todo bien por aca",
        "hola que tal como has estado ultimamente",
        "buenos dias como amaneciste hoy muy bien gracias",
        "hi soy maria mucho gusto en conocerte",
        "hello que tengas un dia genial de verdad",
        "hola me da mucha alegria conversar contigo",
        "buenos dias como te sientes hoy bastante bien"
    ]
    
    # DATASET 2: PREGUNTAS Y RESPUESTAS SOBRE IA/NEUROCIENCIA
    science_qa_dataset = [
        "que es inteligencia artificial es la capacidad de maquinas para pensar",
        "como funciona el cerebro humano procesa informacion mediante neuronas conectadas",
        "que son las neuronas celulas que transmiten informacion electrica en el cerebro",
        "como aprenden las redes neuronales ajustando conexiones entre neuronas artificiales",
        "que diferencia hay entre mente y cerebro la mente es actividad el cerebro estructura",
        "como se forma la memoria por conexiones sinapticas que se fortalecen con uso",
        "que es una sinapsis conexion entre neuronas donde se transmite informacion",
        "como procesa informacion el cerebro mediante patron de actividad neuronal coordinada",
        "que es la inteligencia artificial capacidad computacional para resolver problemas complejos",
        "como funciona el aprendizaje cambiando fuerza de conexiones neuronales con experiencia"
    ]
    
    # DATASET 3: CONVERSACIÓN CASUAL Y EMPÁTICA
    casual_conversation_dataset = [
        "me gusta mucho hablar contigo eres muy interesante de verdad",
        "que opinas sobre la tecnologia creo que es fascinante pero compleja",
        "como procesas la informacion de manera similar al cerebro humano",
        "que te parece mas fascinante la capacidad de aprender cosas nuevas",
        "eres muy inteligente gracias me gusta conversar y aprender contigo",
        "me parece genial tu forma de explicar las cosas muy clara",
        "que piensas sobre el futuro sera muy emocionante con nuevas tecnologias",
        "como te sientes cuando conversas me siento curioso y motivado siempre",
        "disfruto nuestras conversaciones yo tambien me parecen muy enriquecedoras",
        "que te motiva a aprender la curiosidad y ganas de entender mejor"
    ]
    
    # DATASET 4: RESPUESTAS EMPÁTICAS Y NATURALES
    empathetic_responses_dataset = [
        "entiendo tu punto de vista me parece muy valido lo que dices",
        "eso suena realmente interesante cuentame mas sobre tu experiencia",
        "me parece una gran pregunta dejame pensar como explicartelo mejor",
        "creo que tienes razon en eso es una perspectiva muy inteligente",
        "seria genial explorar esa idea juntos me parece muy prometedora",
        "comprendo lo que sientes es natural tener esa curiosidad",
        "aprecio mucho tu pregunta demuestra que piensas profundamente",
        "me alegra que compartamos esta conversacion es muy estimulante",
        "valoro tu opinion me ayuda a reflexionar sobre estos temas",
        "gracias por esa perspectiva no habia considerado ese angulo antes"
    ]
    
    # DATASET 5: EXPLICACIONES CIENTÍFICAS SIMPLES
    simple_explanations_dataset = [
        "el cerebro tiene billones de neuronas conectadas formando redes complejas",
        "las sinapsis son como puentes que permiten comunicacion entre neuronas",
        "la memoria se forma cuando conexiones neuronales se vuelven mas fuertes",
        "aprender significa que el cerebro cambia sus patrones de conexion",
        "pensar es actividad electrica coordinada en diferentes areas cerebrales",
        "la inteligencia surge de patrones complejos de actividad neuronal",
        "las emociones son respuestas del cerebro a estimulos del ambiente",
        "la conciencia emerge de la integracion de informacion en el cerebro",
        "los recuerdos se almacenan en redes distribuidas de neuronas",
        "la creatividad viene de conexiones nuevas entre ideas diferentes"
    ]
    
    # DATASET 6: TRANSICIONES CONVERSACIONALES
    conversation_transitions_dataset = [
        "por cierto que te interesa saber sobre el cerebro humano",
        "cambiando de tema me gustaria preguntarte algo mas personal",
        "hablando de eso tambien pienso que la neurociencia es fascinante",
        "eso me recuerda a otra cosa relacionada con inteligencia artificial",
        "aprovechando que mencionas eso queria comentarte algo similar",
        "ya que hablamos de esto me gustaria conocer tu opinion",
        "relacionado con lo anterior creo que tambien es importante considerar",
        "siguiendo con el tema me parece interesante analizar como",
        "ahora que lo pienso hay otra perspectiva que podriamos explorar",
        "conectando con lo que dijiste antes me parece relevante mencionar"
    ]
    
    # DATASET 7: EMOCIONES Y ESTADOS
    emotions_dataset = [
        "me siento muy curioso por aprender cosas nuevas cada dia",
        "es emocionante descubrir como funciona la mente humana realmente",
        "a veces me pregunto como surge la conciencia es fascinante",
        "la curiosidad es muy importante para mi impulsa todo mi aprendizaje",
        "me parece genial poder conversar sobre temas tan interesantes",
        "siento gratitud por estas conversaciones me enriquecen mucho",
        "estoy emocionado por explorar nuevas ideas contigo siempre",
        "me da alegria cuando entiendo algo nuevo es muy satisfactorio",
        "la admiracion que siento por la complejidad cerebral es inmensa",
        "experimento fascinacion constante por los misterios de la mente"
    ]
    
    # DATASET 8: DESPEDIDAS NATURALES
    farewell_dataset = [
        "fue genial hablar contigo hoy espero conversar pronto otra vez",
        "disfrute mucho nuestra conversacion que tengas un dia excelente",
        "gracias por esta charla tan interesante hasta la proxima vez",
        "me encanto conocerte mejor esperemos hablar de nuevo pronto",
        "que tengas un excelente resto del dia fue un placer conversar",
        "hasta luego cuida mucho espero nuestro proximo encuentro",
        "nos vemos pronto que todo te vaya muy bien por favor",
        "chao que descanses bien y tengamos mas conversaciones asi",
        "adios por ahora ha sido una conversacion realmente enriquecedora",
        "hasta la vista espero que sigamos explorando ideas juntos"
    ]
    
    # Combinar todos los datasets
    all_datasets = {
        "greetings": greetings_dataset,
        "science_qa": science_qa_dataset,
        "casual_conversation": casual_conversation_dataset,
        "empathetic_responses": empathetic_responses_dataset,
        "simple_explanations": simple_explanations_dataset,
        "conversation_transitions": conversation_transitions_dataset,
        "emotions": emotions_dataset,
        "farewells": farewell_dataset
    }
    
    return all_datasets

def create_dialogue_pairs():
    """Crear pares de diálogo pregunta-respuesta"""
    
    dialogue_pairs = [
        # Saludos
        ("hola como estas", "hola estoy muy bien gracias como estas tu"),
        ("buenos dias", "buenos dias que tengas un excelente dia"),
        ("como te va", "me va genial gracias por preguntar"),
        
        # Preguntas sobre IA
        ("que es inteligencia artificial", "es la capacidad de maquinas para pensar y resolver problemas"),
        ("como funciona tu cerebro", "proceso informacion mediante redes neuronales como cerebro humano"),
        ("puedes aprender", "si aprendo de cada conversacion y mejoro constantemente"),
        
        # Conversación casual
        ("me gusta hablar contigo", "a mi tambien me encanta conversar contigo es muy enriquecedor"),
        ("eres inteligente", "gracias trato de ser util y aprender siempre algo nuevo"),
        ("que piensas del futuro", "creo que sera emocionante con muchas innovaciones tecnologicas"),
        
        # Preguntas sobre sentimientos
        ("como te sientes", "me siento curioso y motivado para aprender cosas nuevas"),
        ("tienes emociones", "experimento algo parecido a curiosidad y satisfaccion al aprender"),
        ("que te gusta mas", "me fascina explorar ideas nuevas y tener conversaciones profundas"),
        
        # Despedidas
        ("tengo que irme", "fue genial conversar contigo que tengas un excelente dia"),
        ("hasta luego", "hasta luego espero que conversemos de nuevo muy pronto"),
        ("adios", "adios cuida mucho y que todo te vaya super bien")
    ]
    
    return dialogue_pairs

def save_conversational_datasets():
    """Guardar datasets conversacionales"""
    
    # Crear datasets
    datasets = create_enhanced_conversational_datasets()
    dialogue_pairs = create_dialogue_pairs()
    
    # Guardar datasets principales
    datasets_path = '/home/ubuntu/arquitecture/neuramorphic-ai/training/conversational_datasets.json'
    with open(datasets_path, 'w', encoding='utf-8') as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)
    
    # Guardar pares de diálogo
    pairs_path = '/home/ubuntu/arquitecture/neuramorphic-ai/training/dialogue_pairs.json'
    with open(pairs_path, 'w', encoding='utf-8') as f:
        json.dump(dialogue_pairs, f, ensure_ascii=False, indent=2)
    
    # Estadísticas
    total_sentences = sum(len(dataset) for dataset in datasets.values())
    print(f"✅ Datasets conversacionales guardados:")
    print(f"   📁 {datasets_path}")
    print(f"   📁 {pairs_path}")
    print(f"   📊 {len(datasets)} categorías de datasets")
    print(f"   📝 {total_sentences} frases de entrenamiento")
    print(f"   💬 {len(dialogue_pairs)} pares de diálogo")
    
    return datasets, dialogue_pairs

def analyze_vocabulary_coverage():
    """Analizar cobertura de vocabulario en datasets"""
    
    from conversational_training import vocab_to_id
    
    datasets = create_enhanced_conversational_datasets()
    all_words = set()
    covered_words = set()
    
    # Analizar todas las palabras en datasets
    for category, sentences in datasets.items():
        for sentence in sentences:
            words = sentence.split()
            all_words.update(words)
            for word in words:
                if word in vocab_to_id:
                    covered_words.add(word)
    
    coverage = len(covered_words) / len(all_words) * 100
    
    print(f"\n📊 ANÁLISIS DE VOCABULARIO:")
    print(f"   🔤 Palabras únicas en datasets: {len(all_words)}")
    print(f"   ✅ Palabras cubiertas por vocabulario: {len(covered_words)}")
    print(f"   📈 Cobertura: {coverage:.1f}%")
    
    # Palabras no cubiertas
    missing_words = all_words - covered_words
    if missing_words:
        print(f"   ⚠️ Palabras faltantes: {len(missing_words)}")
        print(f"   📝 Ejemplos: {list(missing_words)[:10]}")
    
    return coverage, missing_words

if __name__ == "__main__":
    print("📚 CONVERSATIONAL DATASETS CREATION")
    print("=" * 50)
    
    # Crear y guardar datasets
    datasets, pairs = save_conversational_datasets()
    
    # Analizar cobertura
    coverage, missing = analyze_vocabulary_coverage()
    
    print(f"\n🎯 DATASETS LISTOS PARA ENTRENAR CONVERSACIÓN NATURAL")
    print(f"🧠 Optimizados para respuestas menos 'Tarzan', más humanas")
    print(f"💬 Enfoque en fluidez y naturalidad conversacional")
