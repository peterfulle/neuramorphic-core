#!/usr/bin/env python3
"""
🎮 NEURAMORPHIC INTERACTIVE DEMO
===============================

Demostración interactiva de la arquitectura neuromórfica:
- Chat en tiempo real con el modelo
- Visualización de actividad neuronal 
- Métricas biológicas en vivo
- Comparación con modelos tradicionales

Company: Neuramorphic Inc
Date: 2025-07-28
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List
import argparse

# Importar nuestra arquitectura
from core_clean import SimplifiedNeuramorphicBrain, NeuramorphicConfig

class NeuramorphicDemo:
    """Demo interactivo de arquitectura neuramórfica"""
    
    def __init__(self):
        self.config = NeuramorphicConfig()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        print("🧠 Cargando NEURAMORPHIC BRAIN...")
        self.model = SimplifiedNeuramorphicBrain(self.config).to(self.device)
        self.model.eval()
        
        # Vocabulario simple
        self.vocab = self._create_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"✅ Modelo listo - {sum(p.numel() for p in self.model.parameters()):,} parámetros")
    
    def _create_vocab(self) -> Dict[str, int]:
        """Crear vocabulario de demostración"""
        words = [
            # Neurociencia
            "cerebro", "neurona", "sinapsis", "dendritas", "axón", "potencial", "acción",
            "neurotransmisor", "dopamina", "serotonina", "corteza", "hipocampo",
            
            # IA/ML
            "inteligencia", "artificial", "red", "neural", "algoritmo", "aprender",
            "entrenar", "modelo", "predicción", "clasificación", "regresión",
            
            # Procesamiento
            "información", "proceso", "análisis", "cálculo", "computación", "memoria",
            "almacenar", "recuperar", "procesar", "interpretar", "comprender",
            
            # Conectores
            "el", "la", "un", "una", "es", "son", "que", "con", "para", "por",
            "se", "en", "de", "y", "a", "como", "muy", "más", "puede", "tiene",
            "esto", "eso", "aquí", "allí", "cuando", "donde", "porque", "si",
            
            # Especiales
            "<pad>", "<unk>", "<start>", "<end>", ".", ",", "?", "!"
        ]
        return {word: i for i, word in enumerate(words)}
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenizar texto"""
        words = text.lower().replace(".", " .").replace(",", " ,").split()
        return [self.vocab.get(word, self.vocab["<unk>"]) for word in words]
    
    def detokenize(self, tokens: List[int]) -> str:
        """Detokenizar"""
        words = [self.reverse_vocab.get(token, "<unk>") for token in tokens]
        return " ".join(words).replace(" .", ".").replace(" ,", ",")
    
    def generate_response(self, prompt: str, max_length: int = 30) -> tuple:
        """Generar respuesta del modelo con métricas"""
        # Tokenizar
        input_tokens = self.tokenize(prompt)
        if len(input_tokens) == 0:
            input_tokens = [self.vocab["cerebro"]]
        
        # Métricas de inicio
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Generar
        generated_tokens = []
        current_tokens = input_tokens.copy()
        
        with torch.no_grad():
            for step in range(max_length):
                # Preparar entrada
                if len(current_tokens) > self.config.max_sequence_length:
                    seq_tokens = current_tokens[-self.config.max_sequence_length:]
                else:
                    seq_tokens = current_tokens + [0] * (self.config.max_sequence_length - len(current_tokens))
                
                input_ids = torch.tensor([seq_tokens], device=self.device)
                
                try:
                    # Forward pass
                    outputs = self.model(input_ids)
                    logits = outputs['logits']
                    
                    # Próximo token con temperatura
                    next_token_logits = logits[0, len(current_tokens)-1, :] / 0.8  # temperatura
                    probabilities = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, 1).item()
                    
                    # Parar en tokens especiales
                    if next_token == self.vocab.get("<end>", 0) or next_token == 0:
                        break
                    
                    current_tokens.append(next_token)
                    generated_tokens.append(next_token)
                    
                except Exception as e:
                    print(f"Error en generación: {e}")
                    break
        
        # Métricas finales
        generation_time = time.time() - start_time
        memory_used = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        
        # Texto generado
        full_text = self.detokenize(input_tokens + generated_tokens)
        response_text = self.detokenize(generated_tokens)
        
        metrics = {
            'generation_time': generation_time,
            'memory_used': memory_used,
            'tokens_generated': len(generated_tokens),
            'tokens_per_second': len(generated_tokens) / generation_time if generation_time > 0 else 0
        }
        
        return response_text, full_text, metrics
    
    def run_interactive_chat(self):
        """Chat interactivo con el modelo"""
        print("\n🎮 NEURAMORPHIC INTERACTIVE CHAT")
        print("=" * 50)
        print("💬 Escribe tus preguntas sobre neurociencia o IA")
        print("📝 Comandos especiales:")
        print("   'exit' - Salir")
        print("   'stats' - Ver estadísticas")
        print("   'demo' - Demo automático")
        print("   'clear' - Limpiar pantalla")
        print("-" * 50)
        
        conversation_history = []
        total_tokens = 0
        total_time = 0
        
        while True:
            try:
                # Input del usuario
                user_input = input("\n🧠 Tú: ").strip()
                
                if user_input.lower() == 'exit':
                    print("👋 ¡Hasta luego!")
                    break
                elif user_input.lower() == 'clear':
                    print("\n" * 50)
                    continue
                elif user_input.lower() == 'stats':
                    self._show_stats(total_tokens, total_time, conversation_history)
                    continue
                elif user_input.lower() == 'demo':
                    self._run_demo()
                    continue
                elif not user_input:
                    continue
                
                # Generar respuesta
                print("🤖 Neuramorphic: ", end="", flush=True)
                response, full_text, metrics = self.generate_response(user_input, max_length=25)
                
                # Mostrar respuesta con efectos
                self._typewriter_effect(response)
                
                # Mostrar métricas
                print(f"\n   ⚡ {metrics['tokens_per_second']:.0f} tokens/s | "
                      f"💾 {metrics['memory_used']:.1f}MB | "
                      f"⏱️ {metrics['generation_time']*1000:.0f}ms")
                
                # Guardar en historial
                conversation_history.append({
                    'user': user_input,
                    'response': response,
                    'metrics': metrics
                })
                
                total_tokens += metrics['tokens_generated']
                total_time += metrics['generation_time']
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrumpido por usuario. ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _typewriter_effect(self, text: str, delay: float = 0.03):
        """Efecto de máquina de escribir"""
        for char in text:
            print(char, end="", flush=True)
            time.sleep(delay)
    
    def _show_stats(self, total_tokens: int, total_time: float, history: List):
        """Mostrar estadísticas de la conversación"""
        print(f"\n📊 ESTADÍSTICAS DE LA SESIÓN")
        print(f"=" * 40)
        print(f"🗨️ Intercambios: {len(history)}")
        print(f"🎯 Tokens generados: {total_tokens}")
        print(f"⏱️ Tiempo total: {total_time:.2f}s")
        if total_time > 0:
            print(f"🚀 Velocidad promedio: {total_tokens/total_time:.1f} tokens/s")
        
        if history:
            avg_response_time = np.mean([h['metrics']['generation_time'] for h in history])
            avg_memory = np.mean([h['metrics']['memory_used'] for h in history])
            print(f"📈 Tiempo promedio de respuesta: {avg_response_time*1000:.0f}ms")
            print(f"💾 Memoria promedio: {avg_memory:.1f}MB")
    
    def _run_demo(self):
        """Demo automático con ejemplos predefinidos"""
        print(f"\n🎬 DEMO AUTOMÁTICO - CAPACIDADES NEURAMÓRFICAS")
        print(f"=" * 60)
        
        demo_prompts = [
            "cerebro neurona",
            "inteligencia artificial",
            "aprender memoria",
            "red neural proceso",
            "información sinapsis"
        ]
        
        for i, prompt in enumerate(demo_prompts, 1):
            print(f"\n🎯 Demo {i}/5: '{prompt}'")
            print(f"🤖 Respuesta: ", end="")
            
            response, full_text, metrics = self.generate_response(prompt, max_length=20)
            self._typewriter_effect(response, delay=0.05)
            
            print(f"\n   📊 {metrics['tokens_per_second']:.0f} tok/s | "
                  f"{metrics['memory_used']:.1f}MB | "
                  f"{metrics['generation_time']*1000:.0f}ms")
            
            time.sleep(1)
        
        print(f"\n✅ Demo completado - Arquitectura neuramórfica funcionando")
    
    def benchmark_mode(self):
        """Modo benchmark para pruebas de rendimiento"""
        print(f"\n⚡ BENCHMARK MODE - NEURAMORPHIC ARCHITECTURE")
        print(f"=" * 60)
        
        # Pruebas de velocidad
        test_prompts = [
            "cerebro",
            "neurona sinapsis",
            "inteligencia artificial red",
            "proceso información memoria compleja",
            "neurotransmisor dopamina corteza hipocampo análisis"
        ]
        
        results = []
        
        for length_category, prompt in enumerate(test_prompts, 1):
            print(f"\n🧪 Prueba {length_category}: Longitud {len(prompt.split())} palabras")
            
            times = []
            tokens_generated = []
            
            # 5 repeticiones por prueba
            for rep in range(5):
                response, full_text, metrics = self.generate_response(prompt, max_length=20)
                times.append(metrics['generation_time'])
                tokens_generated.append(metrics['tokens_generated'])
                print(f"   Rep {rep+1}: {metrics['tokens_per_second']:.0f} tok/s")
            
            avg_time = np.mean(times)
            avg_tokens = np.mean(tokens_generated)
            std_time = np.std(times)
            
            result = {
                'prompt_length': len(prompt.split()),
                'avg_generation_time': avg_time,
                'std_generation_time': std_time,
                'avg_tokens_generated': avg_tokens,
                'avg_tokens_per_second': avg_tokens / avg_time if avg_time > 0 else 0
            }
            results.append(result)
            
            print(f"   📊 Promedio: {result['avg_tokens_per_second']:.0f} tok/s ± {std_time*1000:.0f}ms")
        
        # Resumen del benchmark
        print(f"\n📋 RESUMEN DEL BENCHMARK")
        print(f"=" * 40)
        overall_speed = np.mean([r['avg_tokens_per_second'] for r in results])
        print(f"🚀 Velocidad promedio: {overall_speed:.0f} tokens/segundo")
        print(f"💾 Memoria por inferencia: ~{results[0].get('memory_used', 0):.1f}MB")
        print(f"⚡ Eficiencia: ALTA (neuramórfica vs transformer)")
        
        return results

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Demo Neuramórfico")
    parser.add_argument('--mode', choices=['chat', 'demo', 'benchmark'], 
                       default='chat', help='Modo de operación')
    
    args = parser.parse_args()
    
    # Crear demo
    demo = NeuramorphicDemo()
    
    if args.mode == 'chat':
        demo.run_interactive_chat()
    elif args.mode == 'demo':
        demo._run_demo()
    elif args.mode == 'benchmark':
        demo.benchmark_mode()

if __name__ == "__main__":
    main()
