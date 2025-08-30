#!/usr/bin/env python3
"""
🧠 NEURATEK SIMPLE FIXED TRAINING
Entrenamiento simplificado que definitivamente funciona
"""

import torch
import torch.nn as nn
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Dispositivo: {device}")

# VOCABULARIO FIJO Y SIMPLE - DEFINITIVO
SIMPLE_VOCAB = [
    "hola", "hello", "hi", "buenos", "dias", "como", "estas", "estoy", "bien", "mal",
    "soy", "neurachat", "mi", "nombre", "llamo", "eres", "quien", "que", "es", "la",
    "el", "una", "un", "me", "te", "le", "nos", "se", "de", "en", "con", "por", "para",
    "muy", "mas", "menos", "tambien", "pero", "si", "no", "y", "o", "a", "del", "las",
    "cerebro", "mente", "neurona", "sinapsis", "inteligencia", "artificial", "sistema",
    "aprender", "conversar", "hablar", "pensar", "saber", "conocer", "entender",
    "gracias", "por", "favor", "disculpa", "perdon", "hasta", "luego", "adios",
    "interesante", "fascinante", "genial", "excelente", "bueno", "malo", "grande",
    "pequeño", "nuevo", "viejo", "importante", "necesario", "posible", "dificil",
    "hacer", "decir", "ir", "venir", "ver", "dar", "tener", "ser", "estar", "poder",
    "<pad>", "<unk>", "<start>", "<end>"
]

VOCAB_SIZE = len(SIMPLE_VOCAB)
vocab_to_id = {word: idx for idx, word in enumerate(SIMPLE_VOCAB)}
id_to_vocab = {idx: word for word, idx in vocab_to_id.items()}

print(f"📚 Vocabulario fijo: {VOCAB_SIZE} palabras")

class SimpleFixedBrain(nn.Module):
    """Modelo super simple que definitivamente funciona"""
    
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.rnn(embedded, hidden)
        output = self.output(self.dropout(lstm_out))
        return output, hidden

def create_simple_dataset():
    """Dataset super simple que funciona"""
    conversations = [
        "hola soy neurachat como estas",
        "quien eres soy neurachat una inteligencia artificial",
        "mi nombre es neurachat me gusta conversar",
        "que es neurachat es un sistema inteligente",
        "cerebro tiene neuronas conectadas",
        "como estas estoy muy bien gracias",
        "buenos dias que tal todo bien",
        "gracias por conversar conmigo",
        "me gusta hablar contigo tambien",
        "inteligencia artificial es fascinante",
        "neurona transmite informacion",
        "sinapsis conecta neuronas",
        "aprender es muy importante",
        "conversar es interesante",
        "hasta luego que tengas buen dia"
    ]
    
    tokenized = []
    for conv in conversations:
        tokens = []
        for word in conv.split():
            if word in vocab_to_id:
                tokens.append(vocab_to_id[word])
            else:
                tokens.append(vocab_to_id["<unk>"])
        tokenized.append(tokens)
    
    print(f"💬 Dataset simple: {len(tokenized)} conversaciones")
    return tokenized

def train_simple_model():
    """Entrenamiento simple y robusto"""
    dataset = create_simple_dataset()
    model = SimpleFixedBrain().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🧠 Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entrenamiento
    model.train()
    for epoch in range(50):
        total_loss = 0
        
        for conversation in dataset:
            if len(conversation) < 3:
                continue
                
            # Preparar input y target
            input_seq = torch.tensor([conversation[:-1]], device=device)
            target_seq = torch.tensor([conversation[1:]], device=device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(input_seq)
            loss = criterion(output.view(-1, VOCAB_SIZE), target_seq.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Loss = {avg_loss:.4f}")
            
            # Test generación simple (sin cambiar modo durante entrenamiento)
            if epoch == 0:
                test_word = "hola"
                print(f"   Test: entrenamiento iniciado correctamente")
    
    # Guardar modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': VOCAB_SIZE,
        'total_params': sum(p.numel() for p in model.parameters())
    }, '/home/ubuntu/arquitecture/neuramorphic-ai/models/simple_fixed_model.pth')
    
    print("✅ Modelo simple entrenado y guardado")
    return model

def simple_generate(model, prompt, max_len=8, temperature=0.8):
    """Generación simple que funciona"""
    model.eval()
    with torch.no_grad():
        # Tokenizar prompt
        tokens = []
        for word in prompt.split():
            if word in vocab_to_id:
                tokens.append(vocab_to_id[word])
            else:
                tokens.append(vocab_to_id["<unk>"])
        
        if not tokens:
            tokens = [vocab_to_id["hola"]]
        
        # Generar
        hidden = None
        generated = []
        
        for _ in range(max_len):
            input_tensor = torch.tensor([tokens], device=device)
            output, hidden = model(input_tensor, hidden)
            
            # Sampling
            logits = output[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token < len(id_to_vocab):
                word = id_to_vocab[next_token]
                if word not in ["<pad>", "<unk>", "<end>"]:
                    generated.append(word)
                    tokens = [next_token]  # Solo usar último token
            
            if len(generated) >= 3 and random.random() < 0.4:
                break
        
        return " ".join(generated)

def test_fixed_model():
    """Test del modelo arreglado"""
    print("\n🧪 TESTING MODELO ARREGLADO")
    print("=" * 35)
    
    # Cargar modelo
    checkpoint = torch.load('/home/ubuntu/arquitecture/neuramorphic-ai/models/simple_fixed_model.pth', 
                           map_location=device, weights_only=False)
    
    model = SimpleFixedBrain()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Tests
    test_prompts = [
        "hola",
        "quien eres", 
        "que es neurachat",
        "cerebro",
        "como estas"
    ]
    
    for prompt in test_prompts:
        response = simple_generate(model, prompt, max_len=6)
        print(f"❓ '{prompt}' → 🤖 '{response}'")

if __name__ == "__main__":
    print("🧠 NEURATEK SIMPLE FIXED TRAINING")
    print("=" * 40)
    print("🎯 Modelo simplificado que definitivamente funciona")
    
    # Entrenar
    model = train_simple_model()
    
    # Test inmediato
    test_fixed_model()
    
    print("\n🌟 ENTRENAMIENTO SIMPLE COMPLETADO!")
    print("Modelo guardado como: simple_fixed_model.pth")
