# Sistema de Detecção e Contagem de Objetos
## YOLOv7 + BoT-SORT Tracker

Sistema completo para detectar e contar pessoas, carros, motos, ônibus e caminhões em tempo real com tracking para evitar contagem duplicada.

## 🎯 Características

- ✅ **Detecção**: YOLOv7 (tiny ou completo)
- ✅ **Tracking**: BoT-SORT (evita contagem duplicada)
- ✅ **Contagem Única**: Cada objeto é contado apenas uma vez
- ✅ **Multiplicadores**: Pesos diferentes por classe (ex: ônibus = 20 pessoas)
- ✅ **Visualização**: Bounding boxes coloridas por classe + HUD informativo

## 📦 Instalação

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar pesos do YOLOv7 (se ainda não tiver)
# Os pesos já devem estar em weights/yolov7-tiny.pt ou weights/yolov7.pt
```

## 🚀 Como Usar

```bash
python main.py
```

### Controles durante execução:
- **ESC**: Encerrar o programa
- **R**: Resetar contador

## ⚙️ Configuração

Edite o arquivo `config.yaml` para ajustar:

### Classes detectadas (COCO dataset IDs):
```yaml
classes:
  person: 0
  car: 2
  motorcycle: 3
  bus: 5
  truck: 7
```

### Multiplicadores (quantas "pessoas equivalentes"):
```yaml
multipliers:
  person: 1
  car: 1.5
  motorcycle: 1
  bus: 20      # Um ônibus = 20 pessoas
  truck: 1
```

### Parâmetros de detecção:
```yaml
conf_thres: 0.35        # Confiança mínima (0-1)
device: "cpu"           # Ou "cuda:0" se tiver GPU
weights: "weights/yolov7-tiny.pt"
```

### Parâmetros de tracking:
```yaml
track_high_thresh: 0.45  # Threshold alto para detecções
track_low_thresh: 0.10   # Threshold baixo 
new_track_thresh: 0.45   # Threshold para novos tracks
match_thresh: 0.72       # Threshold para matching
track_buffer: 120        # Frames que track pode ficar perdido
```

## 📊 Como Funciona

1. **Detecção**: YOLOv7 detecta objetos em cada frame
2. **Filtragem**: Apenas classes configuradas são processadas
3. **Tracking**: BoT-SORT associa detecções aos mesmos objetos
4. **Contagem Única**: Cada track_id único é contado apenas uma vez
5. **Multiplicadores**: Total ponderado = Σ(contagem × multiplicador)

### Exemplo de saída:
```
Frame: 450
Deteccoes: 8
Tracks: 5

TOTAL Ponderado: 24.5

car: 2 (x1.5 = 3.0)
person: 3 (x1 = 3.0)
bus: 1 (x20 = 20.0)
```

## 📁 Estrutura do Projeto

```
Pytorch/
├── main.py                 # Script principal
├── config.yaml            # Configurações
├── requirements.txt       # Dependências
│
├── detector/
│   └── yolov7_detector.py  # Wrapper do YOLOv7
│
├── tracker/
│   ├── bot_sort.py         # BoT-SORT tracker
│   ├── basetrack.py        # Base do tracking
│   ├── kalman_filter.py    # Filtro de Kalman
│   ├── matching.py         # Algoritmos de matching
│   └── gmc.py              # Compensação de movimento
│
├── helpers/
│   ├── counting.py         # Sistema de contagem única
│   └── drawing.py          # Funções de visualização
│
└── weights/
    ├── yolov7-tiny.pt      # Pesos do modelo
    └── yolov7.pt
```

## 🎨 Cores das Bounding Boxes

- 🟢 **Verde**: Pessoas
- 🔵 **Azul**: Carros
- 🟠 **Laranja**: Motos
- 🔴 **Vermelho**: Ônibus
- 🔷 **Ciano**: Caminhões

## 🔧 Troubleshooting

### Erro: "Module not found"
```bash
pip install -r requirements.txt
```

### Stream não abre
- Verifique a URL no `config.yaml`
- Teste com arquivo de vídeo local: `stream_url: "video.mp4"`

### Performance ruim
- Use `yolov7-tiny.pt` (mais rápido)
- Aumente `conf_thres` para reduzir detecções
- Use GPU: `device: "cuda:0"`

### Muitos falsos positivos
- Aumente `conf_thres` (ex: 0.5)
- Aumente `track_high_thresh` (ex: 0.6)

### Tracks perdendo objetos
- Aumente `track_buffer` (ex: 180)
- Diminua `track_low_thresh` (ex: 0.05)

## 📝 Notas

- O contador **não reseta** automaticamente entre frames
- Pressione **R** para resetar manualmente durante execução
- Use **with_reid: true** para melhor tracking (requer fast_reid)
- GMC (compensação de movimento) melhora tracking em câmeras móveis

## 📄 Licença

Baseado em:
- YOLOv7: https://github.com/WongKinYiu/yolov7
- BoT-SORT: https://github.com/NirAharon/BoT-SORT
