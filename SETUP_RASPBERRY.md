# 🍓 Setup para Raspberry Pi com TensorFlow Lite

Este guia configura o projeto para rodar em Raspberry Pi 4 com TensorFlow Lite.

## 📋 Pré-requisitos

### Hardware
- **Raspberry Pi 4** (2GB mínimo, 4GB recomendado)
- **Cartão SD**: 32GB+
- **Fonte de alimentação**: 5V/3A mínimo

### Sistema Operacional
```bash
# Verificar versão
lsb_release -a
cat /proc/cpuinfo | grep model
```

Testado em:
- ✅ **Armbian** (Debian/Ubuntu-based) - RECOMENDADO
- ✅ Raspberry Pi OS (Bullseye/Bookworm)
- ✅ Ubuntu 22.04 ARM64

---

## 🚀 Instalação Passo a Passo

### 1. Atualizar Sistema
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install python3-pip python3-venv libatlas-base-dev ffmpeg -y
```

### 2. Criar Ambiente Virtual
```bash
cd ~/Pytorch  # ou seu diretório
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependências
```bash
# Atualizar pip
pip install --upgrade pip setuptools wheel

# Instalar dependências base
pip install numpy scipy PyYAML opencv-python lap

# Instalar PyTorch para ARM (SEM CUDA)
# IMPORTANTE: Use versão pré-compilada para ARM
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar TensorFlow Lite
pip install tensorflow
```

**Aviso:** A instalação do TensorFlow pode levar 10-30 minutos em Raspberry Pi.

### 4. (Opcional) Google Coral TPU
Se tiver Coral TPU USB:
```bash
pip install pycoral
```

### 5. Obter Modelo TFLite

**Opção A: Baixar de Hugging Face (MAIS CONFIÁVEL)**

```bash
cd weights

# YOLOv5n INT8 (recomendado para Pi)
wget https://huggingface.co/spaces/deepquest/yolov5-tflite/resolve/main/yolov5n-int8.tflite

# Ou YOLOv5s INT8 (mais acurado)
wget https://huggingface.co/spaces/deepquest/yolov5-tflite/resolve/main/yolov5s-int8.tflite

cd ..
```

**Opção B: Baixar do GitHub (se opção A não funcionar)**

```bash
cd weights

# YOLOv5n
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-int8.tflite 2>/dev/null || \
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n-int8.tflite

cd ..
```

**Opção C: Converter localmente (seu computador)**

Se nenhum dos links funcionar, converta no seu **Windows/Linux**:

```bash
# No seu computador (NÃO no Raspberry)
pip install ultralytics

python3 << 'EOF'
from ultralytics import YOLO

# Carregar modelo
model = YOLO('yolov5n.pt')

# Exportar para TFLite INT8
model.export(format='tflite', imgsz=640, int8=True)

# Arquivo gerado: yolov5n-int8.tflite
EOF

# Depois copiar para Raspberry:
scp yolov5n-int8.tflite usuario@raspberry:~/Pytorch/weights/
```

**Verificar se o arquivo foi baixado:**
```bash
ls -lh ~/Pytorch/weights/*.tflite
```

**Opção B: Converter seu próprio modelo**
```bash
# No seu computador (não no Raspberry)
python3 export_to_tflite.py  # Script que criaremos
```

---

## ⚙️ Configuração

### Editar `config.yaml`
```yaml
detector_type: "tflite"           # Usar TFLite em vez de YOLOv7
tflite_model: "weights/yolov5n-int8.tflite"
tflite_use_coral: false           # true se tiver Coral TPU

imgsz: 640                        # Mesmo tamanho (TFLite otimiza internamente)
conf_vehicle: 0.45
conf_person: 0.40
conf_bicycle: 0.35

show_window: false                # Desabilitar display (não há X11 no Pi)
```

### (Opcional) Criar `.env`
```bash
cp .env.example .env
# Editar conforme necessário
```

---

## ▶️ Executar

### Rodar Detecção
```bash
python3 main.py
```

**Saída esperada:**
```
🔧 Usando TensorFlow Lite Detector (Raspberry Pi)...
✅ Modelo TFLite carregado: weights/yolov5n-int8.tflite
   Entrada: [1, 640, 640, 3]
   Outputs: 3

Stream aberto: https://...
Iniciando detecção e tracking...

Frame 30 | Tracks: 5 | Total: 12.5
Frame 60 | Tracks: 6 | Total: 18.0
```

### Monitorar Performance
```bash
# Terminal 1: Rodar projeto
python3 main.py

# Terminal 2: Monitorar em tempo real (Armbian)
watch -n 1 'top -bn1 | head -n 10 && free -h'

# Verificar temperatura (Armbian/Debian)
# Pode variar conforme o hardware
cat /sys/class/thermal/thermal_zone*/temp
```

---

## 📊 Performance Esperada

### YOLOv5n INT8 em Raspberry Pi 4

| Métrica | Esperado |
|---------|----------|
| **FPS** | 3-5 FPS |
| **Latência** | 200-300ms por frame |
| **CPU** | 80-95% |
| **RAM** | 1.5-2 GB |
| **Temperatura** | 55-65°C (normal) |

### Com Google Coral TPU
| Métrica | Esperado |
|---------|----------|
| **FPS** | 8-12 FPS |
| **Latência** | 80-120ms por frame |
| **CPU** | 30-50% |
| **RAM** | 1.5-2 GB |

---

## 🔧 Troubleshooting

### Erro: `No module named 'tensorflow'`
```bash
# Instalar novamente (pode levar tempo)
pip install --no-cache-dir tensorflow
```

### Erro: `Illegal instruction (core dumped)`
- Seu Pi usa CPU incompatível
- Solução: Use imagem Raspberry Pi OS com suporte ARMv7
- Ou: Compile TensorFlow localmente

### Velocidade muito lenta (< 1 FPS)
**Causas possíveis:**
1. RAM insuficiente (verifique com `free -h`)
2. CPU throttling (temperatura alta)
3. Modelo pesado (YOLOv5s em vez de YOLOv5n)

**Soluções:**
```bash
# Verificar temperatura (Armbian)
cat /sys/class/thermal/thermal_zone*/temp

# Ou no htop
htop  # Procura pela coluna TEMP

# Usar modelo menor
detector_type: "tflite"
tflite_model: "weights/yolov5n-int8.tflite"  # Menor = mais rápido

# Reduzir imagem de entrada
imgsz: 416  # Em vez de 640
```

### Memoria insuficiente durante inferência
```bash
# Adicionar swap (Armbian)
sudo nano /etc/dphys-swapfile  # Se existir
# Ou criar manualmente:
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 💾 Salvar Resultados (Headless Mode)

Se não tem display, você pode salvar vídeo com detecções:

Edite `main.py` (após inicializar detector):
```python
video_writer = None
if not cfg.get("show_window", True):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        'detections_output.mp4',
        fourcc,
        5,  # 5 FPS (estimado)
        (frame.shape[1], frame.shape[0])
    )
```

Depois no loop, após `draw_hud()`:
```python
if video_writer:
    video_writer.write(frame)

# No final, antes de sair:
if video_writer:
    video_writer.release()
```

---

## 📤 Transferir Vídeo do Pi

```bash
# Do seu computador
scp usuario@raspberrypi:~/Pytorch/detections_output.mp4 ./

# Ou com rsync
rsync -avz --progress usuario@raspberrypi:~/Pytorch/ ./backup/
```

---

## 🚀 Próximos Passos

1. **Otimizar modelos**: Tentar YOLOv5n, YOLOv5s diferentes
2. **Google Coral TPU**: Se performance não é suficiente
3. **Quantização**: Converter modelo para INT8 (mais rápido)
4. **Múltiplos Pis**: Usar em rede com load balancing

---

## 📚 Referências

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [YOLOv5 Export Guide](https://github.com/ultralytics/yolov5/wiki/Export)
- [Google Coral Docs](https://coral.ai/docs/)
- [Raspberry Pi Performance](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)

---

## ⚡ Dicas Finais

- **Sempre use fonte com 5V/3A** (fonte fraca causa resets)
- **Use ventilador** (throttling mata performance)
- **SSD externo** (cartão SD é lento)
- **Monitor de temperatura** (`cat /sys/class/thermal/thermal_zone*/temp`)
- **Armbian geralmente tem melhor performance** que Raspberry Pi OS

**Boa sorte! 🍓🚀**
