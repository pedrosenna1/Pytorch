# 🐧 Guia de Instalação e Execução em Linux

Este guia explica como rodar o projeto YOLOv7 + BoT-SORT em um sistema Linux (Ubuntu/Debian).

## ✅ Compatibilidade

O código é **100% compatível com Linux**. As mudanças necessárias são:
- ✅ Dependências (idênticas ao Windows)
- ✅ Paths (já tratados pelo Python)
- ⚠️ Display/Visualização (requer X11 ou salvar em arquivo)
- ✅ Stream RTSP/HTTP (funciona nativamente)

---

## 📋 Pré-requisitos

### 1. Python 3.8+
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### 2. FFmpeg (recomendado, mas opcional para HTTPS)
```bash
sudo apt install ffmpeg
```

**Quando é necessário:**
- Streams **RTSP**: `rtsp://servidor/stream` (obrigatório)
- Streams **HTTPS**: `https://...` (recomendado, mas pode funcionar sem)

**Quando NÃO é necessário:**
- Arquivos locais: `/caminho/video.mp4`
- Câmera USB: `/dev/video0`

Se seu projeto usa apenas **HTTPS** (como no caso atual), você pode pular FFmpeg na primeira tentativa.

### 3. Dependências do Sistema (para OpenCV)
```bash
sudo apt install libsm6 libxext6 libxrender-dev
```

---

## 🚀 Instalação Passo a Passo

### Passo 1: Clonar/Copiar o Projeto
```bash
# Se já tem os arquivos, apenas entre no diretório
cd /caminho/para/Pytorch
```

### Passo 2: Criar Ambiente Virtual
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate
```

### Passo 3: Instalar Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Ou**, para instalação otimizada em Linux:
```bash
pip install -r requirements_linux.txt
```

### Passo 4: Baixar Pesos do YOLOv7
```bash
# Os pesos devem estar em: weights/yolov7.pt
# Se não tiver, baixe manualmente:
cd weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
cd ..
```

---

## ▶️ Executar o Projeto

### Opção 1: Com Display X11 (Mostrar Visualização)
```bash
python3 main.py
```

**Requer:**
- Monitor ou X11 forwarding
- `cv2.imshow()` funciona

### Opção 2: Headless (Sem Display) - Salvar em Arquivo
Se não tiver display ou quiser apenas salvar o vídeo:

Edite `config.yaml`:
```yaml
show_window: false       # Desabilita cv2.imshow()
draw_boxes: true         # Continua desenhando as boxes
```

Depois adicione ao final de `main.py` (antes do `if __name__`):
```python
# Para salvar o vídeo com as detecções
video_writer = cv2.VideoWriter(
    'output.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,  # FPS
    (frame.shape[1], frame.shape[0])
)
```

E no loop principal, após `draw_hud()`:
```python
if video_writer:
    video_writer.write(frame)
```

### Opção 3: Via SSH com X11 Forwarding
```bash
ssh -X usuario@servidor
cd /caminho/para/Pytorch
source venv/bin/activate
python3 main.py
```

---

## 🔧 Configuração do Stream

O projeto suporta:
- **RTSP:** `rtsp://servidor:554/stream`
- **HTTP/HTTPS:** `https://dev.tixxi.rio/outvideo3/?CODE=003215&KEY=G5325`
- **Arquivos locais:** `/caminho/para/video.mp4`
- **Câmera USB:** `0` (para `/dev/video0`)

Edite em `config.yaml`:
```yaml
stream_url: "https://dev.tixxi.rio/outvideo3/?CODE=003215&KEY=G5325"
```

---

## 📊 Monitorar Performance

### Ver uso de CPU/Memória em tempo real
```bash
# Terminal 1: rodar o projeto
python3 main.py

# Terminal 2: monitorar
watch -n 1 'ps aux | grep main.py'
```

### Usar `htop` para visualização melhor
```bash
sudo apt install htop
htop
# Procura por "python3" e vê o uso de recursos
```

---

## 🐛 Troubleshooting

### Erro: `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### Erro: `No module named 'torch'`
```bash
# Para CPU
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html

# Para GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Erro: `RTSP stream timeout`
- Verifique se o FFmpeg está instalado: `ffmpeg -version`
- Teste a URL manualmente: `ffplay rtsp://seu_stream`

### Erro: `Can't connect to stream`
```bash
# Teste conexão
curl -I https://dev.tixxi.rio/outvideo3/?CODE=003215&KEY=G5325
```

### Avisos do PyTorch sobre `torch.meshgrid`
Apenas avisos, funcionam normalmente. Podem ser ignorados.

---

## 📁 Estrutura de Diretórios em Linux

```
Pytorch/
├── main.py                    # Script principal
├── config.yaml               # Configuração
├── requirements.txt          # Dependências
├── requirements_linux.txt    # (opcional, idêntico)
├── detector/
│   └── yolov7_detector.py
├── tracker/
│   ├── bot_sort.py
│   ├── kalman_filter.py
│   ├── matching.py
│   ├── gmc.py
│   └── basetrack.py
├── helpers/
│   ├── counting.py
│   └── drawing.py
├── weights/
│   ├── yolov7.pt            # Download necessário
│   └── yolov7-tiny.pt
└── venv/                     # Ambiente virtual
```

---

## 🎯 Performance em Linux vs Windows

| Aspecto | Linux | Windows |
|---------|-------|---------|
| Startup | ✅ Mais rápido | ⏳ Mais lento |
| FPS | ✅ Similar | ✅ Similar |
| Uso RAM | ✅ Menor | Menor |
| Multiprocessing | ✅ Melhor | Normal |

---

## 📝 Exemplo Completo: Rodar em Modo Batch

```bash
#!/bin/bash
# arquivo: run_detection.sh

cd /home/usuario/Pytorch
source venv/bin/activate

# Rodar com log
python3 main.py > detection.log 2>&1 &
echo $! > detection.pid

# Monitorar por 1 hora
sleep 3600

# Parar
kill $(cat detection.pid)
```

Executar:
```bash
chmod +x run_detection.sh
./run_detection.sh
```

---

## 🔗 Recursos Adicionais

- [YOLOv7 GitHub](https://github.com/WongKinYiu/yolov7)
- [BoT-SORT Paper](https://arxiv.org/abs/2206.14651)
- [PyTorch Linux Guide](https://pytorch.org/get-started/locally/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## ❓ Dúvidas?

Se tiver problemas específicos do Linux:
1. Verifique a versão do Python: `python3 --version`
2. Confirme distribuição: `lsb_release -a`
3. Tente instalar com `--no-cache-dir`: `pip install --no-cache-dir -r requirements.txt`

**Tudo funcionando igual ao Windows!** 🎉
