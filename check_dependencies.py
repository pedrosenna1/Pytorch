"""
Script para verificar se todas as dependências estão instaladas
"""

import sys

def check_dependency(module_name, package_name=None):
    """Tenta importar um módulo"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name} - OK")
        return True
    except ImportError:
        print(f"❌ {package_name} - FALTANDO")
        return False

print("=" * 50)
print("Verificando dependências do projeto...")
print("=" * 50)

dependencies = [
    ("torch", "PyTorch"),
    ("cv2", "OpenCV (opencv-python)"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("yaml", "PyYAML"),
    ("lap", "lap (Linear Assignment Problem)"),
    ("matplotlib", "matplotlib (opcional)"),
]

missing = []
for module, name in dependencies:
    if not check_dependency(module, name):
        missing.append(name)

print("\n" + "=" * 50)

if not missing:
    print("✅ Todas as dependências necessárias estão instaladas!")
    print("\nVocê pode executar o projeto com:")
    print("  python main.py")
else:
    print(f"❌ Faltam {len(missing)} dependências:")
    for dep in missing:
        print(f"  - {dep}")
    print("\nInstale com:")
    print("  pip install -r requirements.txt")

print("=" * 50)

# Verificar arquivos importantes
import os

print("\nVerificando arquivos do projeto...")
files = [
    "config.yaml",
    "main.py",
    "detector/yolov7_detector.py",
    "tracker/bot_sort.py",
    "helpers/counting.py",
    "helpers/drawing.py",
    "weights/yolov7-tiny.pt",
]

for file in files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - FALTANDO")

print("\n" + "=" * 50)
print("Verificação concluída!")
print("=" * 50)
