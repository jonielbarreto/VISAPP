import os
import json
import shutil
from PIL import Image
from pathlib import Path
from sahi.slicing import slice_coco
from ultralytics.data.converter import convert_coco

def create_dataset(src_dir, dest_dir):
    """
    Ajusta e remapeia as classes dos labels originais para o formato YOLOv8.
    Corrigido para lidar com labels que já possuem o prefixo do vídeo no nome original.
    
    Regra de classes:
    2 e 3 -> Viram 0
    0 e 1 -> Mantém igual
    """
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    
    # Divisão dos vídeos
    split_mapping = {
        'video1': 'train',
        'video2': 'train',
        'video4': 'val',
        'video5': 'val',
        'video3': 'test'
    }
    
    print("Iniciando o processamento corrigido dos labels...")

    for video_name, split in split_mapping.items():
        video_folder = src_path / video_name
        
        if not video_folder.exists():
            print(f"Aviso: Pasta {video_name} não encontrada. Pulando...")
            continue
            
        images_src = video_folder / 'images'
        labels_src = video_folder / 'labels'
        
        if not images_src.exists() or not labels_src.exists():
            print(f"Aviso: 'images' ou 'labels' ausentes em {video_name}. Pulando...")
            continue
            
        # Lista todas as imagens originais (ex: frame_00000.jpg)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in images_src.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        
        for img_file in image_files:
            # --- PARTE DE COPIAR IMAGENS COMENTADA ---
            # img_dest = dest_path / split / 'images' / f"{video_name}_{img_file.name}"
            # shutil.copy2(img_file, img_dest)
            # ----------------------------------------
            
            # CORREÇÃO: O label original já se chama "videoX_frame_00000.txt"
            nome_label_original = f"{video_name}_{img_file.stem}.txt"
            label_file_origem = labels_src / nome_label_original
            
            # O destino final deve manter o mesmo nome padronizado do seu novo dataset
            label_file_destino = dest_path / split / 'labels' / nome_label_original
            
            if label_file_origem.exists():
                linhas_modificadas = []
                with open(label_file_origem, 'r') as f:
                    for linha in f:
                        partes = linha.strip().split()
                        if not partes:
                            continue
                            
                        class_id = int(partes[0])
                        
                        # Aplica a regra de agrupamento de classes (2 e 3 viram 0)
                        if class_id == 2 or class_id == 3:
                            class_id = 0
                        
                        partes[0] = str(class_id)
                        linhas_modificadas.append(" ".join(partes) + "\n")
                
                # Grava as anotações corrigidas e remapeadas na pasta de destino
                with open(label_file_destino, 'w') as f:
                    f.writelines(linhas_modificadas)
            else:
                # Se por acaso a imagem não tiver o label correspondente, cria um arquivo vazio (background)
                with open(label_file_destino, 'w') as f:
                    pass

    print(f"Labels corrigidos e salvos com sucesso em: {dest_dir}")

def yolo_to_coco(dataset_dir, subset, output_json_path):
    """
    Converte as anotações do formato YOLOv8 da pasta 'subset' para o formato COCO JSON
    exigido pela ferramenta de fatiamento do SAHI.
    
    Classes fixadas: 0 -> surface_objects, 1 -> vessels
    """
    dataset_path = Path(dataset_dir)
    train_images_dir = dataset_path / subset / 'images'
    train_labels_dir = dataset_path / subset / 'labels'
    
    # Estrutura base do arquivo COCO
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "surface_objects", "supercategory": "none"},
            {"id": 1, "name": "vessels", "supercategory": "none"}
        ]
    }
    
    image_id = 1
    annotation_id = 1
    
    # Lista todas as imagens do treino
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in train_images_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"Encontradas {len(image_files)} imagens no conjunto de treino. Iniciando conversão...")
    
    for img_path in image_files:
        # Abre a imagem para capturar a resolução exata (ex: 1920x1080)
        with Image.open(img_path) as img:
            width, height = img.size
            
        # Adiciona a imagem à estrutura COCO
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })
        
        # Procura o arquivo de label correspondente
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                        
                    class_id = int(parts[0])
                    # Coordenadas normalizadas do YOLO (centro_x, centro_y, largura, altura)
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    yolo_w = float(parts[3])
                    yolo_h = float(parts[4])
                    
                    # Converte de YOLO (normalizado) para COCO (pixels absolutos)
                    # COCO usa: x_min, y_min, largura, altura
                    coco_w = yolo_w * width
                    coco_h = yolo_h * height
                    x_min = (x_center * width) - (coco_w / 2)
                    y_min = (y_center * height) - (coco_h / 2)
                    
                    # Garante que as coordenadas fiquem dentro dos limites da imagem
                    x_min = max(0, min(x_min, width))
                    y_min = max(0, min(y_min, height))
                    coco_w = min(coco_w, width - x_min)
                    coco_h = min(coco_h, height - y_min)
                    
                    # Cálculo da área da bounding box
                    area = coco_w * coco_h
                    
                    # Adiciona a anotação à estrutura COCO
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [round(x_min, 2), round(y_min, 2), round(coco_w, 2), round(coco_h, 2)],
                        "area": round(area, 2),
                        "iscrowd": 0
                    })
                    annotation_id += 1
                    
        image_id += 1

    # Salva o arquivo JSON final
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"Sucesso! Arquivo COCO JSON gerado em: {output_json_path}")

def converter_sahi_json_para_yolo(path_json_sahi, pasta_imagens_sahi, pasta_destino_yolo):
    """
    Lê o JSON gerado pelo SAHI e exporta os labels diretamente 
    no formato YOLO (.txt) e copia as imagens fatiadas correspondentes.
    """
    path_json = Path(path_json_sahi)
    img_sahi_dir = Path(pasta_imagens_sahi)
    
    dest_dir = Path(pasta_destino_yolo)
    img_dest_dir = dest_dir / "images"
    label_dest_dir = dest_dir / "labels"
    
    # Cria as pastas finais de treino se não existirem
    img_dest_dir.mkdir(parents=True, exist_ok=True)
    label_dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Carrega o JSON fatiado pelo SAHI
    with open(path_json, 'r') as f:
        coco_data = json.load(f)
        
    # Mapeia as imagens por ID para facilitar a busca
    images_map = {img['id']: img for img in coco_data['images']}
    
    # Dicionário para agrupar as anotações de cada imagem
    annotations_per_image = {img_id: [] for img_id in images_map.keys()}
    
    print("Processando anotações do SAHI...")
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in images_map:
            continue
            
        img_info = images_map[img_id]
        dw = 1.0 / img_info['width']
        dh = 1.0 / img_info['height']
        
        # COCO usa: [x_min, y_min, width, height] em pixels absolutos
        x_min, y_min, w, h = ann['bbox']
        
        # Converte para YOLO: [x_center, y_center, width, height] normalizado
        x_center = (x_min + w / 2.0) * dw
        y_center = (y_min + h / 2.0) * dh
        yolo_w = w * dw
        yolo_h = h * dh
        
        class_id = ann['category_id']
        
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}\n"
        annotations_per_image[img_id].append(line)

    print("Gravando arquivos .txt e copiando as imagens fatiadas...")
    contador_arquivos = 0
    
    for img_id, lines in annotations_per_image.items():
        img_info = images_map[img_id]
        nome_arquivo_base = Path(img_info['file_name']).stem
        nome_imagem = img_info['file_name']
        
        # Caminhos de origem e destino
        img_origem = img_sahi_dir / nome_imagem
        img_destino = img_dest_dir / nome_imagem
        label_destino = label_dest_dir / f"{nome_arquivo_base}.txt"
        
        # 1. Copia a imagem fatiada correspondente
        if img_origem.exists():
            #shutil.copy2(img_origem, img_destino)
            
            # 2. Grava o arquivo .txt (mesmo se estiver vazio, criando imagem de background)
            with open(label_destino, 'w') as f:
                f.writelines(lines)
                
            contador_arquivos += 1
        else:
            print(f"Aviso: Imagem fatiada {nome_imagem} não encontrada na pasta do SAHI.")

    print(f"Concluído! {contador_arquivos} pares de imagens e labels salvos em: {pasta_destino_yolo}")


root_path = "/home/jonba/Documents/AuSSys/Wide2"
subset = "train"
json_path = f"{root_path}/{subset}_coco.json"
"""
yolo_to_coco(root_path, subset, json_path)  

slice_coco(
    image_dir=os.path.join(root_path,subset,"images"),
    coco_annotation_file_path=json_path,
    output_dir=os.path.join(root_path,"train_sahi"),
    output_coco_annotation_file_name="sahi_coco",
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
"""
caminho_do_json_sahi = f"{root_path}/sahi_coco.json" 
pasta_com_imagens_sahi = f"{root_path}/dataset_sahi/images"            
pasta_final_treino_yolo = f"{root_path}/train_sahi"        

converter_sahi_json_para_yolo(caminho_do_json_sahi, pasta_com_imagens_sahi, pasta_final_treino_yolo)
