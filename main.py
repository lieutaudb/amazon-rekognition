from pathlib import Path
import boto3
from mypy_boto3_rekognition.type_defs import (
    CelebrityTypeDef,
    RecognizeCelebritiesResponseTypeDef,
)
from PIL import Image, ImageDraw, ImageFont

rekognition_client = boto3.client("rekognition")

# Função para obter o caminho completo de um arquivo na pasta 'assets'
def get_file_path(file_name: str, folder: str = "assets") -> str:
    return str(Path(__file__).resolve().parent / folder / file_name)

# Função para reconhecer celebridades em uma imagem usando o Amazon Rekognition
def detect_celebrities(image_path: str) -> RecognizeCelebritiesResponseTypeDef:
    with open(image_path, "rb") as img_file:
        return rekognition_client.recognize_celebrities(Image={"Bytes": img_file.read()})

# Função para desenhar caixas delimitadoras e informações na imagem
def annotate_image(
    input_image: str, 
    output_image: str, 
    celebrity_faces: list[CelebrityTypeDef],
    font_path: str = "Ubuntu-R.ttf"
):
    image = Image.open(input_image)
    drawer = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, size=18)
    except IOError:
        print(f"Fonte '{font_path}' não encontrada. Usando fonte padrão.")
        font = ImageFont.load_default()

    img_width, img_height = image.size

    for celeb in celebrity_faces:
        bbox = celeb["Face"]["BoundingBox"]
        left = int(bbox["Left"] * img_width)
        top = int(bbox["Top"] * img_height)
        right = int((bbox["Left"] + bbox["Width"]) * img_width)
        bottom = int((bbox["Top"] + bbox["Height"]) * img_height)

        confidence = celeb.get("MatchConfidence", 0)
        if confidence >= 85:  # Definindo um limiar de confiança
            drawer.rectangle([left, top, right, bottom], outline="green", width=2)

            name = celeb.get("Name", "Desconhecido")
            text_pos = (left, max(0, top - 15))
            text_box = drawer.textbbox(text_pos, name, font=font)
            drawer.rectangle(text_box, fill="green")
            drawer.text(text_pos, name, font=font, fill="white")

    image.save(output_image)
    print(f"Resultados salvos em: {output_image}")

if __name__ == "__main__":
    image_files = ["ptv.jpeg"]
    for img_file in image_files:
        full_path = get_file_path(img_file)
        output_file = get_file_path(f"{Path(img_file).stem}_annotated.jpg")

        try:
            detection_result = detect_celebrities(full_path)
            celebrities = detection_result.get("CelebrityFaces", [])
            if celebrities:
                annotate_image(full_path, output_file, celebrities)
            else:
                print(f"Nenhuma celebridade detectada em: {img_file}")
        except Exception as e:
            print(f"Erro ao processar {img_file}: {e}")