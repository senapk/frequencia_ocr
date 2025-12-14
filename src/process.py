import argparse
from image import Image
from align import Align
from border import Border
from ocr import OCR
from cache_db import CacheDb
from sheet_cutter import SheetCutter
import os

from region import Decomposer
import datetime as dt
from color import Color

MODEL_SIZE = 20
WIDTH_DICT: dict[int, tuple[int, int]] = {
    20: (1527, 1140),
    28: (2054, 1560)
}   
HEIGHT: int = WIDTH_DICT[MODEL_SIZE][0]
WIDTH: int = WIDTH_DICT[MODEL_SIZE][1]
raw_cache_db_folder = "cache_digits_raw"
bordered_cache_db_folder = "cache_digits_processed"
cells_folder = "cache_temp"


# retorna a lista de listas de caminhos das células recortadas
# retorna a lista com as células recortadas

def load_info_sheet(path: str) -> dict[int, tuple[str, str]]:
    import json
    output: dict[int, tuple[str, str]] = {}
    path = path.replace(".png", ".json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            output = {int(key): (value[0], value[1]) for key, value in data.items()}
    return output


def main():
    parser = argparse.ArgumentParser(description="Alinhar imagem usando marcadores ArUco.")
    parser.add_argument("--input", '-i', default="folhas/m2-1.png", help="Caminho para a imagem de entrada.")
    parser.add_argument("--sheet_model", '-m', type=int, default=2, help="Modelo da folha (1 ou 2).")
    parser.add_argument("--align", '-a', default="folhas/alinhada.png", help="Caminho para salvar a imagem alinhada.")
    parser.add_argument("--filter", '-f', default="folhas/filtrada.png", help="Caminho para salvar a imagem filtrada.")
    parser.add_argument("--qtd", type=int, default=0, help="Quantidade de linhas a processar (0 = todas).")
    parser.add_argument("--zoom", "-z", type=float, default=1)
    args = parser.parse_args()


    info = load_info_sheet(args.input)
    Image.preview_zoom = args.zoom
    aligner = Align(WIDTH, HEIGHT)
    image = Image().load_from_file(args.input)

    image_aligned = Image().set_data(aligner.aruco(image.data))
    image_aligned.save_to_file(args.align)

    # filtering = Filtering()
    # img = filtering.otsu(img, args.filter)

    sheet_cutter = SheetCutter(cells_folder)
    model = SheetCutter.Model.M1 if args.sheet_model == 1 else SheetCutter.Model.M2
    images, _ = sheet_cutter.cut_info(image_aligned, model)

    raw_cache_db = CacheDb(raw_cache_db_folder).load()
    # bordered_cache_db = CacheDb(bordered_cache_db_folder).load()
    init_time = dt.datetime.now()
    ocr_digit = OCR().set_debug(False)
    print("Loading time:", dt.datetime.now() - init_time)

    count_success = 0
    total = 0
    for i, row in enumerate(images):
        if args.qtd > 0 and i >= args.qtd:
            break
        print(f"{i:02}:", end="")
        for j, image_origin in enumerate(row):
            raw_cache_db.store_image(image_origin)
            # print(image)
            image = image_origin.binarize(gaussian=False).inversion().cut_borders(2)
            # print(image.get_h_w())
            #bordered_before_paddding: ImageFilter = Border(image).dynamic_cutter_while_visible_border().dynamic_bounding_box()
            filtered = Decomposer(image)
            # origin_size = len(filtered.regions)
            filtered = filtered.erase_small_regions(0.2).erase_regions_outside_bounds()
            bordered = Border(filtered.get_image()).dynamic_bounding_box().centralize_and_pad(MODEL_SIZE * 2)
            # big_size = len(filtered.regions)
            # print(image_origin, image, filtered.get_image())
            # if bordered.get_image().is_empty(): # check empty before padding
            #     continue
            # bordered_padded = bordered_before_paddding.centralize_and_pad()
            # bordered_cache_db.store_image(bordered_padded.get_image())
            # write_ratio = filtered.get_image().written_ratio()
            digit, _ = ocr_digit.predict_from_filtered_image(bordered)
            # h, w = filtered.get_image().get_dim()
            digit_str = f"{digit}"
            num = i + 1
            if num in info:
                total += 1
                mat, _ = info[num]
                if mat[j] == digit_str:
                    digit_str = Color.green(digit_str)
                    count_success += 1
                else:
                    digit_str = Color.red(digit_str)
            print(f"{image_origin}{bordered.get_image()}{digit_str}, ", end="")
        print("")

    print(f"Accuracy: {count_success} / {total} = {count_success / total * 100:.0f}%")

if __name__ == "__main__":
    main()