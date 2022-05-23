# camera-calibration



Tato aplikace byla vytvořena v rámci bakálářské práce. Jejím cílem je pomocí vhodných metod provádět kalibraci obrazu, konkrétně vracet hodnoty poměru milimetry/pixel.



![frame_001810](https://user-images.githubusercontent.com/75748212/169849021-9082a9d8-149d-4ba0-878d-d007c500e2f9.png)

Umožňuje kalibraci pomocí QR kódu, neuronových sítí a Houghovy transformace.

## Požadavky před spuštěním:
- Python 3.8
- Numpy
- Detectron 2
- CV2

Pro její spuštění je potřeba mít nainstalovaný Detectron 2 a veškeré importované balíky.

##  Příklad použití
```
    if __name__ == '__main__':

        start_time=time.perf_counter()
        register_coco_instances("pig_leg_surgery", {}, Path(f"C:\\Users\\Juryx\Desktop\\annotations\instances_default.json"), Path(f""))
        pig_leg_surgery_metadata = MetadataCatalog.get("pig_leg_surgery")
        dataset_dicts = DatasetCatalog.get("pig_leg_surgery")
        cfg = get_cfg()
        configure_network()
        output=calibrate(cv2.imread(f"PATH_TO_FILE"))
``` 
Metoda calibrate() má za vstup obrázek ve formátu matice. Také lze nastavit hodnoty rozměrů reálného objektu.

```
def calibrate(image:np.ndarray,real_sideAC_mm:int=320,real_sideBD_mm:int=135,real_diag_mm:int=350,real_r_mm:float=5.5)->np.ndarray:

    
    #show_predictions(image)
    corners,pred_boxes,masks=find_corners(image)
    metrics,radiuses_NN=measure_table(pred_boxes,image,masks)
    radiuses_HT=hough_transform_find_holes(corners)
    pix_size_QR=main_qr(image)
    
```
## Zvolení modelu

Předtrénované modely neuronové sítě jsou uloženy zde: https://drive.google.com/drive/u/0/folders/1RJDIW6De3LLQCrutjxjieI2tllJzfJ9D

Pro kalibraci pomocí neuronové sítě a měření kruhových výřezů je potřeba použít modely (mode_name)_holes.pth

V metodě configure_network() se zvolí model. Zde je ještě třeba zvolit i správné nastavení vah.
```
def configure_network():
    #for rcnn R50
    cfg.merge_from_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #for rcnn R101
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "PATH_TO_MODEL")
```
