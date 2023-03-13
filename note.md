files in code folder:
    DTC_technology.py: some preprocessing steps applied during interference. ony normalization is important
    *_backup: original files by Ngai Synh
    shape.py: pipeline from raw trainset -> classified image into categories (box, abstract and unknown)
    color2HSV.py: color investigation
    clean.sh: clean dataset/classify directory for fresh run of shape.py

Note on dataset directory stucture: 
-   parent_folder:
  - code
    - [codes and instruction here]
    - DTC_aug [some preprocessing steps applied during interference. Not very important]
  - dataset:
    - classify [contains data of categorical items (box, abstract and unknown)]
      - box
        - label [the segmentation label]
        - seg [the rbg image of the object]
      - abstract [same structure as box]
      - unknown [same structure as box]
    - train [original train set by AIC]
    - segmentation_labels [original train set by AIC]
    - testA [original train set by AIC23]
    - test_a_old [original train set by AIC22 (last year)]
    - backup [backup of classify folder]
    - sample_dataset_team_dtc [sample dataset by DTC, as a reference for Quality control]
    - 

Todo:
- Investigate a suitable colorspace and conversion (bach)
- Cut obj faces (an)
- Remove/denoise noisy obj faces
  - Histogram based
  - Colorspace based (convert to HSV, HSL, Luv, Lab..)
  - Hough lines
  - Normalization
  - Convolution (MSRCR)
- Steal MSRCR technology from DTC team (sieu)
- Human hand object augmentation (arc-shaped, box-shaped cuts into objects)
- Object background denoise/lighting adjustment (same for interfere pipeline)
- Assemble objects into background to make train/test set for yolov8
- High/medium object density augmentation to cope with tests where a lot of objects stacked together on the white plate (important on hidden test)
- Low object density augmentation to cope with test where object are gracefully passed through the white plate
- Consideration for interfere speed