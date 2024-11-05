# A reproducible 3D convolutional neural network with dual attention module (3D-DAM) for Alzheimer's disease classification

The journal version of the paper describing this work is available [here](https://doi.org/10.48550/arXiv.2310.12574)

## Abstract

Alzheimer's disease is one of the most common types of neurodegenerative disease, characterized by the accumulation of amyloid-beta plaque and tau tangles. Recently, deep learning approaches have shown promise in Alzheimer's disease diagnosis. In this study, we propose a reproducible model that utilizes a 3D convolutional neural network with a dual attention module for Alzheimer's disease classification. We trained the model in the ADNI database and verified the generalizability of our method in two independent datasets (AIBL and OASIS1). Our method achieved state-of-the-art classification performance, with an accuracy of 91.94% for MCI progression classification and 96.30% for Alzheimer's disease classification on the ADNI dataset. Furthermore, the model demonstrated good generalizability, achieving an accuracy of 86.37% on the AIBL dataset and 83.42% on the OASIS1 dataset. These results indicate that our proposed approach has competitive performance and generalizability when compared to recent studies in the field.

## Model Architecture
![model architecture](https://github.com/giaminhgist/3D-DAM/blob/main/photo/model.png)

## Main Results

### ADNI - AIBL - OASIS

| Training| Test| Accuracy(%) | Sensitivity(%) | Specificity(%) | 
|-------------|----------|-----------|--------|----------|
| ADNI |  AIBL | 86.3 | 80.2 | 87.1 |
| ADNI |  OASIS |  83.4 | 85.8 | 82.6 |
| AIBL - OASIS |  ADNI | 85.4 |80.1 | 89.5 |

![Test Performance](https://github.com/giaminhgist/3D-DAM/blob/main/photo/test_performance.png)

## Citation

If you find this project useful for your research, please use the following BibTeX entries.

    @misc{vu2024reproducible,
          title={A reproducible 3D convolutional neural network with dual attention module (3D-DAM) for Alzheimer's disease classification}, 
          year={2024},
          eprint={2310.12574},
          archivePrefix={arXiv},
          primaryClass={eess.IV}
    }
