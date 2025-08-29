[stars-img]: https://img.shields.io/github/stars/zzhangsm/Awesome-Histopathology-Generative-Translation?color=yellow
[stars-url]: https://github.com/zzhangsm/Awesome-Histopathology-Generative-Translation/stargazers
[fork-img]: https://img.shields.io/github/forks/zzhangsm/Awesome-Histopathology-Generative-Translation?color=lightblue&label=fork
[fork-url]: https://github.com/zzhangsm/Awesome-Histopathology-Generative-Translation/network/members
[AHBG-url]: https://github.com/zzhangsm/Awesome-Histopathology-Generative-Translation

# Awesome-Histopathology-Generative-Translation
Awesome-Histopathology-Generative-Translation is a collection of histopathology image-based generation works, including papers, codes and datasets🔥.  We center on the core task of "generative translation for histopathology data" — a key research direction in computational pathology that aims to bridge different data modalities or staining types using generative models. Specifically, it focuses on translating information from hematoxylin and eosin (H&E) stained histopathology images (the most common and accessible histology imaging modality) to other critical biological or imaging data.

Any problems, please contact the zzhangsm615@gmail.com. Any other interesting papers or codes are welcome. If you find this repository useful to your research or work, it is really appreciated to star this repository.

[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]


## 📋 Table of Contents

- [📊 Datasets](#-datasets)
- [📚 Papers](#-papers)
  - [Survey](#survey)
  - [H&E To Spatial Transcriptomics](#he-to-spatial-transcriptomics)
  - [H&E To Protein Biomarker](#he-to-protein-biomarker)
- [🤝 Contributing](#-contributing)
- [📧 Contact](#-contact)
- [🙏 Acknowledgments](#-acknowledgments)

## 📊 Datasets

### Public Datasets
| Year | Title                                                        |  Venue  |        Built For        |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :-------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2023 | **An AI-ready multiplex staining dataset for reproducible and accurate characterization of tumor immune microenvironment** | MICCAI | H&E To Protein | [link](https://arxiv.org/abs/2305.16465)    | [link](https://github.com/nadeemlab/DeepLIIF)                   |

## 📚 Papers
### Survey
| Year | Title                                                        |  Venue  |        Method        |                            Paper                             |
| ---- | ------------------------------------------------------------ | :-----: | :-------------------: | :----------------------------------------------------------: |
| 2025 | **Generative Models in Computational Pathology: A Comprehensive Survey on Methods, Applications, and Challenges** | arXiv |  Computational Pathology | [link](https://arxiv.org/abs/2505.10993)    |
| 2022 | **Deep learning-based prediction of molecular tumor biomarkers from H&E: A practical review** | J. Pers. Med | H&E To Protein | [link](https://www.mdpi.com/2075-4426/12/12/2022)    |

### H&E To Spatial Transcriptomics

| Year | Title                                                        |  Venue  |        Method        |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :-------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025 | **A visual–omics foundation model to bridge histopathology with spatial transcriptomics** | Nat. Methods | OmiCLIP | [link](https://www.nature.com/articles/s41592-025-02707-1)    | [link](https://github.com/GuangyuWangLab2021/Loki)  
| 2025 | **Predicting Spatial Transcriptomics from H&E Image by Pretrained Contrastive Alignment Learning** | bioRxiv | CarHE | [link](https://www.biorxiv.org/content/10.1101/2025.06.15.659438v1.abstract)    | [link](https://github.com/Jwzouchenlab/CarHE) |
| 2025 | **Gene-DML: Dual-Pathway Multi-Level Discrimination for Gene Expression Prediction from Histopathology Images** | arXiv | Gene-DML | [link](https://arxiv.org/abs/2507.14670)    | [link](https://github.com/hrlblab/Img2ST-Net) |
| 2025 | **Img2ST-Net: Efficient High-Resolution Spatial Omics Prediction from Whole Slide Histology Images via Fully Convolutional Image-to-Image Learning** | arXiv | Img2ST-Net | [link](https://arxiv.org/abs/2508.14393)    | [link](https://github.com/hrlblab/Img2ST-Net) |
| 2025 | **MV-Hybrid: Improving Spatial Transcriptomics Prediction with Hybrid State Space-Vision Transformer Backbone in Pathology Vision Foundation Models** | MICCAI Workshop | MV-Hybrid | [link](https://arxiv.org/abs/2508.00383)    | [link](https://github.com/deepnoid-ai/MVHybrid) |
| 2025 | **Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images** | ICLR | Stem | [link](https://arxiv.org/abs/2501.15598)    | [link](https://github.com/SichenZhu/Stem)                   |
| 2025 | **Spatially resolved gene expression prediction from histology images via bi-modal contrastive learning** | NeurIPS | BLEEP | [link](https://arxiv.org/pdf/2306.01859)    | [link](https://github.com/bowang-lab/BLEEP)                   |
| 2025 | **M2ORT: Many-To-One Regression Transformer for Spatial Transcriptomics Prediction from Histopathology Images** | AAAI | M2ORT | [link](https://ojs.aaai.org/index.php/AAAI/article/view/32830/34985)    | [link](https://github.com/Dootmaan/M2ORT/) |
| 2024| **HEMIT: H&E to Multiplex-immunohistochemistry Image Translation with Dual-Branch Pix2pix Generator** | arXiv | HEMIT | [link](https://arxiv.org/abs/2403.18501)    | [link](https://github.com/BianChang/HEMIT-DATASET) |
| 2024 | **Accurate spatial gene expression prediction by integrating multi-resolution features** | CVPR | TRIPLEX | [link](https://openaccess.thecvf.com/content/CVPR2024/html/Chung_Accurate_Spatial_Gene_Expression_Prediction_by_Integrating_Multi-Resolution_Features_CVPR_2024_paper.html)    |   [link](https://github.com/NEXGEM/TRIPLEX)                   |
| 2023 | **Exemplar guided deep neural network for spatial transcriptomics analysis of gene expression prediction** | CVPR | EGN                  | [link](https://arxiv.org/abs/2210.16721)    | [link](https://github.com/Yan98/EGN)                   |
| 2022 | **Spatial transcriptomics prediction from histology jointly through transformer and graph neural networks** | BIB | Hist2ST                  | [link](https://doi.org/10.1093/bib/bbac297)    | [link](https://github.com/biomed-AI/Hist2ST)                   |
| 2021 | **Leveraging information in spatial transcriptomics to predict super-resolution gene expression from histology images in tumors** | bioRxiv | HisToGene                  | [link](https://www.biorxiv.org/content/10.1101/2021.11.28.470212v1)    | [link](https://github.com/maxpmx/HisToGene)                   |
| 2020 | **Integrating spatial gene expression and breast tumour morphology via deep learning** | Nature biomedical engineering | ST-Net                  | [link](https://www.nature.com/articles/s41551-020-0578-x)    | [link](https://github.com/bryanhe/ST-Net)                   |

### H&E To Protein Biomarker

| Year | Title                                                        |  Venue  |        Method        |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :-------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025 | **ROSIE: AI generation of multiplex immunofluorescence staining from histopathology images** | Nat. Commun. | ROSIE | [link](https://www.nature.com/articles/s41467-025-62346-0)    | [link](https://gitlab.com/enable-medicine-public/rosie) |
| 2025 | **Histopathology-based protein multiplex generation using deep learning** | Nat. Mach. Intell. | HistoPlexer | [link](https://doi.org/10.1038/s42256-025-01074-y)    | [link](https://github.com/ratschlab/HistoPlexer) |
| 2024 | **Virtual multiplexed immunofluorescence staining from non-antibody-stained fluorescence imaging for gastric cancer prognosis** | EBioMedicine | MAS | [link](https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(24)00323-2/fulltext)    |    [link](https://github.com/Eva0720/MAS) |
| 2024 | **Accelerating histopathology workflows with generative AI-based virtually multiplexed tumour profiling** | Nat. Mach. Intell. | VirtualMultiplexer | [link](https://www.nature.com/articles/s42256-024-00889-5)    | [link](https://github.com/AI4SCR/VirtualMultiplexer)     
| 2023 | **7-UP: Generating in silico CODEX from a small set of immunofluorescence markers** |  PNAS Nexus | 7-UP | [link](https://doi.org/10.1093/pnasnexus/pgad171)    | [link](https://gitlab.com/enable-medicine-public/7-up)                |
| 2023 | **Unpaired stain transfer using pathology-consistent constrained generative adversarial networks** |  IEEE Trans. Med. imaging | PC-StainGAN | [link](https://ieeexplore.ieee.org/abstract/document/9389763)    | [link]( https://github.com/fightingkitty/PC-StainGAN)                |
| 2022 | **Tackling stain variability using CycleGAN-based stain augmentation** | J. Pathol. Inf.  | StainAugm | [link](https://doi.org/10.1016/j.jpi.2022.100140)    | [link](https://github.com/NBouteldja/KidneyStainAugmentation)  
| 2022 | **Deep Learning-Inferred Multiplex ImmunoFluorescence for Immunohistochemical Image Quantification** | Nat. Mach. Intell. | DeepLIIF | [link](https://arxiv.org/pdf/2204.11425)    | [link](https://github.com/nadeemlab/DeepLIIF)   
| 2022 | **Deep learning-based restaining of histopathological images** | BIBM | StainGAN | [link](https://ieeexplore.ieee.org/document/9994934)    |   
| 2022 | **BCI: breast cancer immunohistochemical image generation through pyramid pix2pix** | CVPR Workshop| Pix2pix | [link](https://www.nature.com/articles/s42256-022-00471-x)    | [link](https://bupt-ai-cz.github.io/BCI)   
| 2022 | **MVFStain: Multiple virtual functional stain histopathology images generation based on specific domain mapping** | Med. Image Anal. | Multi-V-Stain | [link](https://doi.org/10.1016/j.media.2022.102520)    |    
| 2021 | **Deep learning-based transformation of H&E stained tissues into special stains** |  Nat. Commun. |  | [link](https://www.nature.com/articles/s41467-021-25221-2)    | [link](https://github.com/kevindehaan/stain-transformation) |
| 2020 | **SHIFT: speedy histological-to-immunofluorescent translation of a tumor signature enabled by deep learning** |Sci. Rep. | SHIFT | [link](https://www.nature.com/articles/s41598-020-74500-3)    | [link](https://gitlab.com/eburling/shift)                   |


## 🤝 Contributing

We welcome all forms of contributions! If you want to add new papers, codes, or resources, please:

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📧 Contact

If you have any questions or suggestions, please contact us through:
- Email: zzhangsm615@gmail.com
- GitHub Issues: [Create Issue](https://github.com/zzhangsm/Awesome-Histopathology-Generative-Translation/issues)

## 🙏 Acknowledgments

Thanks to all researchers and developers who have contributed to the field of histopathology image generation.

---

⭐ If you find this project useful, please give us a star!
