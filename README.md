# Detecting and mitigating issues in image-based COVID-19 diagnosis

Machine Learning (ML) has become a foremost approach for solving real-world problems from data. Supervised learning, in particular, has risen to be a popular method for building models that learn to classify inputs from a set of examples. When developing these models, the overarching goal is to learn a function that can generalize to examples outside of those provided during training. Indeed, fitting a model to a set of training samples is often trivial, in particular with high capacity models such as deep neural networks. However, generalizing to instances outside of training is a much more challenging problem.

In settings where robustness to generalization is critical, such as medical and healthcare studies, black-box models that fail to generalize are particularly problematic and forbid practical utility. ML researchers have explored several approaches for making the models more reliable. [Yang et al., 2021] provides a literature review that describes how explainable AI can aid in developing ML models that work in the real world. For the task of COVID-19 image-based detection specifically, [Ter-Sarkisov, 2020] takes a two step approach that first detects instances of ground glass opacity and consolidation in CT scans, then predicts the patient’s condition using only the ranked bounding box detections (thus making the model able to generalize even with a very small amount of training data). [Teixeira et al., 2021] uses a U-Net CNN architecture to perform semantic segmentation that isolates the lungs on CXR scans before trying to classify whether the patient has COVID-19, but still finds that a strong bias from underlying data source issues severely hampers results. Our work investigates current issues with some of the most popular datasets used for COVID-19 diagnosis from CXR images in an attempt to more closely understand the problems faced by researchers when dealing with those datasets, and how to mitigate some of them.

<img src="https://i.imgur.com/B8yLDXA.png" alt="drawing" width="300"/>

*Samples of chest X-Rays taken from the datasets used in this paper. Each image is labeled COVID-19 positive or not. Some datasets present extra classes, such as viral and bacterial pneumonia.*

<img src="https://i.imgur.com/FoZVpQH.png" alt="drawing" width="400"/>

*Distribution of maximum similarities between the images of the COVID-19 Radiography Database using pixel-wise similarity. The purple dashed line is the threshold chosen by our algorithm to select the images to be removed. Only pairs with maximum similarities close to 1 are shown in the graph.*

<img src="https://i.imgur.com/UIf4hCi.png" alt="drawing" width="600"/>

*Model accuracies on the COVID-19 Radiography Database for different amounts of training samples available (proportional to the whole target training dataset), using both random intializations and pre-training on each of the other two datasets. The shaded areas indicate a range of one standard deviation, estimated with 5-fold cross-validation.*


## Abstract

> As urgency over the coronavirus disease 2019 (COVID-19) increased, many datasets with chest radiography (CXR) and chest computed tomography (CT) images emerged aiming at the detection and prognostication of COVID-19. Over the last two years, thousands of studies have been published, reporting promising results. However, a deeper analysis of the datasets and the methods employed reveal issues that may hamper conclu-
sions and practical applicability. We investigate three major datasets commonly used in these studies, detect problems related to the existence of duplicates, address the specificity of classes within those datasets, and propose a way to perform external validation via cross-dataset evaluation. Our
guidelines and findings contribute towards a trustworthy application of machine learning in the context of image-based diagnosis, as well as offer a
more accurate assessment of models applied to the prognostication of diseases using image datasets and pave the way towards models that can be relied upon in the real world.


## Software implementation

All source code used to generate the results and figures in the paper are contained in this repository. The code is built on top of DeepMind's Haiku and Google's JAX (see [Reproducing the results](#reproducing-the-results) for details on how to setup).

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/JoaoMarcosCSilva/xray_duplicates_detector.git

or [download a zip archive](https://github.com/JoaoMarcosCSilva/xray_duplicates_detector/archive/refs/heads/main.zip)

## Reproducing the results

All experiments were run on Google TPU v3 VMs. It is recommended that the experiments are reproduced on a similar hardware.

To download the datasets and necessary libraries run setup.sh:

    chmod +x setup.sh
    ./setup.sh

To run, edit or create a JSON run setting in runs/. Then, execute the following command:

    run.py PATH_TO_JSON_FILE.json [[your run parameters]]

The specific JSONs and commands used in the paper are included as examples in the runs folder (see [Run parameters](#run-parameters) for details on the meaning of each parameter).

## Run parameters

Apart from the JSON-specified settings, we also provide a set of command line arguments.
|  Command | Description  |
|---|---|
| --save-only-first | Set this to true to save the weights to W&B. |
| --wandb | Whether or not to log the experiment to weights and biases. |
| --save | Whether or not to save the trained weights.  |
|  --load | Path of a pickle file containing pretrained weights. |
| --name | Name of the run. If undefined, will create a random hash of the configuration. |
| -f/--force | If this is set, any existing weights with the same name will be removed |
| --cv | The number of cross-validation folds to use. If unset, doesn't use cross-validation. |
| --cv-id | The cross validation fold id to use |
| --dedup | If set, will run the dataset deduplication procedure. |
| --save-dedup | If set, will save the deduplicated dataset to the specified directory. |
| --split-dedup | If set, the deduplicated dataset will be split into train and test sets. |
| --pixel-space | If set, will run the deduplication in pixel space. This option also skips training as a whole. |

## Acknowledgments

* Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC).
* Weights & Biases for providing free accounts for students.

## Authors

João Marcos Cardoso da Silva (@JoaoMarcosCSilva)

Pedro Martelleto (@PedroMartelleto)

We would like to specially thank Moacir A. Ponti (@maponti) for helping us and supervising the whole process.

## License

All source code is made available under a BSD 3-clause license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE.md` for the full license text. The manuscript text is not open source. The authors reserve the rights to the
article content.
