# Deep Learning Final Project Report
## Point Cloud Human Identification and Point Cloud Compression

**Course**: MLDS 432 Deep Learning
**Institution**: Northwestern University

## 1. Abstract
This project studies two related tasks on 3D human point clouds derived from the FAUST dataset: subject classification and point-cloud autoencoding. The classification task predicts one of 10 identities from a single point cloud, while the autoencoder task compresses the same input into a latent representation and reconstructs it with low geometric error. We compare three classification models (`mlp`, `cnn1d`, `pointnet`) and two autoencoder models (`mlp_ae`, `pointnet_ae`), then package the workflow into a small platform with command-line tooling, a Flask backend, a browser GUI, and Docker support.

The current experimental results show a clear winner in both tasks. `pointnet` reaches 72.00% test accuracy and 0.9600 ROC-AUC on classification, substantially outperforming the MLP and CNN baselines. `pointnet_ae` achieves a mean Chamfer Distance of 0.004430, which is much lower than the MLP autoencoder's 0.015800. A centering ablation further shows that PointNet benefits strongly from centered inputs, while CNN1D is comparatively insensitive and MLP slightly improves without centering. Overall, the project demonstrates that point-cloud-aware architectures are much better suited than flatten-based baselines for both recognition and reconstruction.

## 2. Problem Motivation
Human identification from sparse 3D structure is an attractive alternative to camera-based recognition because it can preserve privacy and remain usable in low-light conditions. mmWave radar is one motivating sensor for this setting because it naturally produces sparse geometric observations rather than dense RGB images. In this repository, however, the actual training data comes from FAUST human-body meshes rather than real radar captures. We therefore treat FAUST as a controlled proxy dataset for point-cloud modeling rather than claiming that the current experiments are direct radar benchmarks.

This leads to two project goals:

1. Build a subject classification pipeline for unordered 3D point sets.
2. Build a compression and reconstruction pipeline that can preserve geometry in a low-dimensional latent representation.

The first goal measures discriminative performance. The second measures whether the same data representation can be encoded compactly without losing too much geometric information.

## 3. Dataset and Task Formulation
The project uses the FAUST training set, which contains 100 meshes corresponding to 10 subjects with 10 poses each. Each mesh is sampled into point clouds and augmented to produce a larger processed dataset. Under the current configuration, the repository uses:

- `num_points: 500`
- `samples_per_mesh: 100`
- `normalize_center: true`
- `normalize_scale: false`

This produces a processed cache in `data/processed/faust_pc.npz`. Because each source mesh creates many augmented samples, the split procedure uses grouped splitting to reduce leakage between train, validation, and test partitions. This is important because random splitting at the sample level would allow near-duplicate augmentations from the same source mesh to appear in multiple splits, inflating reported performance.

The two learning tasks are defined as follows:

- **Classification**: input a point cloud and predict one of 10 subject IDs.
- **Autoencoding**: input a point cloud, compress it into a latent vector, and reconstruct the original geometry with low Chamfer Distance.

## 4. Data Pipeline and Preprocessing
The preprocessing pipeline begins with raw FAUST mesh files stored in `data/raw/`. These meshes are converted into sampled point clouds using farthest point sampling and then saved into a processed `.npz` cache for repeated training runs. The current augmentation and preprocessing settings in `config.yaml` are:

```yaml
augmentation:
  normalize: true
  rotation_range: 360
  translation_range: 0.0

data:
  normalize_center: true
  normalize_scale: false
  num_points: 500
  samples_per_mesh: 100
```

These settings create an interesting configuration: point clouds are centered, random full-circle rotation is allowed, but scale normalization is turned off and translation jitter is disabled. This means the model still sees size information while benefiting from canonical centering. The later ablation study shows that centering is especially important for PointNet in this setup.

The repository also includes an exploratory notebook in `notebooks/eda.ipynb` with class balance checks, coordinate histograms, point-cloud visualizations, and outlier-oriented plots. The main takeaways are that the class distribution is balanced and that geometric spread varies across samples enough to make preprocessing decisions meaningful.

## 5. Model Architectures
### 5.1 Classification Models
The classification baseline is an MLP that flattens the full point cloud into a single vector before applying fully connected layers. This design is easy to train and useful for sanity checks, but it is fundamentally order-dependent and does not explicitly model point-set geometry.

The second model is a 1D CNN that treats the input as an ordered sequence of points and extracts local patterns with convolutional filters. It is stronger than the MLP baseline but still depends on the way points are represented and ordered.

The strongest classifier is a PointNet-style model (`pointnet`) with T-Net alignment, shared MLP feature extraction, and global max pooling. This architecture is better aligned with unordered point-cloud input because it aggregates features through a symmetric pooling operation and can learn transformations that stabilize the input geometry.

### 5.2 Autoencoder Models
The autoencoder baseline (`mlp_ae`) also flattens the point cloud and reconstructs it with fully connected layers. It is simple but inherits the same weakness as the classification MLP: the model has no built-in notion that the input is a set of points rather than a fixed-order vector.

The stronger autoencoder (`pointnet_ae`) uses a point-cloud-oriented encoder and reconstructs point sets with a geometry-aware objective. Reconstruction quality is evaluated with Chamfer Distance, which measures how closely the predicted and target point sets align in 3D space.

## 6. Experimental Setup
The classification and autoencoder pipelines are both configured through `config.yaml`. The current classification defaults are:

- batch size: 64
- epochs: 120
- learning rate: 0.001
- weight decay: 0.001
- dropout: 0.5
- device: `cuda`

The current autoencoder section includes separate sub-configurations for MLP and PointNet variants, plus dedicated training hyperparameters:

- AE batch size: 64
- AE learning rate: 0.001
- AE weight decay: 0.00001
- AE epochs: 120
- AE augment: false

Metrics used in the project are:

- **Classification**: accuracy, macro F1, precision, recall, ROC-AUC
- **Autoencoding**: Chamfer Distance and reconstruction summaries

Training and evaluation are available through both the CLI and the GUI. The CLI is useful for reproducible batch runs, while the GUI exposes preprocessing, model selection, live training curves, and artifact download through a browser interface.

## 7. Results: Classification
The current classification comparison from `results/model_comparison.csv` is shown below.

| Model | Test Accuracy (%) | F1 (%) | ROC-AUC | Precision (%) | Recall (%) | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 10.85 | 10.94 | 0.5296 | 11.33 | 10.85 | 419,210 |
| CNN1D | 34.80 | 33.90 | 0.7851 | 37.54 | 34.80 | 158,986 |
| PointNet | 72.00 | 72.04 | 0.9600 | 78.23 | 72.00 | 1,606,419 |

Several conclusions follow from these results.

First, PointNet is clearly the best classification model in the current repository. Its 72.00% accuracy and 0.9600 ROC-AUC indicate that the model can separate subjects far more reliably than either baseline. This supports the hypothesis that point-set-aware design is necessary for this task.

Second, CNN1D is a meaningful improvement over MLP, but it is still not competitive with PointNet. The convolutional model can capture stronger local structure than a flattened baseline, yet it remains sensitive to how the point cloud is arranged.

Third, the MLP performs near chance level for a 10-class problem. This makes it useful mainly as a diagnostic baseline rather than a practical model.

## 8. Results: Autoencoder
The current autoencoder comparison from `results/ae_comparison.md` is shown below.

| Model | CD Mean | CD Median | CD Std | CD Max | Parameters | Inference (s) | ms/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp_ae | 0.015800 | 0.014414 | 0.005920 | 0.049512 | 870,236 | 0.3628 | 0.18 |
| pointnet_ae | 0.004430 | 0.002753 | 0.003628 | 0.016544 | 506,115 | 0.4559 | 0.23 |

The point-cloud-aware autoencoder again wins by a wide margin. Its mean Chamfer Distance is roughly 3.6 times lower than the MLP baseline, indicating much better geometric reconstruction. Even though its inference time is slightly higher per sample, the trade-off is favorable because the reconstruction quality gain is substantial. Another useful observation is that `pointnet_ae` uses fewer parameters than `mlp_ae` in the current implementation, so the better performance is not simply explained by model size.

## 9. Ablation Study: Centering
The centering ablation in `results/experiments/summary_report.txt` compares with-centering and no-centering settings for the three classification models.

| Model | With Centering (%) | No Centering (%) | Difference |
| --- | ---: | ---: | ---: |
| MLP | 20.18 | 30.27 | -10.09 |
| CNN1D | 60.91 | 59.09 | +1.82 |
| PointNet | 71.73 | 54.55 | +17.18 |

This is one of the most informative findings in the entire project.

- PointNet benefits strongly from centering, improving by 17.18 percentage points.
- CNN1D changes very little, which suggests it is comparatively robust to this preprocessing choice.
- MLP slightly prefers no-centering in this experiment, which likely reflects its inability to exploit geometric normalization in a meaningful way.

The practical implication is that preprocessing is not model-agnostic. A data transformation that helps one architecture can be neutral or even harmful for another. For this repository, centering should be considered part of the recommended PointNet pipeline rather than an optional cosmetic step.

## 10. Platform and Operations
In addition to model experiments, the repository now functions as a small point-cloud ML platform. The backend is implemented with Flask, the frontend is a lightweight HTML/JavaScript interface, and Docker support allows the full stack to be launched with `bash start.sh`.

The end-to-end platform flow is:

1. Upload or place FAUST mesh files in `data/raw/`.
2. Preprocess them into sampled point clouds.
3. Train either a classification model or an autoencoder.
4. Monitor loss and accuracy curves in the GUI.
5. Generate evaluation reports and download artifacts.

This platformization matters for the course deliverable because it turns the project from a collection of training scripts into a reproducible workflow that can be demonstrated, rerun, and extended. It also provides a natural path for future sensor integration: the FAUST preprocessing stage could be replaced with a real radar point-cloud loader while leaving most of the downstream training and evaluation logic intact.

## 11. Limitations
The most important limitation is domain mismatch. Although the project motivation is privacy-preserving sensing inspired by mmWave radar, the actual experiments use FAUST meshes rather than real radar point clouds. FAUST provides clean geometry but does not capture real radar noise, sparsity patterns, occlusion behavior, or temporal dynamics.

Additional limitations include:

- the dataset is relatively small
- the experiments focus on static point clouds rather than sequences
- the GUI generates JSON reports rather than polished PDF output
- some performance claims in earlier drafts had to be corrected once the latest result files were used as the single source of truth

These limitations do not invalidate the project, but they do define the correct interpretation of the results: this is currently a strong point-cloud learning prototype, not a finished real-world radar deployment.

## 12. Conclusion and Future Work
This project successfully addressed two tasks on human point clouds: identification and reconstruction. In both cases, point-cloud-aware architectures clearly outperform flatten-based baselines. `pointnet` is the strongest classifier at 72.00% accuracy, while `pointnet_ae` is the strongest autoencoder with a mean Chamfer Distance of 0.004430. The centering ablation further shows that preprocessing decisions materially affect downstream performance, especially for PointNet.

Beyond the modeling results, the repository now includes a usable platform with CLI scripts, a GUI, a Flask backend, Docker deployment, experiment automation, and organized result outputs. The next major step is to move from FAUST-as-proxy toward real sparse sensing data, ideally with temporal modeling for motion-aware identification and richer evaluation under domain noise.

## 13. References
1. Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. *CVPR*.
2. Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. *NeurIPS*.
3. Bogo, F., et al. (2014). FAUST: Dataset and Evaluation for 3D Mesh Registration. *CVPR*.
4. PyTorch Documentation: https://pytorch.org/docs/
5. FAUST Dataset: http://faust.is.tue.mpg.de/
