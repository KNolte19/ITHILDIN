
<p align="left">
  <img src="./static/images/title_page.jpg" width="1200">
</p>

**Automated wing morphometric analysis and species identification for insects**

ITHILDIN is a deep learning-based pipeline for analyzing insect wing images, with a focus on mosquitoes and related Diptera. The system performs automated segmentation, landmark detection, species classification, and geometric morphometric analysis.

## Abstract
Diptera represent one of the most diverse insect orders, including species with essential ecological functions but also important vectors of human and animal pathogens. Their accurate species identification remains a major bottleneck in ecological and epidemiological studies. Morphological identification requires taxonomic expertise, while molecular methods are costly and not universally reliable. Wing geometric morphometrics offers an alternative, but manual landmark annotation is time-consuming and introduces observer bias. We developed ITHILDIN, a fully automated pipeline for landmark and semilandmark annotation of Diptera wings, combining UNet++ segmentation and an Hourglass landmark prediction model. Using mosquitoes as the primary model system, models were trained on 5,991 annotated images and evaluated on 12,522 images across 34 taxa. We assessed landmark annotation accuracy against human observers and ML-Morph, evaluated species identification performance using Linear Discriminant Analysis on 17 homologous landmarks and 52 semilandmarks, and tested out-of-distribution generalisation by reproducing an independent study. Transferability was demonstrated by adapting the pipeline to Drosophilidae and Glossinidae. The Hourglass model achieved a mean landmark error of 4.5 pixels (95% CI: 4.3–4.6), within human observer variability (4.7 pixels, 95% CI: 4.4–5.0) and substantially outperforming ML-Morph (12.7 pixels, 95% CI: 11.1–14.2). The semilandmark-based approach for species identification achieved 91% balanced accuracy across 34 taxa, approaching CNN performance (94%). On out-of-distribution data, the landmark pipeline generalised substantially better than the CNN and a soft-voting ensemble of the landmark and CNN classifiers achieved 88% balanced accuracy on a replicated study. Combining geometric morphometrics with deep learning provides a reproducible, interpretable, and generalisable alternative to black-box CNN classifiers for Diptera wing analysis. By acting as a consistent single observer, the system eliminates inter-observer bias, enabling large-scale and cross-study morphometric analyses of Dipteran wings. The system is publicly available at ithildin.bnitm.de and transferable to other Diptera families with moderate retraining effort.

# Pipeline Overview

<p align="left">
  <img src="./static/images/workflow.jpg" width="1200">
</p>

IHTILDIN is a pipeline of processing steps and deep learning model predictions. It transforms a wing image into (semi)landmark coordinates and predicts the species. First, the image undergoes a subset of the preprocessing steps proposed by Nolte, Baumbach, et al. (2025). Here the background is removed, the wing horizontally aligned, and image features that could serve as spurious correlations are reduced using Contrast-Limited Adaptive Histogram Equalisation. Next, the processed wing image is segmented using an UNet++ model (Zhou et al. 2018). The segmentation and processed image are concatenated and passed into an Hourglass model, which predicts the landmark positions (Wu et al. 2018). Using the predicted landmarks and segmentation, a digital representation of the wing is constructed. This representation consists of landmarks as points and vein segments as paths. Semilandmarks are allocated at equidistant intervals along these veins. Additionally, a CNN trained according to Nolte, Baumbach et al. (2025), predicts the species based on the processed image. A decision module combines the CNN-based prediction with a landmark-based prediction from LDA to produce the final taxonomic classification. 

## Installation

### Requirements

- Python 3.8+
- R 4.0+ with geomorph (for geometric morphometrics)
- PyTorch 2.0+
- Docker (optional, recommended for production deployment)

## Usage

### Docker Deployment (Recommended for just running the application)

The easiest way to deploy ITHILDIN on a server is using Docker:

```bash
# Build and start the container
docker compose up -d

# Access at http://127.0.0.1:8080
```

See [DOCKER.md](DOCKER.md) for complete Docker deployment guide including production configuration, troubleshooting, and security considerations.

### Manual Setup (Recommended for development and training)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KNolte19/ITHILDIN.git
   cd ITHILDIN
   ```

2. **Install Python dependencies**:
Before installing, we recommend creating a virtual environment to avoid conflicts with existing packages:
   ```bash
   python -m venv ithildin_env
   source ithildin_env/bin/activate  # On Windows: ithildin_env\Scripts\activate
   ```

Then you can install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install R dependencies** (for Procrustes analysis):
   ```R
   install.packages(c("geomorph", "shapes"))
   ```

4. **Configure settings**:
   - Edit `config_loader.py` to set your `root_path` and `device` (cpu/cuda/mps)


5. **Start the Application**:

Start the Flask application for interactive batch processing:

```bash
python app.py
```

Then open your browser to `http://localhost:5000`

## Project Structure

```
ithildin_wing_analysis/
├── analysis/              # Morphometric analysis and LDA
│   ├── geomorph.py       # R integration for geometric morphometrics
│   └── landmark_analysis.py
├── predictor/            # Neural network models
│   ├── classification.py
│   ├── landmark.py
│   ├── prediction.py
│   └── segmentation.py
├── transform/            # Image and landmark processing
│   ├── image_processing.py
│   ├── landmark_processing.py
│   ├── segmentation_processing.py
│   └── wing_processing.py
├── training/             # Model training scripts and weights
│   ├── data_preparation/ # Python scripts for preparing training data
│   ├── *.ipynb          # Training notebooks
│   └── models/          # Pretrained model weights
├── static/               # Web interface assets
├── templates/            # HTML templates
├── app.py               # Flask web application
├── main.py              # Core pipeline functions
├── utils.py             # Utility functions
├── config.py            # Configuration settings
└── requirements.txt     # Python dependencies
```

## Landmark Reference and Adding New Families

You can add new families to the system by defining a new config family in `config_loader.py` and retraining both the segmentation and landmark model. Please refer to the training guides in `training/` for detailed instructions on preparing training data and running the training scripts. We have provided a small "example" dataset to test the training pipeline locally, you can replace this with your own data for training on a new family. The training is computationally intensive and while it can be done on a local machine with a good GPU, we recommend using Google Colab or a similar cloud service for training. For this reason we also provide Colab notebooks for training both the segmentation and landmark models. 

First retrain the segmentation model to segment the wings of your new family, then prepare training data for the landmark model and retrain it. Herafter you can use the new models for inference in the same way as before, just make sure to update the config to match the new family and model paths.

## Citation

If you use ITHILDIN in your research, please cite:

```
[Citation to be added]
```

## License

CC BY-NC 4.0 License

## Contact

For questions or support, please contact the repository maintainer.

## Acknowledgments

- Uses the `rembg` library for background removal
- Integrates with R's `geomorph` package for morphometric analysis
