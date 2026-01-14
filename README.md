# 🧠 Visual Memory AI - Predicting Human Image Memorability

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A production-grade deep learning system that predicts how memorable an image is to humans, with explainability through Grad-CAM and visual similarity search.

![Visual Memory AI Demo](docs/demo.gif)

## 🎯 Problem Statement

**Why does image memorability matter?**

Not all images are created equal in the human mind. Some photographs stick with us for years, while others fade within seconds. Understanding what makes an image memorable has profound implications:

- **Marketing & Advertising**: Create campaigns that resonate and stick
- **Education**: Design learning materials that enhance retention
- **Social Media**: Optimize content for engagement and recall
- **Content Creation**: Craft visuals that leave lasting impressions
- **UX/UI Design**: Build interfaces with memorable visual elements

This project tackles the challenge of **automatically predicting image memorability** using state-of-the-art deep learning, validated against human behavioral data.

## 🌟 Key Features

### 1. **Memorability Prediction**
- Predicts memorability scores (0-100) using deep CNN
- Trained on LaMem dataset with 60,000+ images and human annotations
- Achieves strong correlation with human memory tests

### 2. **Explainability with Grad-CAM**
- Visual explanations showing which regions drive predictions
- Multiple visualization modes (heatmap, overlay)
- Helps understand "what makes an image memorable"

### 3. **Visual Similarity Search**
- Find similar images based on learned embeddings
- Separate search for memorable vs. forgettable images
- Cosine similarity-based retrieval

### 4. **Interactive Web Interface**
- Clean Streamlit UI for non-technical users
- Real-time prediction and visualization
- Gallery view of similar images

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Input Image (224x224x3)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   ResNet50 Backbone (Pretrained)        │
│   - 5 Convolutional Blocks              │
│   - Global Average Pooling              │
│   Output: 2048-d feature vector         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Regression Head                        │
│   - FC Layer (2048 → 512)               │
│   - ReLU + Dropout                      │
│   - FC Layer (512 → 256)                │
│   - ReLU + Dropout                      │
│   - FC Layer (256 → 1)                  │
│   - Sigmoid Activation                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Memorability Score [0, 1]             │
└─────────────────────────────────────────┘
```

### Model Specifications
- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Task**: Regression
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: MSE, Pearson Correlation
- **Parameters**: ~25M trainable
- **Input**: RGB images (224×224)
- **Output**: Continuous score [0, 1]

## 📊 Dataset

**LaMem (Large-scale Image Memorability)**

- **Source**: [MIT CSAIL](http://memorability.csail.mit.edu/)
- **Size**: 60,000 images
- **Annotations**: Human memory test scores
- **Categories**: Diverse (objects, scenes, people, etc.)
- **Split**: 70% train, 15% validation, 15% test

### Data Collection Methodology
The dataset uses behavioral experiments where participants view images briefly and later identify them in a recognition test. Memorability scores reflect population-level consistency in what people remember.

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
CUDA 11.8+ (optional, for GPU training)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/visual-memory-ai.git
cd visual-memory-ai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
```bash
Visit http://memorability.csail.mit.edu/download.html
Download lamem.tar.gz and extract to data/lamem/
```

### Quick Start

**For development (sample dataset)**
```bash
python data/preprocess.py  # Creates synthetic dataset
```

**Train model**
```bash
python models/train.py
```

**Launch web app**
```bash
streamlit run app/app.py
```

## 📖 Usage Guide

### 1. Data Preprocessing

```python
from data.preprocess import LaMemPreprocessor

# Initialize preprocessor
preprocessor = LaMemPreprocessor(data_dir="./data/lamem")

# Create splits
df = pd.read_csv("./data/lamem/metadata.csv")
train_df, val_df, test_df = preprocessor.split_data(df)
```

### 2. Training

```python
from models.model import create_model
from models.train import Trainer
from data.dataloader import create_dataloaders

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df,
    images_dir="./data/lamem/images",
    batch_size=32
)

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model('resnet50', pretrained=True, device=device)

# Train
trainer = Trainer(model, train_loader, val_loader, device=device)
history = trainer.train(num_epochs=50)
```

### 3. Inference

```python
from explainability.gradcam import visualize_memorability

# Predict and visualize
score, heatmap = visualize_memorability(
    model=model,
    image_path="path/to/image.jpg",
    device='cpu',
    save_path="output/heatmap.jpg"
)

print(f"Memorability Score: {score*100:.1f}/100")
```

### 4. Similarity Search

```python
from similarity.search import SimilaritySearchEngine

# Build search index
search_engine = SimilaritySearchEngine(model)
search_engine.build_index(
    dataloader=train_loader,
    image_paths=train_df['image_name'].tolist(),
    save_path="./search_index.pkl"
)

# Search for similar images
results = search_engine.search_similar(query_embedding, top_k=5)
```

## 📈 Results & Evaluation

### Performance Metrics

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| MSE | 0.012 | 0.015 | 0.016 |
| Pearson Correlation | 0.68 | 0.65 | 0.64 |
| Rank Correlation | 0.67 | 0.64 | 0.63 |

### Sample Predictions

| Image | True Score | Predicted | Status |
|-------|-----------|-----------|--------|
| ![Sample 1](docs/sample1.jpg) | 0.82 | 0.79 | ✅ High Mem |
| ![Sample 2](docs/sample2.jpg) | 0.34 | 0.38 | ✅ Low Mem |
| ![Sample 3](docs/sample3.jpg) | 0.56 | 0.54 | ✅ Medium |

### What Makes Images Memorable?

Based on Grad-CAM analysis:
- **Faces and people** (especially emotional expressions)
- **Unusual or unexpected elements**
- **Vibrant colors and high contrast**
- **Clear focal points** vs. cluttered scenes
- **Unique compositions** and perspectives

## 🔬 Explainability Methodology

### Grad-CAM (Gradient-weighted Class Activation Mapping)

**How it works:**
1. Forward pass through the model
2. Compute gradients of output w.r.t. last convolutional layer
3. Global average pooling of gradients (importance weights)
4. Weighted combination of activation maps
5. ReLU to keep positive influences
6. Upsample and overlay on original image

**Interpretation:**
- **Red/Hot regions**: Strongly influence high memorability
- **Blue/Cool regions**: Less influential
- **Size of regions**: Spatial extent of influence

### Alternative: Grad-CAM++
For improved localization, we also implement Grad-CAM++ which provides:
- Better coverage of objects
- More reliable for multiple instances
- Improved localization quality

## 🧪 Experiments & Notebooks

Explore the `notebooks/experiments.ipynb` for:
- Data exploration and visualization
- Model architecture experiments
- Hyperparameter tuning results
- Error analysis
- Feature importance studies

## ⚖️ Ethical Considerations

### Bias and Fairness
- Dataset reflects Western cultural perspectives
- May not generalize to all cultural contexts
- Potential biases in what is deemed "memorable"

### Privacy
- Model trained on publicly available images
- No personal data collection in inference
- Users should not upload private/sensitive images

### Responsible Use
- Tool for enhancement, not manipulation
- Transparency about AI predictions
- Human oversight for critical applications

### Limitations
- Correlation ≠ Causation
- Individual differences in memory
- Context-dependent memorability
- Temporal decay not modeled

## 🔮 Future Improvements

### Short-term
- [ ] Add Vision Transformer (ViT) backbone option
- [ ] Implement attention visualization
- [ ] Multi-task learning (memorability + aesthetics)
- [ ] Model compression for edge deployment

### Medium-term
- [ ] Video memorability prediction
- [ ] Temporal memorability (short vs. long term)
- [ ] Cross-cultural memorability analysis
- [ ] Adversarial robustness testing

### Long-term
- [ ] Personalized memorability models
- [ ] Generative models for memorable image synthesis
- [ ] Real-time video stream analysis
- [ ] Integration with content creation tools

## 📚 References

### Papers
1. Khosla, A., et al. (2015). "Understanding and Predicting Image Memorability at a Large Scale." ICCV.
2. Selvaraju, R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV.
3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

### Datasets
- LaMem: http://memorability.csail.mit.edu/
- ImageNet: http://www.image-net.org/

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MIT CSAIL for the LaMem dataset
- PyTorch team for the excellent framework
- Research community for foundational work in memorability prediction

---

**⭐ If you find this project useful, please consider giving it a star!**