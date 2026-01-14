"""
Streamlit web application for Visual Memory AI.
Interactive interface for image memorability prediction.
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.model import create_model
from explainability.gradcam import GradCAM
from similarity.search import SimilaritySearchEngine, create_similarity_gallery
from torchvision import transforms


# Page configuration
st.set_page_config(
    page_title="Visual Memory AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .score-display {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str = None, device: str = 'cpu'):
    """Load trained model with caching."""
    model = create_model('resnet50', pretrained=True, device=device)
    
    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.sidebar.success(f"Loaded model from {model_path}")
    else:
        st.sidebar.warning("Using pretrained model (not fine-tuned)")
    
    return model


@st.cache_resource
def load_search_engine(_model, index_path: str = None):
    """Load similarity search engine with caching."""
    search_engine = SimilaritySearchEngine(_model)
    
    if index_path and Path(index_path).exists():
        search_engine.load_index(index_path)
        st.sidebar.success(f"Loaded search index with {len(search_engine.embeddings)} images")
    else:
        st.sidebar.warning("Search index not loaded")
    
    return search_engine


def preprocess_image(image: Image.Image, device: str) -> torch.Tensor:
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0).to(device)


def predict_memorability(model, image_tensor, device):
    """Predict memorability score."""
    model.eval()
    with torch.no_grad():
        score = model(image_tensor).item()
    return score * 100  # Convert to 0-100 scale


def generate_explanation(model, image_tensor):
    """Generate Grad-CAM heatmap."""
    gradcam = GradCAM(model)
    heatmap = gradcam.generate_cam(image_tensor)
    return heatmap


def main():
    # Header
    st.markdown('<div class="main-header">🧠 Visual Memory AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict how memorable your images are to humans</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"Running on: {device.upper()}")
    
    model_path = st.sidebar.text_input("Model checkpoint path (optional)", "")
    index_path = st.sidebar.text_input("Search index path (optional)", "")
    
    # Load model and search engine
    model = load_model(model_path if model_path else None, device)
    search_engine = load_search_engine(model, index_path if index_path else None)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to analyze its memorability"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("🔮 Analyze Memorability", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Preprocess
                    image_tensor = preprocess_image(image, device)
                    
                    # Predict
                    score = predict_memorability(model, image_tensor, device)
                    
                    # Store in session state
                    st.session_state.score = score
                    st.session_state.image_tensor = image_tensor
                    st.session_state.original_image = image
    
    with col2:
        st.subheader("📊 Results")
        
        if 'score' in st.session_state:
            score = st.session_state.score
            
            # Display memorability score
            st.markdown(f"""
            <div class="metric-card">
                <h3>Memorability Score</h3>
                <div class="score-display">{score:.1f}/100</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretation
            if score >= 70:
                interpretation = "🌟 Highly Memorable"
                color = "green"
                explanation = "This image is predicted to be very memorable to humans!"
            elif score >= 40:
                interpretation = "👍 Moderately Memorable"
                color = "orange"
                explanation = "This image has average memorability."
            else:
                interpretation = "👎 Less Memorable"
                color = "red"
                explanation = "This image may be less memorable to humans."
            
            st.markdown(f"**{interpretation}**")
            st.info(explanation)
            
            # Show progress bar
            st.progress(score / 100)
    
    # Explainability section
    if 'score' in st.session_state:
        st.markdown("---")
        st.subheader("🔍 Explainability - What Makes It Memorable?")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.write("**Original Image**")
            st.image(st.session_state.original_image, use_column_width=True)
        
        with col2:
            st.write("**Attention Heatmap**")
            with st.spinner("Generating heatmap..."):
                heatmap = generate_explanation(model, st.session_state.image_tensor)
                st.image(heatmap, use_column_width=True, clamp=True)
        
        with col3:
            st.write("**Overlay**")
            with st.spinner("Creating overlay..."):
                gradcam = GradCAM(model)
                image_np = np.array(st.session_state.original_image.resize((224, 224)))
                overlayed = gradcam.overlay_heatmap(image_np, heatmap, alpha=0.5)
                st.image(overlayed, use_column_width=True)
        
        st.info("🔥 Red regions indicate areas that most influence the memorability prediction.")
    
    # Similar images section
    if 'score' in st.session_state and search_engine.embeddings is not None:
        st.markdown("---")
        st.subheader("🔎 Similar Images")
        
        tab1, tab2 = st.tabs(["🌟 Similar Memorable Images", "👎 Similar Forgettable Images"])
        
        with tab1:
            with st.spinner("Finding similar memorable images..."):
                try:
                    # Extract query embedding
                    with torch.no_grad():
                        query_emb = model.get_features(st.session_state.image_tensor).cpu().numpy()
                    
                    results = search_engine.search_by_memorability(
                        query_emb,
                        memorable=True,
                        top_k=3
                    )
                    
                    if results:
                        cols = st.columns(3)
                        for i, (idx, sim, mem) in enumerate(results):
                            with cols[i]:
                                img_path = search_engine.image_paths[idx]
                                if Path(img_path).exists():
                                    st.image(img_path, use_column_width=True)
                                    st.caption(f"Memorability: {mem*100:.1f}/100")
                                    st.caption(f"Similarity: {sim:.2f}")
                    else:
                        st.warning("No memorable images found in the database.")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with tab2:
            with st.spinner("Finding similar forgettable images..."):
                try:
                    with torch.no_grad():
                        query_emb = model.get_features(st.session_state.image_tensor).cpu().numpy()
                    
                    results = search_engine.search_by_memorability(
                        query_emb,
                        memorable=False,
                        top_k=3
                    )
                    
                    if results:
                        cols = st.columns(3)
                        for i, (idx, sim, mem) in enumerate(results):
                            with cols[i]:
                                img_path = search_engine.image_paths[idx]
                                if Path(img_path).exists():
                                    st.image(img_path, use_column_width=True)
                                    st.caption(f"Memorability: {mem*100:.1f}/100")
                                    st.caption(f"Similarity: {sim:.2f}")
                    else:
                        st.warning("No forgettable images found in the database.")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # About section
    with st.expander("ℹ️ About Visual Memory AI"):
        st.markdown("""
        **Visual Memory AI** predicts how memorable an image is to humans using deep learning.
        
        **How it works:**
        1. **Deep Learning Model**: ResNet50 trained on the LaMem dataset
        2. **Memorability Score**: 0-100 scale predicting human recall ability
        3. **Explainability**: Grad-CAM highlights influential regions
        4. **Similarity Search**: Finds visually similar memorable/forgettable images
        
        **Applications:**
        - Marketing & Advertising
        - Content Creation
        - Education & Learning Materials
        - Social Media Optimization
        
        **Dataset**: LaMem (Large-scale Image Memorability)
        """)


if __name__ == "__main__":
    main()