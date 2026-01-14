"""
Similarity search module for Visual Memory AI.
Finds similar memorable and forgettable images.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


class SimilaritySearchEngine:
    """
    Image similarity search using learned embeddings.
    Finds similar images based on visual features.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Initialize search engine.
        
        Args:
            model: Trained model with get_features() method
            device: Device for computation
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.embeddings = None
        self.image_paths = None
        self.memorability_scores = None
        
    def extract_features(
        self,
        image_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Extract feature embedding from image.
        
        Args:
            image_tensor: Preprocessed image tensor [1, 3, 224, 224]
            
        Returns:
            Feature vector as numpy array
        """
        with torch.no_grad():
            features = self.model.get_features(image_tensor)
            features = features.cpu().numpy()
        return features
    
    def build_index(
        self,
        dataloader: torch.utils.data.DataLoader,
        image_paths: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Build search index from dataset.
        
        Args:
            dataloader: DataLoader for images
            image_paths: List of image paths corresponding to dataloader
            save_path: Optional path to save index
        """
        all_embeddings = []
        all_scores = []
        
        print("Building search index...")
        for images, scores in tqdm(dataloader):
            images = images.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model.get_features(images)
                all_embeddings.append(features.cpu().numpy())
                all_scores.extend(scores.numpy())
        
        # Concatenate all embeddings
        self.embeddings = np.vstack(all_embeddings)
        self.memorability_scores = np.array(all_scores)
        self.image_paths = image_paths
        
        print(f"Index built with {len(self.embeddings)} images")
        
        # Normalize embeddings for cosine similarity
        self.embeddings = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Save index
        if save_path:
            self.save_index(save_path)
    
    def save_index(self, path: str) -> None:
        """Save search index to disk."""
        index_data = {
            'embeddings': self.embeddings,
            'image_paths': self.image_paths,
            'memorability_scores': self.memorability_scores
        }
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Index saved to {path}")
    
    def load_index(self, path: str) -> None:
        """Load search index from disk."""
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.embeddings = index_data['embeddings']
        self.image_paths = index_data['image_paths']
        self.memorability_scores = index_data['memorability_scores']
        print(f"Index loaded from {path}")
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        exclude_idx: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Find most similar images to query.
        
        Args:
            query_embedding: Query feature vector [1, feature_dim]
            top_k: Number of results to return
            exclude_idx: Optional index to exclude from results
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_norm, self.embeddings)[0]
        
        # Exclude query image itself if specified
        if exclude_idx is not None:
            similarities[exclude_idx] = -1
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(idx, similarities[idx]) for idx in top_indices]
        return results
    
    def search_by_memorability(
        self,
        query_embedding: np.ndarray,
        memorable: bool = True,
        top_k: int = 5
    ) -> List[Tuple[int, float, float]]:
        """
        Find similar images that are memorable or forgettable.
        
        Args:
            query_embedding: Query feature vector
            memorable: If True, find memorable images; else forgettable
            top_k: Number of results
            
        Returns:
            List of (index, similarity, memorability) tuples
        """
        # Find similar images
        similar_results = self.search_similar(query_embedding, top_k=top_k * 10)
        
        # Filter by memorability threshold
        threshold = 0.5
        filtered = []
        
        for idx, sim in similar_results:
            mem_score = self.memorability_scores[idx]
            
            if memorable and mem_score > threshold:
                filtered.append((idx, sim, mem_score))
            elif not memorable and mem_score < threshold:
                filtered.append((idx, sim, mem_score))
            
            if len(filtered) >= top_k:
                break
        
        return filtered
    
    def get_image_info(self, idx: int) -> Dict:
        """
        Get information about an image in the index.
        
        Args:
            idx: Image index
            
        Returns:
            Dictionary with image information
        """
        return {
            'path': self.image_paths[idx],
            'memorability': self.memorability_scores[idx],
            'embedding': self.embeddings[idx]
        }


def create_similarity_gallery(
    model: torch.nn.Module,
    query_image_path: str,
    search_engine: SimilaritySearchEngine,
    device: str = 'cpu',
    top_k: int = 3
) -> Dict[str, List[Dict]]:
    """
    Create a gallery of similar memorable and forgettable images.
    
    Args:
        model: Trained model
        query_image_path: Path to query image
        search_engine: Similarity search engine
        device: Device for computation
        top_k: Number of results per category
        
    Returns:
        Dictionary with 'memorable' and 'forgettable' lists
    """
    from torchvision import transforms
    
    # Load and preprocess query image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(query_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract query embedding
    with torch.no_grad():
        query_embedding = model.get_features(input_tensor).cpu().numpy()
    
    # Search for memorable images
    memorable_results = search_engine.search_by_memorability(
        query_embedding,
        memorable=True,
        top_k=top_k
    )
    
    # Search for forgettable images
    forgettable_results = search_engine.search_by_memorability(
        query_embedding,
        memorable=False,
        top_k=top_k
    )
    
    # Format results
    gallery = {
        'memorable': [
            {
                'path': search_engine.image_paths[idx],
                'similarity': float(sim),
                'memorability': float(mem)
            }
            for idx, sim, mem in memorable_results
        ],
        'forgettable': [
            {
                'path': search_engine.image_paths[idx],
                'similarity': float(sim),
                'memorability': float(mem)
            }
            for idx, sim, mem in forgettable_results
        ]
    }
    
    return gallery


if __name__ == "__main__":
    from models.model import create_model
    
    # Create dummy model
    model = create_model('resnet50', device='cpu')
    
    # Initialize search engine
    search_engine = SimilaritySearchEngine(model)
    
    # Create dummy embeddings
    dummy_embeddings = np.random.randn(100, 2048)
    dummy_paths = [f"image_{i}.jpg" for i in range(100)]
    dummy_scores = np.random.rand(100)
    
    search_engine.embeddings = dummy_embeddings
    search_engine.image_paths = dummy_paths
    search_engine.memorability_scores = dummy_scores
    
    # Test search
    query = np.random.randn(1, 2048)
    results = search_engine.search_similar(query, top_k=5)
    
    print("Similar images:")
    for idx, sim in results:
        print(f"  Image {idx}: similarity={sim:.3f}, mem={dummy_scores[idx]:.3f}")