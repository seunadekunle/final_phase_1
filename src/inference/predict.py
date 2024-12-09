import torch
import logging
from pathlib import Path
from typing import List, Tuple, Dict

from src.models.style_classifier import StyleClassifier
from src.data.dataset import DeepFashionDataset

class StylePredictor:
    def __init__(self, checkpoint_path: str):
        """Initialize the predictor with a trained model.
        
        Args:
            checkpoint_path: Path to the model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_config = self.checkpoint['version_info']['model_config']
        
        # Initialize model
        self.model = StyleClassifier(
            num_categories=self.model_config['architecture']['num_categories'],
            num_attributes=self.model_config['architecture']['num_attributes']
        )
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"Loaded model version: {self.checkpoint['version_info']['version_id']}")
        
    def predict_single(self, image_path: str) -> Dict[str, float]:
        """Predict style categories and attributes for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing predicted categories and attributes with their probabilities
        """
        # Load and preprocess image
        image = DeepFashionDataset.load_and_transform_image(image_path)
        image = image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            category_logits, attribute_logits = self.model(image)
            category_probs = torch.softmax(category_logits, dim=1)
            attribute_probs = torch.sigmoid(attribute_logits)
            
        return {
            'category_probabilities': category_probs[0].cpu().numpy(),
            'attribute_probabilities': attribute_probs[0].cpu().numpy()
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, float]]:
        """Predict style categories and attributes for a batch of images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of dictionaries containing predicted categories and attributes
        """
        # Load and preprocess images
        images = []
        for path in image_paths:
            image = DeepFashionDataset.load_and_transform_image(path)
            images.append(image)
        
        batch = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            category_logits, attribute_logits = self.model(batch)
            category_probs = torch.softmax(category_logits, dim=1)
            attribute_probs = torch.sigmoid(attribute_logits)
            
        return [
            {
                'category_probabilities': cat_prob.cpu().numpy(),
                'attribute_probabilities': attr_prob.cpu().numpy()
            }
            for cat_prob, attr_prob in zip(category_probs, attribute_probs)
        ]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Style Classifier Prediction")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize predictor
    predictor = StylePredictor(args.checkpoint)
    
    # Make prediction
    result = predictor.predict_single(args.image)
    
    # Print results
    print("\nTop 5 Categories:")
    cat_probs = result['category_probabilities']
    top_cats = torch.topk(torch.from_numpy(cat_probs), k=5)
    for idx, prob in zip(top_cats.indices, top_cats.values):
        print(f"Category {idx}: {prob:.4f}")
    
    print("\nTop 5 Attributes:")
    attr_probs = result['attribute_probabilities']
    top_attrs = torch.topk(torch.from_numpy(attr_probs), k=5)
    for idx, prob in zip(top_attrs.indices, top_attrs.values):
        print(f"Attribute {idx}: {prob:.4f}") 