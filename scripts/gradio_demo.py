"""
Interactive Web Demo for Raman Spectroscopy Classification
Using Gradio for browser-based interface
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from typing import Tuple, Optional

from src.model import create_model


class V3ModelLoader:
    """Load and manage the v3 model for inference."""
    
    def __init__(self, model_dir: str = "models/saved_models_v3"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.target_grid = None
        self.class_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load the trained v3 model and associated files."""
        print(f"Loading model from {self.model_dir}...")
        
        # Load class names
        class_names_path = self.model_dir / "class_names_v3.json"
        if not class_names_path.exists():
            raise FileNotFoundError(f"Class names file not found: {class_names_path}")
        
        with open(class_names_path, 'r') as f:
            class_names_dict = json.load(f)
        
        # Convert dict to list ordered by class index
        # Handle both string keys ("0") and int keys (0)
        n_classes = len(class_names_dict)
        self.class_names = []
        for i in range(n_classes):
            # Try string key first, then int key
            name = class_names_dict.get(str(i)) or class_names_dict.get(i)
            if name is None:
                # Fallback: use dict value directly if keys are class names
                name = list(class_names_dict.values())[i] if i < len(class_names_dict) else f"Class_{i}"
            self.class_names.append(name)
        
        # Load target grid
        target_grid_path = self.model_dir / "target_grid_v3.npy"
        if not target_grid_path.exists():
            raise FileNotFoundError(f"Target grid file not found: {target_grid_path}")
        
        self.target_grid = np.load(target_grid_path)
        input_length = len(self.target_grid)
        
        # Load model
        model_state_path = self.model_dir / "model_state_v3.pth"
        if not model_state_path.exists():
            # Try checkpoint instead
            checkpoint_path = self.model_dir / "model_checkpoint_v3.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                raise FileNotFoundError(f"Model file not found in {self.model_dir}")
        else:
            state_dict = torch.load(model_state_path, map_location=self.device)
        
        # Create model architecture (matching training config)
        self.model = create_model(
            input_length=input_length,
            n_classes=n_classes,
            config={
                'n_channels': [32, 64, 128, 256],
                'kernel_sizes': [7, 5, 5, 3],
                'pool_sizes': [2, 2, 2, 2],
                'use_batch_norm': True,
                'dropout': 0.3,
                'fc_hidden': [128, 64],
                'use_ordinal_head': False
            }
        )
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"  Classes: {self.class_names}")
        print(f"  Input length: {input_length}")
        print(f"  Device: {self.device}")
    
    def predict_spectrum(self, spectrum_1d: np.ndarray) -> Tuple[str, dict, float]:
        """
        Predict class for a single spectrum.
        
        Args:
            spectrum_1d: 1D array of intensities (same length as target_grid)
            
        Returns:
            Tuple of (predicted_class_name, class_probabilities_dict, confidence)
        """
        # Ensure spectrum is on the correct grid
        if len(spectrum_1d) != len(self.target_grid):
            raise ValueError(
                f"Spectrum length ({len(spectrum_1d)}) doesn't match "
                f"model input length ({len(self.target_grid)}). "
                f"Expected wavenumber range: {self.target_grid.min():.1f} - {self.target_grid.max():.1f} cm⁻¹"
            )
        
        # Convert to tensor: (batch=1, channels=1, length=N)
        x = torch.tensor(spectrum_1d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)
        
        # Predict
        with torch.no_grad():
            class_logits, _ = self.model(x)
            probs = torch.softmax(class_logits, dim=1).cpu().numpy()[0]
        
        pred_idx = int(probs.argmax())
        predicted_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Create probability dictionary
        prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        
        return predicted_class, prob_dict, confidence


# Global model loader
model_loader = None


def load_spectrum_from_file(file_path: str) -> np.ndarray:
    """Load spectrum from various file formats."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.npy':
        spectrum = np.load(file_path)
    elif file_path.suffix == '.txt' or file_path.suffix == '.csv':
        # Try loading as CSV first (two columns: wavenumber, intensity)
        try:
            data = np.loadtxt(file_path, delimiter=',')
            if data.ndim == 2 and data.shape[1] == 2:
                # Two columns: extract intensities
                spectrum = data[:, 1]
            else:
                # Single column: intensities only
                spectrum = data
        except:
            # Try space/tab separated
            data = np.loadtxt(file_path)
            if data.ndim == 2 and data.shape[1] == 2:
                spectrum = data[:, 1]
            else:
                spectrum = data
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Flatten if needed
    if spectrum.ndim > 1:
        spectrum = spectrum.flatten()
    
    return spectrum


def predict_from_file(file) -> Tuple[str, dict, plt.Figure]:
    """Gradio interface function for file upload."""
    if file is None:
        return "Please upload a spectrum file", {}, None
    
    try:
        # Load spectrum
        spectrum = load_spectrum_from_file(file.name)
        
        # Predict
        predicted_class, prob_dict, confidence = model_loader.predict_spectrum(spectrum)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(model_loader.target_grid, spectrum, linewidth=1.5)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)
        ax.set_title(
            f"Predicted: {predicted_class}\nConfidence: {confidence:.2%}",
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return f"**{predicted_class}** (Confidence: {confidence:.2%})", prob_dict, fig
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, {}, None


def predict_from_array(spectrum_array: np.ndarray) -> Tuple[str, dict, plt.Figure]:
    """Gradio interface function for numpy array input."""
    if spectrum_array is None or len(spectrum_array) == 0:
        return "Please provide a spectrum array", {}, None
    
    try:
        # Convert to numpy array if needed
        if not isinstance(spectrum_array, np.ndarray):
            spectrum_array = np.array(spectrum_array)
        
        # Flatten if needed
        if spectrum_array.ndim > 1:
            spectrum_array = spectrum_array.flatten()
        
        # Predict
        predicted_class, prob_dict, confidence = model_loader.predict_spectrum(spectrum_array)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(model_loader.target_grid, spectrum_array, linewidth=1.5)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)
        ax.set_title(
            f"Predicted: {predicted_class}\nConfidence: {confidence:.2%}",
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return f"**{predicted_class}** (Confidence: {confidence:.2%})", prob_dict, fig
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, {}, None


def create_demo(model_dir: str = "models/saved_models_v3"):
    """Create and launch the Gradio demo."""
    global model_loader
    
    # Load model
    try:
        model_loader = V3ModelLoader(model_dir=model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPlease make sure you have trained the v3 model first by running:")
        print("  python train_v3.py --data 'v3 data/synthetic_graphene_parametric_9class_v2.npz'")
        raise
    
    # Create Gradio interface
    with gr.Blocks(title="Graphene Raman Classifier") as demo:
        gr.Markdown(
            """
            # 🧪 Graphene Raman Spectroscopy Classifier
            
            Upload a Raman spectrum file (.npy, .txt, or .csv) or paste a spectrum array to classify graphene-related materials.
            
            **Supported Classes:**
            - Graphite
            - Exfoliated Graphene
            - GNP (High Quality)
            - GNP (Medium Quality)
            - Multilayer Graphene
            - Graphene Oxide (GO)
            - Reduced Graphene Oxide (rGO)
            - Defective Graphene
            - Graphitized Carbon
            
            **Expected Spectrum:**
            - Length: 1500 points
            - Wavenumber range: 800 - 3200 cm⁻¹
            - Format: 1D array of intensities
            """
        )
        
        with gr.Tabs():
            with gr.Tab("📁 Upload File"):
                file_input = gr.File(
                    label="Upload Raman Spectrum",
                    file_types=[".npy", ".txt", ".csv"]
                )
                file_output_text = gr.Markdown(label="Prediction")
                file_output_probs = gr.Label(
                    label="Class Probabilities",
                    num_top_classes=len(model_loader.class_names)
                )
                file_output_plot = gr.Plot(label="Spectrum Visualization")
                
                file_input.change(
                    fn=predict_from_file,
                    inputs=file_input,
                    outputs=[file_output_text, file_output_probs, file_output_plot]
                )
            
            with gr.Tab("📊 Paste Array"):
                array_input = gr.Textbox(
                    label="Paste Spectrum Array",
                    placeholder="Paste comma-separated values or space-separated values (1500 numbers)\nExample: 100.5, 102.3, 98.7, ...",
                    lines=5
                )
                array_output_text = gr.Markdown(label="Prediction")
                array_output_probs = gr.Label(
                    label="Class Probabilities",
                    num_top_classes=len(model_loader.class_names)
                )
                array_output_plot = gr.Plot(label="Spectrum Visualization")
                
                def parse_and_predict(text):
                    """Parse text input and predict."""
                    if not text or text.strip() == "":
                        return "Please paste spectrum values", {}, None
                    
                    try:
                        # Try comma-separated first
                        if ',' in text:
                            values = [float(x.strip()) for x in text.split(',')]
                        else:
                            # Try space-separated
                            values = [float(x.strip()) for x in text.split()]
                        
                        spectrum = np.array(values)
                        return predict_from_array(spectrum)
                    except Exception as e:
                        return f"Error parsing array: {str(e)}\nPlease provide comma or space-separated numbers.", {}, None
                
                array_input.change(
                    fn=parse_and_predict,
                    inputs=array_input,
                    outputs=[array_output_text, array_output_probs, array_output_plot]
                )
        
        gr.Markdown(
            """
            ---
            **Note:** This model was trained on synthetic graphene Raman spectra without preprocessing.
            Make sure your input spectrum matches the expected format and wavenumber range.
            """
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio web demo for Raman classifier")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/saved_models_v3",
        help="Directory containing saved model files"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link (gradio share)"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server name (default: 127.0.0.1 for local)"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )
    
    args = parser.parse_args()
    
    # Create and launch demo
    demo = create_demo(model_dir=args.model_dir)
    
    print(f"\n{'='*60}")
    print("Launching Gradio Web Demo...")
    print(f"{'='*60}")
    print(f"\nLocal URL: http://{args.server_name}:{args.server_port}")
    if args.share:
        print("Public link will be generated...")
    print(f"\nPress Ctrl+C to stop the server.\n")
    
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )

