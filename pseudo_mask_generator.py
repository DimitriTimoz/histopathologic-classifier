import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import tqdm
from typing import Optional, List, Tuple
import cv2
from model import CNN_PDA, CNN
from dataset import FilenameLabelImageDataset, list_image_files, label_from_filename
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class PseudoMaskGenerator:
    def __init__(
        self,
        model: CNN_PDA,
        device: torch.device,
        class_to_idx: dict,
        idx_to_class: dict,
        bg_threshold: float = 0.20,
        fg_thresholds: List[float] = [0.25, 0.35, 0.45, 0.55],
    ):

        self.model = model
        self.device = device
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.num_classes = len(class_to_idx)
        self.bg_threshold = bg_threshold
        self.fg_thresholds = sorted(fg_thresholds)
        
        self.model.eval()
    
    def generate_cam_for_image(
        self, 
        image: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:

        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Obtenir les CAMs pour toutes les classes
        with torch.no_grad():
            cams = self.model.get_attention_maps(image)
            logits = self.model(image, apply_attention=False)
            predicted_class = torch.argmax(logits, dim=1).item()
        
        # Sélectionner la CAM de la classe cible ou prédite
        if target_class is None:
            target_class = predicted_class
        
        cam = cams[0, target_class].cpu().numpy()
        
        # Normaliser la CAM entre 0 et 1
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, predicted_class
    
    def resize_cam_to_image(
        self, 
        cam: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> np.ndarray:

        cam_resized = cv2.resize(
            cam, 
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        return cam_resized
    
    def generate_multi_layer_mask(
        self, 
        cam: np.ndarray, 
        predicted_class: int,
        image_size: Tuple[int, int]
    ) -> List[np.ndarray]:

        masks = []
        
        for threshold in self.fg_thresholds:
            mask = np.zeros(image_size, dtype=np.uint8)
            
            mask[cam >= threshold] = predicted_class + 1  # +1 car 0 = background
            
            mask[cam < self.bg_threshold] = 0
            
            masks.append(mask)
        
        return masks
    
    def process_image_file(
        self, 
        image_path: Path,
        transform,
        save_dir: Optional[Path] = None,
        save_visualization: bool = False
    ) -> dict:

        image_pil = Image.open(image_path).convert("RGB")
        image_uint8 = np.array(image_pil)
        image_np = image_uint8.astype(float)
        original_size = (image_np.shape[0], image_np.shape[1])
        
        image_tensor = transform(image_pil)
        
        cam, predicted_class = self.generate_cam_for_image(image_tensor)

        if hasattr(predicted_class, 'item'): 
            predicted_class = predicted_class.item()
        
        cam_resized = self.resize_cam_to_image(cam, original_size)
        
        masks = self.generate_multi_layer_mask(
            cam_resized, 
            predicted_class,
            original_size
        )
        
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            if not os.path.exists(save_dir / image_path.name):
                img_dir = save_dir / image_path.name.split('.')[0]
            
            stem = image_path.stem
            if save_visualization:
                self.save_visualization(
                    image_uint8, 
                    cam_resized, 
                    masks, 
                    predicted_class,
                    save_dir / f"{stem}_visualization.png"
                )
        
        return {
            "image_path": str(image_path),
            "predicted_class": predicted_class,
            "predicted_class_name": self.idx_to_class[predicted_class],
            "cam": cam_resized,
            "masks": masks,
        }
    
    def save_visualization(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        masks: List[np.ndarray],
        predicted_class: int,
        save_path: Path
    ):
        n_masks = len(masks)
        
        fig, axes = plt.subplots(2, n_masks + 1, figsize=(4 * (n_masks + 1), 8))
        
        axes[0, 0].imshow(image.astype(np.uint8))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        
        for i in range(n_masks):
            axes[0, i + 1].imshow(image.astype(np.uint8))
            axes[0, i + 1].imshow(cam, alpha=0.5, cmap='jet')
            axes[0, i + 1].set_title(f"CAM Overlay\nThreshold: {self.fg_thresholds[i]:.2f}")
            axes[0, i + 1].axis("off")
        
        axes[1, 0].text(
            0.5, 0.5, 
            f"Predicted:\n{self.idx_to_class[predicted_class]}", 
            ha='center', va='center',
            fontsize=12, fontweight='bold'
        )
        axes[1, 0].axis("off")

        tab10_colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        
        colors = ['black'] + tab10_colors.tolist()
        
        cmap = ListedColormap(colors)
        
        for i, (mask, threshold) in enumerate(zip(masks, self.fg_thresholds)):
            im = axes[1, i + 1].imshow(mask, cmap=cmap, vmin=0, vmax=self.num_classes)
            axes[1, i + 1].set_title(f"Pseudo Mask\nThreshold: {threshold:.2f}")
            axes[1, i + 1].axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close(fig)
    
    def process_dataset(
        self,
        data_folder: str,
        output_folder: str,
        transform,
        apply_crf: bool = False,
        save_visualization: bool = True,
        exclude_classes: Optional[set[str]] = None
    ):

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list_image_files(data_folder)
        
        print(f"Processing {len(image_files)} images...")
        
        results = []
        MAX_IMG_PER_CLASS = 10
        nb_img_per_class = {'Ignore': 0, 'Necrosis': 0, 'Stroma': 0, 'Tumor': 0}
        for image_path in tqdm.tqdm(image_files):

            try:
                label = label_from_filename(image_path.name)
                if exclude_classes:
                    if label in exclude_classes:
                        continue

                if nb_img_per_class[label] == MAX_IMG_PER_CLASS:
                    continue
                
                result = self.process_image_file(
                    image_path,
                    transform=transform,
                    save_dir=output_path,
                    apply_crf=apply_crf,
                    save_visualization=save_visualization
                )
                nb_img_per_class[label] += 1
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"Done! Results saved to {output_path}")


def main():
    import torchvision.transforms as T
    
    DATA_FOLDER = "mask_output/images"
    MODEL_PATH = "model_20260113-16:07/best_model.pth"  # Chemin vers votre modèle entraîné
    OUTPUT_FOLDER = "pseudo_masks_1"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_to_idx = {
        "Ignore": 0,
        "Necrosis": 1,
        "Stroma": 2,
        "Tumor": 3,
    }
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    print("Loading model...")
    gamma = 0.95
    model = CNN_PDA(num_classes=num_classes, gamma=gamma)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]) 

    generator = PseudoMaskGenerator(
        model=model,
        device=DEVICE,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        bg_threshold=0.20,
        fg_thresholds=[0.25, 0.35, 0.45, 0.55],
    )
    
    print("Generating pseudo-masks...")
    generator.process_dataset(
        data_folder=DATA_FOLDER,
        output_folder=OUTPUT_FOLDER,
        transform=transform,
        apply_crf=False,
        save_visualization=True,
        exclude_classes={"Region"}
    )
    
    print("Pseudo-mask generation complete!")


if __name__ == "__main__":
    main()
