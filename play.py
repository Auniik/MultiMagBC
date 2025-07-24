

import json
import torch

from config import SLIDES_PATH
from preprocess.kfold_splitter import PatientWiseKFoldSplitter

# from backbones.our.model import MMNet
# from evaluate.gradcam import GradCAMExtractor


# dummy_batch = {
#     f'mag_{mag}': torch.randn(2, 3, 224, 224) for mag in ['40', '100', '200', '400']
# }
# model = MMNet().to('cpu')
# output = model(dummy_batch)
# print(output.shape)

# importance = model.get_magnification_importance()
# print("Magnification Importance Scores:", importance)

# # Example usage
# model.eval()
# cam = GradCAMExtractor(model, target_layer='extractors.extractor_40x.conv_head')  # Adjust the layer name based on actual structure

# output = model(dummy_batch)
# class_idx = output.argmax(dim=1).item()
# loss = output[:, class_idx].sum()
# loss.backward()

# gradcam_map = cam.get_gradcam()

# print("Grad-CAM Map Shape:", gradcam_map.shape)

splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        stratify_subtype=False
    )

# print(json.dumps(splitter.patient_dict, indent=2))

print(len(splitter.patient_dict))
splitter.print_summary()


splitter.visualize()
