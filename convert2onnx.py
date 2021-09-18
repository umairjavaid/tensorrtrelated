import torch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=2)
model.load_state_dict(torch.load("/home/umair/Desktop/umair/genderClassification/merged_CEFL.pt"))
dummy_input = torch.randn(1, 3, 224, 224)

model.set_swish(memory_efficient=False)
torch.onnx.export(model, dummy_input, "gender_model1.onnx", verbose=True)