import torch
import torch.nn as nn
from model import get_model

def convert():
    model = get_model(num_classes=4)
    
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(model, 
                      dummy_input, 
                      "model.onnx", 
                      opset_version=11, 
                      input_names=['input'], 
                      output_names=['output'])

    print("model.onnx olusturuldu")

if __name__ == "__main__":
    convert()