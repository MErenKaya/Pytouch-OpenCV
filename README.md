Land Classification AI Project

This project is a **land cover classification system** using a deep learning model trained in PyTorch and deployed with OpenCV in C++.

---

## Project Structure

Land-Classification-AI/
│
├── python-training/        # PyTorch training code
├── cpp-opencv-app/         # C++ OpenCV inference app
├── model/                  # Exported ONNX model
├── sample-images/          # Test images
└── README.md

---

## Technologies Used

- Python (PyTorch)
- OpenCV (DNN module)
- C++
- ONNX (model export)
- Visual Studio / VS Code

---

## Model Classes

- AnnualCrop  
- Forest  
- Highway  
- River  

---

## How It Works

1. Model is trained in PyTorch (Python)
2. Exported to ONNX format
3. Loaded in C++ OpenCV application
4. Image is preprocessed and passed to model
5. Prediction is displayed on image

---

## Example Output

(Add screenshots of your results here)

---

## Model Path Example

```cpp
readNetFromONNX("model/model.onnx");
