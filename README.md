# Handwritten Character Recognition - CodeAlpha ML Internship

**Task 3:** Identify handwritten characters or alphabets using image processing and deep learning with CNNs.

## 📋 Project Overview

This project implements a handwritten character recognition system using Convolutional Neural Networks (CNNs) on MNIST (digits) and EMNIST (characters) datasets. The model can recognize handwritten digits 0-9 and can be extended to full alphabets.

## 🎯 Objective

Build and compare multiple CNN architectures to accurately recognize handwritten characters with high accuracy, demonstrating the power of deep learning in computer vision.

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV & PIL** - Image processing
- **numpy & pandas** - Data manipulation
- **matplotlib & seaborn** - Visualization
- **scikit-learn** - Model evaluation

## 🖼️ Datasets

### 1. **MNIST** (Modified National Institute of Standards and Technology)
- **Size**: 70,000 images (60,000 train + 10,000 test)
- **Classes**: 10 digits (0-9)
- **Image Size**: 28×28 pixels, grayscale
- **Built-in**: Automatically loaded with Keras
- **Accuracy Achievable**: 99%+

### 2. **EMNIST** (Extended MNIST)
- **Size**: 814,255 images
- **Classes**: 62 characters (0-9, A-Z, a-z)
- **Image Size**: 28×28 pixels, grayscale
- **Installation**: `pip install emnist`
- **Accuracy Achievable**: 85-90%

## 🧠 CNN Architectures Implemented

### 1. **Basic CNN**
```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(128) → Output
```
- **Parameters**: ~300K
- **Accuracy**: 98-99% (MNIST)
- **Training Time**: ~3 minutes

### 2. **Deep CNN**
```
2×Conv2D(32) → MaxPool → 2×Conv2D(64) → MaxPool → Dense(256+128) → Output
+ Batch Normalization + Dropout
```
- **Parameters**: ~500K
- **Accuracy**: 99%+ (MNIST)
- **Training Time**: ~5 minutes

### 3. **Advanced CNN**
```
3 Blocks of 2×Conv2D → MaxPool with BatchNorm & Dropout
Dense(512+256) → Output
```
- **Parameters**: ~1M
- **Accuracy**: 99.5%+ (MNIST)
- **Training Time**: ~8 minutes

## 📁 Project Structure

```
CodeAlpha_HandwrittenRecognition/
│
├── handwritten_recognition.py    # Main implementation
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── models/                        # Saved models
│   ├── mnist_cnn_best.h5
│   ├── mnist_deep_cnn_best.h5
│   └── emnist_advanced_cnn_best.h5
├── results/                       # Visualizations
│   ├── mnist_predictions.png
│   ├── mnist_training_history.png
│   ├── mnist_confusion_matrix.png
│   └── mnist_filters.png
└── custom_images/                 # Test your own images
```

## 🚀 How to Run

### Option 1: Google Colab (Easiest - No Setup Required!)

```python
# 1. Open Google Colab: colab.research.google.com
# 2. Create new notebook
# 3. Copy the handwritten_recognition.py code
# 4. Run! MNIST is automatically downloaded

# The script will:
# - Download MNIST dataset automatically
# - Train 2 CNN models
# - Generate visualizations
# - Compare results
```

### Option 2: Local Machine

```bash
# Clone repository
git clone https://github.com/yourusername/CodeAlpha_HandwrittenRecognition.git
cd CodeAlpha_HandwrittenRecognition

# Install dependencies
pip install -r requirements.txt

# Run the model
python handwritten_recognition.py
```

## 📊 Expected Results

### MNIST Results:

| Model | Accuracy | Parameters | Training Time |
|-------|----------|------------|---------------|
| Basic CNN | 98.5% | ~300K | ~3 min |
| Deep CNN | 99.2% | ~500K | ~5 min |
| **Advanced CNN** | **99.5%+** | ~1M | ~8 min |

### EMNIST Results:

| Model | Accuracy | Classes | Training Time |
|-------|----------|---------|---------------|
| Basic CNN | 82% | 62 | ~15 min |
| Deep CNN | 87% | 62 | ~20 min |
| **Advanced CNN** | **90%+** | 62 | ~30 min |

## 🎬 Usage Examples

### Training the Model:

```python
from handwritten_recognition import HandwrittenRecognitionModel

# Initialize for MNIST (digits)
model = HandwrittenRecognitionModel(dataset='mnist', model_type='deep_cnn')

# Load and preprocess data
X_train, y_train, X_test, y_test = model.load_mnist_data()
X_train_prep, y_train_prep, X_test_prep, y_test_prep, _, _ = \
    model.preprocess_data(X_train, y_train, X_test, y_test)

# Train
model.train(X_train_prep, y_train_prep, X_test_prep, y_test_prep, 
           epochs=20, batch_size=128)

# Evaluate
accuracy, cm, predictions = model.evaluate(X_test_prep, y_test_prep, y_test)

# Visualize results
model.plot_training_history()
model.plot_confusion_matrix(cm)
```

### Predicting from Custom Image:

```python
# Predict digit/character from your own handwriting
result = model.predict_from_image('path/to/your/digit.png')

print(f"Predicted Digit: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All Probabilities: {result['all_probabilities']}")
```

### Switching to EMNIST (Characters):

```python
# Initialize for EMNIST (0-9, A-Z, a-z)
model = HandwrittenRecognitionModel(dataset='emnist', model_type='advanced_cnn')

# Everything else remains the same!
X_train, y_train, X_test, y_test = model.load_emnist_data()
# ... continue training as before
```

## 🎨 Image Requirements for Prediction

To test with your own handwritten images:

1. **Format**: PNG, JPG, or any common image format
2. **Background**: White background, black ink (or will be inverted)
3. **Content**: Single digit/character centered
4. **Size**: Any size (will be resized to 28×28)
5. **Quality**: Clear, no noise if possible

**Example preparation:**
```python
# The model automatically:
# - Converts to grayscale
# - Resizes to 28×28
# - Inverts colors if needed
# - Normalizes pixel values
```

## 🔍 Key Features

✅ **Multiple CNN Architectures** - Compare basic, deep, and advanced models  
✅ **Data Augmentation** - Rotation, zoom, shift for better generalization  
✅ **Batch Normalization** - Faster training, better accuracy  
✅ **Dropout Regularization** - Prevents overfitting  
✅ **Learning Rate Scheduling** - Adaptive learning  
✅ **Early Stopping** - Prevents overtraining  
✅ **Model Checkpointing** - Saves best model automatically  
✅ **Comprehensive Visualization** - Training curves, confusion matrices, filters  
✅ **Custom Image Prediction** - Test your own handwriting  

## 📈 Training Improvements Used

1. **Data Augmentation**: Rotation, zoom, shift → +1-2% accuracy
2. **Batch Normalization**: Faster convergence, +0.5-1% accuracy
3. **Dropout**: Reduces overfitting, better generalization
4. **Adam Optimizer**: Adaptive learning rates
5. **Learning Rate Reduction**: When validation loss plateaus
6. **Early Stopping**: Prevents overtraining

## 🎓 Learning Outcomes

- Image preprocessing and normalization
- Convolutional Neural Networks (CNNs)
- Understanding filters and feature maps
- Batch normalization and dropout
- Data augmentation techniques
- Model evaluation and comparison
- Transfer learning concepts
- Real-world deployment considerations

## 🔧 Advanced Usage

### Visualize What the Network Learned:

```python
# See convolutional filters
model.visualize_filters(layer_idx=0)

# Visualize predictions with correct/incorrect labels
model.visualize_predictions(X_test, y_test, predictions, num_samples=25)
```

### Fine-tune Hyperparameters:

```python
# Experiment with different configurations
model.train(
    X_train, y_train, X_test, y_test,
    epochs=50,          # More epochs
    batch_size=64,      # Smaller batches
    use_augmentation=True  # Enable augmentation
)
```

### Save and Load Models:

```python
# Models are automatically saved during training as:
# mnist_deep_cnn_best.h5

# Load saved model
from tensorflow.keras.models import load_model
loaded_model = load_model('mnist_deep_cnn_best.h5')
```

## 🐛 Troubleshooting

### Common Issues:

**1. TensorFlow installation fails:**
```bash
# Use pip with specific version
pip install tensorflow==2.13.0

# Or for Apple Silicon Macs:
pip install tensorflow-macos tensorflow-metal
```

**2. EMNIST not loading:**
```bash
# EMNIST requires separate package
pip install emnist

# Or use MNIST only
dataset = 'mnist'  # Change in code
```

**3. Out of memory errors:**
```python
# Reduce batch size
batch_size = 32  # Instead of 128

# Or reduce model size
model_type = 'cnn'  # Instead of 'advanced_cnn'
```

**4. Custom image predictions incorrect:**
- Ensure white background, black text
- Center the digit/character
- Avoid extra markings or noise
- Try inverting colors if needed

## 📊 Confusion Matrix Interpretation

The confusion matrix shows:
- **Diagonal**: Correct predictions (darker = better)
- **Off-diagonal**: Misclassifications
- **Common errors**: 
  - 4 ↔ 9 (similar shapes)
  - 5 ↔ 6 (similar curves)
  - 7 ↔ 1 (similar lines)

## 🚀 Extensions & Future Work

1. **Word Recognition**: Extend to full words using CRNNs
2. **Sentence Recognition**: Use sequence-to-sequence models
3. **Real-time Recognition**: Deploy with webcam input
4. **Mobile App**: TensorFlow Lite for mobile deployment
5. **Multi-language**: Support for other scripts (Arabic, Chinese, etc.)

## 🤝 Contributing

This is an internship project. Feedback and suggestions welcome!

## 📧 Contact

M.KAVYA

Email:kavyamrkml@gmail.com 

## 📄 License

Part of CodeAlpha ML Internship Program - December 2025

---

## 🎯 Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run main script: `python handwritten_recognition.py`
- [ ] Check generated visualizations in results folder
- [ ] Try predicting your own handwritten digits
- [ ] Experiment with different model architectures
- [ ] Upload to GitHub: `CodeAlpha_HandwrittenRecognition`
- [ ] Create LinkedIn video demonstration
- [ ] Submit to CodeAlpha

---

## 📚 References

- LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition"
- Cohen, G., et al. (2017). "EMNIST: Extending MNIST to handwritten letters"
- Keras Documentation: https://keras.io/examples/vision/mnist_convnet/

---

**Happy Recognizing! ✍️🔍**
