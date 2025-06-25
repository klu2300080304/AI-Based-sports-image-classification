
# 🏅 100 Sports Image Classification using Custom CNNs

This project performs image classification on the [100 Sports Image Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification) from Kaggle using custom Convolutional Neural Networks (CNNs) built with TensorFlow/Keras.

![sports-classification-banner](https://storage.googleapis.com/kaggle-datasets-images/1468923/2422107/3c69cf52b393cd7e79b429b2a7628d9f/dataset-cover.jpg)

## 📂 Dataset

The dataset contains:
- 100 different sport categories
- ~13797 images
- Varying image sizes and backgrounds

📥 **Download the dataset here**:  
👉 [Kaggle Dataset - gpiosenka/sports-classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification)

## 📌 Project Structure

```

sports-image-classification/
│
├── data/                     # Contains training and testing images (after extraction)
├── models/                   # Saved trained models
├── notebooks/
│   └── EDA\_and\_Modeling.ipynb  # Exploratory analysis + model training
├── src/
│   ├── dataloader.py         # Custom data loading and preprocessing
│   ├── model.py              # Custom CNN architecture
│   └── train.py              # Training and evaluation script
├── README.md
└── requirements.txt

````

## 🧠 Model Architecture

The custom CNN is designed from scratch and may include:

- Multiple Conv2D + ReLU + MaxPooling layers
- Dropout layers to prevent overfitting
- Batch Normalization
- Dense output layer with 100 units (for 100 classes) + softmax

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(100, activation='softmax')
])
````

## 📊 Evaluation Metrics

* **Accuracy**
* **Confusion Matrix**
* **Top-5 Accuracy** (optional)
* **Classification Report**

## 📈 Sample Results

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | \~94% |
| Validation Accuracy | \~88% |
| Test Accuracy       | \~87% |

> 📌 *Performance varies based on number of epochs, image size, data augmentations, and architecture depth.*

## 🛠️ How to Run

1. **Clone the repo**:

   ```bash
   git clone https://github.com/yourusername/sports-image-classification.git
   cd sports-image-classification
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download and unzip the dataset**:

   * Download from [Kaggle Dataset Link](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
   * Extract into the `data/` directory

4. **Train the model**:

   ```bash
   python src/train.py
   ```

5. **Run the notebook**:

   ```bash
   jupyter notebook notebooks/EDA_and_Modeling.ipynb
   ```

## 🧪 Future Improvements

* Transfer Learning using ResNet, EfficientNet
* Data augmentation
* Early stopping & learning rate scheduling
* Deployment with Streamlit/Gradio

## 📚 References

* [Keras Documentation](https://keras.io/)
* [Kaggle Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
* [CNN Architectures](https://cs231n.github.io/convolutional-networks/)


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


