

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
try:
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    from tensorflow.keras.layers import BatchNormalization, Activation
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("TensorFlow/Keras not installed. Install with: pip install tensorflow")
    exit()

# Image processing
try:
    from PIL import Image
    import cv2
except ImportError:
    print("Install image processing libraries: pip install pillow opencv-python")
    exit()

# Set random seeds
np.random.seed(42)

class HandwrittenRecognitionModel:
    """
    Handwritten Character Recognition using CNNs
    Supports MNIST (digits) and EMNIST (characters)
    """
    
    def __init__(self, dataset='mnist', model_type='cnn'):
        """
        Initialize the handwritten recognition model
        
        Args:
            dataset (str): 'mnist' for digits or 'emnist' for characters
            model_type (str): 'cnn', 'deep_cnn', or 'advanced_cnn'
        """
        self.dataset = dataset
        self.model_type = model_type
        self.model = None
        self.history = None
        self.input_shape = (28, 28, 1)
        self.num_classes = 10 if dataset == 'mnist' else 62  # EMNIST has 62 classes
        
    def load_mnist_data(self):
        """
        Load MNIST dataset (handwritten digits 0-9)
        """
        print("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")
        print(f"Image shape: {X_train.shape[1:]}")
        
        return X_train, y_train, X_test, y_test
    
    def load_emnist_data(self):
        """
        Load EMNIST dataset (handwritten characters A-Z, a-z, 0-9)
        Note: EMNIST requires separate installation
        """
        try:
            from emnist import extract_training_samples, extract_test_samples
            
            print("Loading EMNIST dataset...")
            X_train, y_train = extract_training_samples('byclass')
            X_test, y_test = extract_test_samples('byclass')
            
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Testing samples: {X_test.shape[0]}")
            print(f"Number of classes: {len(np.unique(y_train))}")
            
            return X_train, y_train, X_test, y_test
            
        except ImportError:
            print("EMNIST not installed. Using MNIST instead.")
            print("To use EMNIST: pip install emnist")
            return self.load_mnist_data()
    
    def preprocess_data(self, X_train, y_train, X_test, y_test):
        """
        Preprocess image data
        """
        print("\nPreprocessing data...")
        
        # Reshape to add channel dimension
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        
        # Normalize pixel values to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # One-hot encode labels
        y_train_encoded = to_categorical(y_train, self.num_classes)
        y_test_encoded = to_categorical(y_test, self.num_classes)
        
        print(f"Preprocessed training shape: {X_train.shape}")
        print(f"Preprocessed labels shape: {y_train_encoded.shape}")
        
        return X_train, y_train_encoded, X_test, y_test_encoded, y_train, y_test
    
    def create_data_augmentation(self):
        """
        Create data augmentation generator for improved training
        """
        datagen = ImageDataGenerator(
            rotation_range=10,      # Random rotation
            zoom_range=0.1,         # Random zoom
            width_shift_range=0.1,  # Random horizontal shift
            height_shift_range=0.1  # Random vertical shift
        )
        
        return datagen
    
    def build_basic_cnn(self):
        """
        Build a basic CNN model
        """
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_deep_cnn(self):
        """
        Build a deeper CNN model with batch normalization
        """
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(32, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Conv2D(64, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_advanced_cnn(self):
        """
        Build an advanced CNN with residual-like connections
        """
        model = Sequential([
            # Block 1
            Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(32, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            
            # Block 2
            Conv2D(64, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
            
            # Block 3
            Conv2D(128, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(128, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.4),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_model(self):
        """
        Build model based on specified type
        """
        print(f"\nBuilding {self.model_type} model...")
        
        if self.model_type == 'cnn':
            model = self.build_basic_cnn()
        elif self.model_type == 'deep_cnn':
            model = self.build_deep_cnn()
        elif self.model_type == 'advanced_cnn':
            model = self.build_advanced_cnn()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=30, batch_size=128, use_augmentation=True):
        """
        Train the handwritten recognition model
        """
        print("\n" + "="*70)
        print(f"TRAINING {self.model_type.upper()} MODEL ON {self.dataset.upper()}")
        print("="*70)
        
        # Build model
        self.model = self.build_model()
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            f'{self.dataset}_{self.model_type}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Data augmentation
        if use_augmentation:
            print("\nUsing data augmentation...")
            datagen = self.create_data_augmentation()
            datagen.fit(X_train)
            
            # Train with augmentation
            self.history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                validation_data=(X_test, y_test),
                epochs=epochs,
                callbacks=[early_stop, reduce_lr, checkpoint],
                verbose=1
            )
        else:
            # Train without augmentation
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr, checkpoint],
                verbose=1
            )
        
        print("\nTraining completed!")
        
        return self.history
    
    def evaluate(self, X_test, y_test, y_test_original):
        """
        Evaluate model performance
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Predictions
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test_original, y_pred_classes)
        
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        if self.dataset == 'mnist':
            target_names = [str(i) for i in range(10)]
        else:
            target_names = [str(i) for i in range(self.num_classes)]
        
        print(classification_report(y_test_original, y_pred_classes, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_original, y_pred_classes)
        
        return accuracy, cm, y_pred_classes
    
    def visualize_predictions(self, X_test, y_test_original, y_pred, num_samples=25):
        """
        Visualize sample predictions
        """
        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        
        # Randomly select samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        for idx, ax in enumerate(axes.flat):
            i = indices[idx]
            
            # Display image
            ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
            
            # Title with prediction and actual
            true_label = y_test_original[i]
            pred_label = y_pred[i]
            
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset}_predictions.png', dpi=300, bbox_inches='tight')
        print(f"\nPrediction samples saved as '{self.dataset}_predictions.png'")
        plt.show()
    
    def plot_training_history(self):
        """
        Plot training history
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset}_training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training history saved as '{self.dataset}_training_history.png'")
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(12, 10))
        
        if self.dataset == 'mnist':
            labels = [str(i) for i in range(10)]
        else:
            # For EMNIST, show subset if too many classes
            labels = [str(i) for i in range(min(self.num_classes, 20))]
            cm = cm[:20, :20]  # Show only first 20 classes for visibility
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {self.dataset.upper()}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved as '{self.dataset}_confusion_matrix.png'")
        plt.show()
    
    def predict_from_image(self, image_path):
        """
        Predict digit/character from custom image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28
            img_array = np.array(img)
            
            # Invert if needed (MNIST uses white on black)
            if img_array.mean() > 127:
                img_array = 255 - img_array
            
            # Normalize
            img_array = img_array.astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # Predict
            prediction = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][predicted_class]
            
            result = {
                'predicted_class': int(predicted_class),
                'confidence': float(confidence),
                'all_probabilities': {i: float(prediction[0][i]) for i in range(self.num_classes)}
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def visualize_filters(self, layer_idx=0):
        """
        Visualize learned convolutional filters
        """
        # Get first conv layer
        layer = self.model.layers[layer_idx]
        filters, biases = layer.get_weights()
        
        # Normalize filters
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        
        # Plot filters
        n_filters = min(filters.shape[3], 32)  # Show max 32 filters
        n_cols = 8
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 1.5))
        
        for i in range(n_filters):
            ax = axes[i // n_cols, i % n_cols]
            ax.imshow(filters[:, :, 0, i], cmap='viridis')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_filters, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].axis('off')
        
        plt.suptitle(f'Learned Convolutional Filters - Layer {layer_idx}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.dataset}_filters.png', dpi=300, bbox_inches='tight')
        print(f"Filter visualization saved as '{self.dataset}_filters.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*80)
    print("HANDWRITTEN CHARACTER RECOGNITION - CODEALPHA ML INTERNSHIP")
    print("="*80)
    
    # Configuration
    dataset = 'mnist'  # Change to 'emnist' for character recognition
    model_types = ['cnn', 'deep_cnn']  # Compare different architectures
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*80}")
        
        # Initialize model
        model = HandwrittenRecognitionModel(dataset=dataset, model_type=model_type)
        
        # Load data
        if dataset == 'mnist':
            X_train, y_train, X_test, y_test = model.load_mnist_data()
        else:
            X_train, y_train, X_test, y_test = model.load_emnist_data()
        
        # Preprocess
        X_train_prep, y_train_prep, X_test_prep, y_test_prep, y_train_orig, y_test_orig = \
            model.preprocess_data(X_train, y_train, X_test, y_test)
        
        # Train
        history = model.train(X_train_prep, y_train_prep, X_test_prep, y_test_prep, 
                             epochs=20, batch_size=128, use_augmentation=True)
        
        # Evaluate
        accuracy, cm, y_pred = model.evaluate(X_test_prep, y_test_prep, y_test_orig)
        
        # Store results
        results[model_type] = {
            'model': model,
            'accuracy': accuracy,
            'confusion_matrix': cm
        }
        
        # Visualizations
        model.plot_training_history()
        model.plot_confusion_matrix(cm)
        model.visualize_predictions(X_test_prep, y_test_orig, y_pred, num_samples=25)
        
        if model_type == 'deep_cnn':
            model.visualize_filters(layer_idx=0)
    
    # Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    for model_type, data in results.items():
        print(f"{model_type.upper():20} - Accuracy: {data['accuracy']:.4f} ({data['accuracy']*100:.2f}%)")
    
    # Find best model
    best_model_type = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nBest Model: {best_model_type.upper()} with accuracy {results[best_model_type]['accuracy']:.4f}")
    
    # Example prediction
    print("\n" + "="*80)
    print("EXAMPLE: HOW TO USE FOR CUSTOM IMAGES")
    print("="*80)
    print("""
To predict from your own handwritten image:

1. Draw a digit/character on white paper
2. Take a photo or scan it
3. Use the predict_from_image() method:

    result = model.predict_from_image('path/to/your/image.png')
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    """)
    
    print("\n" + "="*80)
    print("TASK COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nDataset: {dataset.upper()}")
    print(f"Best Model: {best_model_type.upper()}")
    print(f"Best Accuracy: {results[best_model_type]['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
