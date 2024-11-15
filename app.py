# Save this as `waste_sorting_app.py` and run it with `streamlit run waste_sorting_app.py`

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Title and Description
st.title("Waste Sorting Model Training App")
st.write("This app trains a CNN model on the waste sorting dataset.")

# Dataset Directory Input
data_dir = st.text_input("Enter Dataset Directory Path", 'https://github.com/vikram3192/Waste-sorting-using-computer-vision/tree/e17c3e5bf2b1f2c88c07f14e91d658184fbc7602/dataset-resized')
img_size, batch_size = (180, 180), 32

# Prepare data generators
@st.cache_data  # Caches data so it doesn't reload every time
def prepare_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=15, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.1, zoom_range=0.2,
        horizontal_flip=True, validation_split=0.2
    )
    train_data = train_datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='training'
    )
    validation_data = train_datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='validation'
    )
    return train_data, validation_data

train_data, validation_data = prepare_data()

# Display Class Labels
class_labels = {v: k for k, v in train_data.class_indices.items()}
st.subheader("Class Labels:")
st.write(class_labels)

# Define model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(train_data.class_indices), activation='softmax')
    ])
    return model

model = create_model()

# Display Model Summary
st.subheader("Model Summary:")
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
st.text("\n".join(model_summary))

# Compile and Train the Model
if st.button("Start Training"):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_data, epochs=30, validation_data=validation_data, callbacks=[early_stop]
    )
    
    # Display training accuracy and loss
    st.subheader("Training Progress:")
    st.line_chart({"Training Accuracy": history.history["accuracy"], "Validation Accuracy": history.history["val_accuracy"]})
    st.line_chart({"Training Loss": history.history["loss"], "Validation Loss": history.history["val_loss"]})

    st.success("Training complete!")
