from Model.unet_model import unet_model   #Use defined unet model
from keras.api.utils import normalize
from tensorflow.python import keras
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def run_model(training = False, epochs = 50, existing_data = False, n_training_elements = 5):

# Data initialization --------------------------------------------------------------------------------------------------

    train_image_directory = 'Dataset/train/'
    train_mask_directory = 'Dataset/train_masks/'
    valid_image_directory = 'Dataset/valid/'
    valid_mask_directory = 'Dataset/valid_masks/'
    test_image_directory = 'Dataset/test/'
    test_mask_directory = 'Dataset/test_masks/'

    def load_images(directory):
        data = []
        for image_name in os.listdir(directory):
            if image_name.lower().endswith('.jpg'):
                image_path = os.path.join(directory, image_name)
                image = cv2.imread(image_path, 0)  # Read in grayscale
                if image is not None:
                    image = cv2.resize(image, (256, 256))
                    image = Image.fromarray(image)
                    data.append(np.array(image))
        return data

    train_image_dataset = load_images(train_image_directory)
    train_mask_dataset = load_images(train_mask_directory)
    valid_image_dataset = load_images(valid_image_directory)
    valid_mask_dataset = load_images(valid_mask_directory)
    test_image_dataset = load_images(test_image_directory)
    test_mask_dataset = load_images(test_mask_directory)

    #Normalize images
    train_image_dataset = np.expand_dims(normalize(np.array(train_image_dataset), axis=1),3)
    valid_image_dataset = np.expand_dims(normalize(np.array(valid_image_dataset), axis=1),3)
    test_image_dataset = np.expand_dims(normalize(np.array(test_image_dataset), axis=1),3)

    #No normalization for masks, just rescaling to [0,1] (optional, but usefull)
    train_mask_dataset = np.expand_dims((np.array(train_mask_dataset)),3) /255.
    valid_mask_dataset = np.expand_dims((np.array(valid_mask_dataset)),3) /255.
    test_mask_dataset = np.expand_dims((np.array(test_mask_dataset)),3) /255.

# U-Net initialization -------------------------------------------------------------------------------------------------

    IMG_HEIGHT = train_image_dataset.shape[1]     #256
    IMG_WIDTH  = train_image_dataset.shape[2]     #256
    IMG_CHANNELS = train_image_dataset.shape[3]   #1

    def get_model():
        return unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model = get_model()

    if (not training) or existing_data:
        print("Loading model weights...")
        model.load_weights('Model/brain_tumor_model.keras')
        print("Model weights loaded")


    def save_model_checkpoint(epoch, logs):
        if logs['val_loss'] < save_model_checkpoint.best_val_loss:
            save_model_checkpoint.best_val_loss = logs['val_loss']
            model.save("Model/best_model.keras", overwrite=True)
            print('Model checkpoint saved.')

    save_model_checkpoint.best_val_loss = float('inf')

    early_stopping_callback = keras.models.training.callbacks_module.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        restore_best_weights=False,
        verbose=1
    )

    csv_logger_callback = keras.models.training.callbacks_module.CSVLogger(
        filename='Model/training_log.csv',
        separator=',',
        append=False
    )

    callbacks = [
        keras.models.training.callbacks_module.LambdaCallback(on_epoch_end = save_model_checkpoint),
        early_stopping_callback,
        csv_logger_callback
    ]

    history = []
    if training:
        history = model.fit(
            train_image_dataset,
            train_mask_dataset,
            batch_size = 16,
            epochs=epochs,
            validation_data=(valid_image_dataset, valid_mask_dataset),
            callbacks=callbacks
            )

        model.save('Model/brain_tumor_model.keras', overwrite=True)


# Evaluate the model ---------------------------------------------------------------------------------------------------

    if training:
        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], 'ro-')
        plt.plot(history.history['val_loss'], 'bo-')
        plt.title('Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.subplot(1,2,2)
        plt.plot(history.history['binary_accuracy'],'ro-')
        plt.plot(history.history['val_binary_accuracy'], 'bo-')
        plt.title('Training Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.plot(history.history['precision'], 'ro-')
        plt.plot(history.history['val_precision'], 'bo-')
        plt.title('Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.subplot(1,2,2)
        plt.plot(history.history['recall'],'ro-')
        plt.plot(history.history['val_recall'], 'bo-')
        plt.title('Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

    model.evaluate(test_image_dataset, test_mask_dataset)

# Predict on a few images ----------------------------------------------------------------------------------------------

    def test_random_images(n):
        for i in range(n):
            test_img_number = np.random.randint(0, len(test_image_dataset))
            test_img = test_image_dataset[test_img_number]
            ground_truth = test_mask_dataset[test_img_number]
            test_img_norm = test_img[:, :, 0][:, :, None]
            test_img_input = np.expand_dims(test_img_norm, 0)
            #Predict and threshold for pixels values above 0.2 probability
            prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.2).astype(np.uint8)

            plt.figure(figsize=(16, 8))
            plt.subplot(231)
            plt.title('Testing Image')
            plt.imshow(test_img[:, :, 0], cmap='gray')
            plt.subplot(232)
            plt.title('Testing Label')
            plt.imshow(ground_truth[:, :, 0], cmap='gray')
            plt.subplot(233)
            plt.title('Prediction on test image')
            plt.imshow(prediction, cmap='gray')
            plt.show()

    if not training:
        test_random_images(n_training_elements)