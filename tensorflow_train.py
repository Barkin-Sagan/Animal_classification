from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os 
from tensorflow.keras import layers,models
batch_size=64



TRAINING_DIR = "Animal_dataset/animals/train"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    )

VALIDATION_DIR = "Animal_dataset/animals/val"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=batch_size
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=batch_size
)



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(256, (3,3), activation="relu"))
model.add(layers.MaxPooling2D(2,2))




model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(5,activation="softmax"))

model.compile(optimizer= "rmsprop",
              loss = "categorical_crossentropy",
              metrics= ["accuracy"])






total_val=sum([len(files) for r, d, files in os.walk(VALIDATION_DIR)])


history = model.fit(train_generator, epochs=20, validation_data = validation_generator, verbose = 1,validation_steps=total_val//batch_size)