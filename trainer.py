import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# tf.__version__

#for training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2)


training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

#for test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

cnn = tf.keras.models.Sequential()
print(training_set.class_indices)
#convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation='relu',input_shape=[64,64,3]))#input shape[64,64,1] for B/W images
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

#2nd layer
#cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation='relu'))#input shape[64,64,1] for B/W images
#cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#3rd layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation='relu'))#input shape[64,64,1] for B/W images
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


#flattening
cnn.add(tf.keras.layers.Flatten())

#Fully connected layer
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

#ouput layer
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#training
cnn.fit(x=training_set,validation_data=test_set,epochs=3)

#cnn.summary()

cnn.save('model.h5')