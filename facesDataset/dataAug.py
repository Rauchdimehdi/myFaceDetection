from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import mahotas
import sys



# load the image
img_path = sys.argv[1]
save_path = img_path.split('/')[0]
img = load_img(img_path)

# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator

# datagen = ImageDataGenerator(brightness_range=[0.2,1.0],rotation_range=30)
datagen = ImageDataGenerator(zoom_range=[0.5,1.0])

# prepare iterator
it = datagen.flow(samples, batch_size=1)

# generate samples and plot
for i in range(10):
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	
	mahotas.imsave(f"{save_path}image{i}.png", image)

