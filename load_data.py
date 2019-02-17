############################################################################
# Loads the BigEarth dataset (RGB Land-Cover Images) from the given files. #
############################################################################


##### DATASET LINKS
###
### UC-MERCED IMAGES   -->   http://weegee.vision.ucmerced.edu/datasets/landuse.html
###
### Multi-label annotations   -->   http://bigearth.eu/datasets.html


##### SOME NOTES
###
### We have 2100 available observations (256 x 256 x 3) - RGB.   (247 x 242 x 3 in reality)
###
### Total number of possible labels is 17. This dataset follows a
### multi-label classification scenario.


import numpy as np
import scipy.io as scio
import scipy.ndimage as im

np.set_printoptions(threshold=np.inf) # PRINT THE FULL NUMPY ARRAYS, NO MATTER WHAT THEIR SIZE IS


##################
#  MAIN PROGRAM  #
##################


# Load the labels
path_to_labels = '2_multilabels/LandUse_multilabels.mat' # CHANGE THIS PATH ACCORDINGLY

mat_labels   = scio.loadmat(path_to_labels)
numpy_labels = np.transpose( mat_labels['labels'] )
numpy_labels = np.array(numpy_labels, dtype=np.float32)

print( "The shape of the label matrix is {}.\n".format(numpy_labels.shape) )


# Load the images.
path_to_images = '1_UCMerced_LandUse/Images/'  # CHANGE THIS PATH ACCORDINGLY

image_types = [ 'agricultural',  'airplane',    'baseballdiamond', 'beach',   'buildings',          'chaparral',         'denseresidential',
                'forest',        'freeway',     'golfcourse',      'harbor',  'intersection',       'mediumresidential', 'mobilehomepark',
                'overpass',      'parkinglot',  'river',           'runway',  'sparseresidential',  'storagetanks',      'tenniscourt' ]

extension = '.tif'


images = []
associated_type  = []
associated_index = []

# In reality not all images are the same size. So we will crop them in the smallest dimensions possible.
x_limit = 247
y_limit = 242


for current_type in image_types:

    for index in range(100):

        str_index = str(index)

        if index < 10:
            str_index = '0' + str_index

        current_image = im.imread(path_to_images + current_type + '/' + current_type + str_index + extension)

        # Crop the images.
        remove_from_x = current_image.shape[0] - x_limit
        remove_from_y = current_image.shape[1] - y_limit
        current_image = current_image[remove_from_x:, remove_from_y:]

        images.append(current_image)
        associated_type.append(current_type)
        associated_index.append(index)


images = np.array(images, dtype=np.float32)
images = images / 255

associated_type  = np.array(associated_type)
associated_index = np.array(associated_index)

print( "The shape of the image samples matrix is {}.\n".format(images.shape) )


# Shuffle the data.
random_indices = np.arange( images.shape[0] )
np.random.shuffle(random_indices)

numpy_labels     = numpy_labels[random_indices]
images           = images[random_indices]
associated_type  = associated_type[random_indices]
associated_index = associated_index[random_indices]


# Pickle the data.
np.save("BigEarthImages.npy", images)
np.save("BigEarthLabels.npy", numpy_labels)
np.save("Assoc_Types.npy",    associated_type)
np.save("Assoc_Index.npy",    associated_index)
