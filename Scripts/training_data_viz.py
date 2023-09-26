#%%
from osgeo import gdal
import os
from osgeo.gdalconst import *
import numpy as np
#%%
# %% GDAL OPEN
location = '../Data/output/labels/mask_512_512.tif'
image = gdal.Open(location)
arr = np.array(image.GetRasterBand(1).ReadAsArray())
import matplotlib.pyplot as plt
plt.imshow(arr, cmap='Greys')
# %% opencv open
import cv2
from PIL import Image
import numpy as np

location = '../Data/output/labels/cambodia512_512_mask.tif'
mask = cv2.imread(location, cv2.IMREAD_UNCHANGED)
#mask = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask.shape
# %% removing 4th channel
#new_image = image[:,:,:3]
# %% Pillow open
from PIL import Image
import numpy as np

location = '../Data/output/labels/cambodia512_512_mask.tif'
mask = np.array(Image.open(location))
mask.shape
# %%
mask[mask==1.0]
# %%
