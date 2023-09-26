#%% Cut large images into small pieces while retaining geospatial information
import os
from osgeo import gdal

def tifcut(input_dir, input_filename, output_dir, output_filename, tile_size_x,tile_size_y, stride_x, stride_y, image_type="mask"):
    """
    image_type = 'image' if you want to cut main images
    image_type = 'mask' if you want to cut masks for main images
    """
    ds = gdal.Open(input_dir + input_filename)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    for i in range(0, xsize, stride_x):
        for j in range(0, ysize, stride_y):
            com_string = f"gdal_translate -of GTIFF -srcwin {i}, {j}, {tile_size_x}, {tile_size_y} {input_dir}{input_filename} {output_dir}{output_filename}{int(i+stride_x)}_{int(j+stride_y)}_{image_type}.tif"
            os.system(com_string)


# %%
if __name__=="__main__":
    input_dir = '../Data/mask/'
    input_filename = 'cambodia_mask.tif'
    output_dir = '../Data/output/labels/'
    output_filename = 'cambodia'

    tile_size_x = 448
    tile_size_y = 448
    stride_x=448
    stride_y=448

    tifcut(input_dir, input_filename, output_dir, output_filename, tile_size_x,tile_size_y, stride_x, stride_y,image_type="mask")