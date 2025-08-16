# %%
from PIL import Image
import numpy as np
import pyproj
img = Image.open("screenshot_las_file_with_circle.png")
arr_image = np.asarray(img)
num_cols = len(arr_image[0])
num_rows = len(arr_image)
print(num_cols)
print(arr_image.shape)
crs_wgs = pyproj.Proj(init='epsg:26918')
crs_bng = pyproj.Proj(init='epsg:4326')
wgs_line = ""
with open("Actual_A_Star_PathWithCircle.txt", 'r') as file:
    for line in file:
        L = line.strip().split(",")
        z = float(L[2])
        #print(L)
        ind = int(float(L[3]))
        r = (num_rows - int(ind/num_cols))*0.00000878449929923 + 39.635106
        c = (ind%(int(num_cols)))*0.0000178813932318 + -77.479773
        z = z*1.34443921569 + 237.953
        print((r, c, z))
        wgs_pt = pyproj.transform(crs_bng, crs_wgs, c, r, always_xy=True)
        west_pt = wgs_pt[0]
        north_pt = wgs_pt[1]
        wgs_line += str(north_pt) + "," + str(west_pt) + "," + str(z) + '\n'

with open("wgs_shifted_file_test_with_circle.txt", 'w') as file:
    file.write(wgs_line)
# %%
