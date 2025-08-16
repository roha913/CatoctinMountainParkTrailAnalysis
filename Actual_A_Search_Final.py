#%%
from PIL import Image
import numpy as np
from scipy.sparse import csr_matrix
from jax import jit
import jax.numpy as jnp
from jax import debug
from math import floor
import time
import warnings

def cost_func(p1, p2):
    return abs(p2[2] - p1[2])
class Graph:
    def __init__(self, file_path, vul_x, vul_y, vul_r):
        
        img_open = Image.open(file_path)
        arr_image = np.asarray(img_open)
        self.num_cols = len(arr_image[0])
        print(arr_image.shape)
        the_las_x = np.tile(np.arange(1, len(arr_image) + 1), len(arr_image[0]))
        the_las_y = np.repeat(np.arange(1, len(arr_image) + 1), len(arr_image[0]))
        the_las_z = (np.mean(arr_image, axis = 2)).flatten()
        coords = np.vstack((the_las_x, the_las_y, the_las_z)).transpose()
        #print(coords)
        #raise Exception()
        '''
        mask = np.logical_and(np.abs(coords[:, 0] -vul_x+ 77.4822085) < 0.02, np.abs(coords[:, 1] - 39.6348335) < 0.0049)#rectangular boundary
        coords = coords[mask]
        '''
        '''
        mX = np.min(coords[:, 0])
        maX = np.max(coords[:, 0])
        mY = np.min(coords[:, 1])
        rc_values = (coords[:, 1] - mY)*(maX - mX) + (coords[:, 0] - mX)
        coords = coords[rc_values.argsort()]
        '''

        
        coords = np.hstack((coords, np.arange(len(coords)).reshape(-1, 1)))
        ''''
        vulnerable_radius = (coords[:, 0] - vul_x)**2 + (coords[:, 1] - vul_y)**2 - vul_r**2
        
        #print(np.count_nonzero(vulnerable_radius == np.min(vulnerable_radius)))
        #print(np.min(vulnerable_radius))
        #ocl = len(coords)
        coords = coords[vulnerable_radius > 0]
        #print(ocl - len(coords))
        #print(coords.shape)
        '''

        '''
        precision = 10
        coords = coords.reshape((-1, precision, 3))
        coords = coords.mean(axis = 1)
        '''
        
        self.coords = coords
        self.x_coords = coords[:, 0]
        self.y_coords = coords[:, 1]
        self.z_coords = coords[:, 2]
        self.ind_coords = np.arange(len(self.z_coords))
        self.n_points = len(self.z_coords)
        self.inf = 10000000
        print("n_points = " + str(self.n_points))
        self.vul_x = vul_x
        self.vul_y = vul_y
        self.vul_r = vul_r

        self.min_z = np.min(self.z_coords)
        self.max_z = np.max(self.z_coords)    
    def A_star(self, start, goal, h):
        f_Scores = self.inf*np.ones(self.n_points)
        g_Scores = self.inf*np.ones(self.n_points)
        previous = -1*np.ones(self.n_points)
        explored = np.zeros(self.n_points)
        f_Scores[start] = h(self.coords[start], self.coords[goal])
        g_Scores[start] = 0
        for i in range(self.n_points):
            print(str(i) + "/" + str(self.n_points))
            unexplored_distances = f_Scores + self.inf*explored
            v = int(np.argmin(unexplored_distances))
            explored[v] = 1

            num_grid_cols = self.num_cols
            left_coor_ind = int(v- 1)
            right_coor_ind = int(v + 1)
            actual_coor_ind = v
            neighbors = []
            if(actual_coor_ind%num_grid_cols > 0 and explored[left_coor_ind] == 0):
                neighbors.append(left_coor_ind)
            if(actual_coor_ind%num_grid_cols < num_grid_cols - 1  and explored[right_coor_ind] == 0):
                neighbors.append(right_coor_ind)
            below_coor_ind = int(v + num_grid_cols)
            above_coor_ind = int(v - num_grid_cols)
            if(below_coor_ind < len(self.ind_coords) and explored[below_coor_ind] == 0):
                neighbors.append(below_coor_ind)
            if(above_coor_ind >= 0 and explored[above_coor_ind] == 0):
                neighbors.append(above_coor_ind)
            UL_coord = above_coor_ind - 1
            if(actual_coor_ind%num_grid_cols > 0 and UL_coord >= 0 and explored[UL_coord] == 0):
                neighbors.append(UL_coord)
            UR_coord = above_coor_ind + 1
            if(actual_coor_ind%num_grid_cols < num_grid_cols - 1 and UR_coord >= 0 and explored[UR_coord] == 0):
                neighbors.append(UR_coord)
            DL_coord = below_coor_ind - 1
            if(actual_coor_ind%num_grid_cols > 0 and DL_coord < self.n_points and explored[DL_coord] == 0):
                neighbors.append(DL_coord)
            DR_coord = below_coor_ind + 1
            if(actual_coor_ind%num_grid_cols < num_grid_cols - 1 and DR_coord < self.n_points and explored[DR_coord] == 0):
                neighbors.append(DR_coord)

            for u in neighbors:
                d = g_Scores[v] + cost_func(self.coords[v], self.coords[u])
                if d < g_Scores[u]:
                    g_Scores[u] = d
                    f_Scores[u] = d + h(self.coords[u], self.coords[goal])
                    previous[u] = v
                    
        optimal_path = []
        c3 = goal
        while(c3 > -1.0):
            optimal_path.append(c3)
            c3 = int(previous[c3])
        return optimal_path
def hTest(a_s, b):
    return np.zeros(len(a_s))

@jit
def h1(a, b):#TODO
    return a[2] - b[2]

print("About to create Graph object")
g = Graph("Downloads\OperationAntiJupyter\screenshot_las_file_with_circle.png", 0, 0, 1)
'''
print(np.min(g.z_coords))
print(np.max(g.z_coords))
raise Exception
'''
#finding start and end points --- assumes data is using NAD83
#print(np.unique(g.x_coords))
ind_start = 0#---!
print("ind_start = " + str(ind_start))
ind_end = g.n_points - 1#---!
print("ind_end = " + str(ind_end))
print("About to run A*")
optimal_path = g.A_star(ind_start, ind_end, h1)

#%%
print("About to print/write path")
L = len(optimal_path)
f = open("Downloads\OperationAntiJupyter\Actual_A_Star_PathWithCircle.txt", "w")
for i in range(L - 1, -1, -1):
    the_coord_in_question = str(g.coords[int(optimal_path[i])][0]) + "," + str(g.coords[int(optimal_path[i])][1]) + "," + str(g.coords[int(optimal_path[i])][2]) + "," + str(g.coords[int(optimal_path[i])][3]) + "\n"
    f.write(the_coord_in_question)
    #print(optimal_path[i])
f.close()

# %%
