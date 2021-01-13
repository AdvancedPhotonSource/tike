import numpy as np 

scanpoints = 5 # total number of scan points 
imagesize = 4 # means 4x4 image
grpsize = 4

# This means we have an image of size 4x4=16 pixels. 
# We group those into 8 bins such that eeach bin has 2 pixels (indices).
# We want all of those 2 pixels for each position to be unique. 

# # available frequencies
# freqs = np.zeros((scanpoints, imagesize*imagesize, 2))

# # target indexes for randomization
# numindexes = int(imagesize*imagesize / grpsize)
# idx = np.zeros((grpsize, scanpoints, numindexes))

# indices = np.arange(imagesize*imagesize)
# for m in range(scanpoints):
#     np.random.shuffle(indices)
#     for n in range(grpsize):
#         idx[n, m, :] = indices[n*numindexes:(n+1)*numindexes]
# print(idx[2].shape)
# # idx will provide a [8, 5, 2] matrix. 
# # We should call tike.ptycho.simulate in a for loop for each group like:
# # for n in range(grpsize):
# #     tike.ptycho.simulate(... , x=freqs(idx[n], :))

# visualization (optional for sanity checking)
# for n in range(grpsize):
#     print ('Indexes for group no: ' + str(n))
#     print (idx[n])
#     print (' ')

