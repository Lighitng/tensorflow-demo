import matplotlib.pyplot as plt
import imageio,os
images = []
filenames=sorted((fn for fn in os.listdir('./') if fn.endswith('.png')))
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('./Series.gif', images, 'GIF', duration=0.1)

# import imageio, os
# imageio.imread(os.listdir('./predict_result_multi(1).png'))