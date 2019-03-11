import os, random, string
from shutil import copyfile
from PIL import Image

home_folder = "dog_breeds_random"
labels_file = open("labels.csv", "w")
labels_file.write("id,breed\n")
dogs = []
id = 1
for root, dirs, files in os.walk("dog_breeds", topdown = False):
	for name in files:
		rand_name = str(id)
		id += 1
		if not name.endswith(".jpg"):
			im = Image.open(os.path.join(root, name))
			rgb_im = im.convert('RGB')
			rgb_im.save(os.path.join(home_folder, rand_name + ".jpg"))
		else:
			copyfile(os.path.join(root, name), os.path.join(home_folder, rand_name + ".jpg"))
		label = root.split("/")[1]
		dogs.append(rand_name + "," + label + "\n")

random.shuffle(dogs)

for dog_label in dogs:
	labels_file.write(dog_label)
labels_file.close()
