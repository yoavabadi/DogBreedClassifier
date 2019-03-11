import matplotlib.pyplot as plt
import pandas as pd

test_data = pd.read_csv('submission.csv')

for index, row in test_data.iterrows():
	image_path = "input/test/" + row['id'] + ".jpg"
	label = str(index) + ": " + row['max_label']
	im = plt.imread(image_path)
	plt.imshow(im)
	plt.title(label)
	plt.show()
	
