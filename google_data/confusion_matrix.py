
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

keys = ['affenpinscher',
 'afghan hound',
 'african hunting dog',
 'airedale',
 'american staffordshire terrier',
 'appenzeller',
 'australian terrier',
 'basenji',
 'basset',
 'beagle',
 'bedlington terrier',
 'bernese mountain dog',
 'black-and-tan coonhound',
 'blenheim spaniel',
 'bloodhound',
 'bluetick',
 'border collie',
 'border terrier',
 'borzoi',
 'boston bull',
 'bouvier des flandres',
 'boxer',
 'brabancon griffon',
 'briard',
 'brittany spaniel',
 'bull mastiff',
 'cairn',
 'cardigan',
 'chesapeake bay retriever',
 'chihuahua',
 'chow',
 'clumber',
 'cocker spaniel',
 'collie',
 'curly-coated retriever',
 'dandie dinmont',
 'dhole',
 'dingo',
 'doberman',
 'english foxhound',
 'english setter',
 'english springer',
 'entlebucher',
 'eskimo dog',
 'flat-coated retriever',
 'french bulldog',
 'german shepherd',
 'german short-haired pointer',
 'giant schnauzer',
 'golden retriever',
 'gordon setter',
 'great dane',
 'great pyrenees',
 'greater swiss mountain dog',
 'groenendael',
 'ibizan hound',
 'irish setter',
 'irish terrier',
 'irish water spaniel',
 'irish wolfhound',
 'italian greyhound',
 'japanese spaniel',
 'keeshond',
 'kelpie',
 'kerry blue terrier',
 'komondor',
 'kuvasz',
 'labrador retriever',
 'lakeland terrier',
 'leonberg',
 'lhasa',
 'malamute',
 'malinois',
 'maltese dog',
 'mexican hairless',
 'miniature pinscher',
 'miniature poodle',
 'miniature schnauzer',
 'newfoundland',
 'norfolk terrier',
 'norwegian elkhound',
 'norwich terrier',
 'old english sheepdog',
 'otterhound',
 'papillon',
 'pekinese',
 'pembroke',
 'pomeranian',
 'pug',
 'redbone',
 'rhodesian ridgeback',
 'rottweiler',
 'saint bernard',
 'saluki',
 'samoyed',
 'schipperke',
 'scotch terrier',
 'scottish deerhound',
 'sealyham terrier',
 'shetland sheepdog',
 'shih-tzu',
 'siberian husky',
 'silky terrier',
 'soft-coated wheaten terrier',
 'staffordshire bullterrier',
 'standard poodle',
 'standard schnauzer',
 'sussex spaniel',
 'tibetan mastiff',
 'tibetan terrier',
 'toy poodle',
 'toy terrier',
 'vizsla',
 'walker hound',
 'weimaraner',
 'welsh springer spaniel',
 'west highland white terrier',
 'whippet',
 'wire-haired fox terrier',
 'yorkshire terrier']


dict_idx = dict(zip(keys, [i for i in range(120)]))

results = pd.read_csv('submission_confusion.csv')

y_true = [dict_idx[cur_key] for cur_key in results['y_true'].values]
y_pred = [dict_idx[cur_key] for cur_key in results['y_pred'].values]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
np.savetxt("conf_mtx.csv", cnf_matrix, delimiter=",")

exit()
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=keys,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=keys, normalize=True,
                      title='Normalized confusion matrix')

plt.show()



