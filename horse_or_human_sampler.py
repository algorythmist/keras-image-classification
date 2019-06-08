import os
from horse_or_human_loader import prepare_dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == '__main__':
    directory = prepare_dataset()

    # Directory with our training horse pictures
    train_horse_dir = os.path.join(directory, 'horses')

    # Directory with our training human pictures
    train_human_dir = os.path.join(directory, 'humans')

    train_horse_names = os.listdir(train_horse_dir)
    train_human_names = os.listdir(train_human_dir)

    # Parameters for our graph; we'll output iintermages in a 4x4 configuration
    nrows = 4
    ncols = 4

    # Index for iterating over images
    pic_index = 0

    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_horse_pix = [os.path.join(train_horse_dir, fname)
                      for fname in train_horse_names[pic_index - 8:pic_index]]
    next_human_pix = [os.path.join(train_human_dir, fname)
                      for fname in train_human_names[pic_index - 8:pic_index]]

    for i, img_path in enumerate(next_horse_pix + next_human_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()