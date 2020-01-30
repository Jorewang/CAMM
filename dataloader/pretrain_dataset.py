import csv
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class Dataset64(Dataset):
    def __init__(self, imsize, data_path, startidx=0):
        super(Dataset64, self).__init__()
        self.imsize = imsize
        self.data_path = data_path
        self.startidx = startidx
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                             transforms.RandomResizedCrop(imsize),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])

        self.path_images = os.path.join(data_path, 'miniimagenet', 'images')  # image path

        csvdata = self.loadCSV(os.path.join(data_path, 'miniimagenet', 'train.csv'))  # csv path
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.extend(v)
            self.img2label[k] = i + self.startidx
        self.num_classes = len(self.img2label)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename = self.data[index]
        label = self.img2label[filename[:9]]
        image = self.transform(os.path.join(self.path_images, filename))
        return image, label


if __name__ == '__main__':
    d = Dataset64(84, '../../data')
    loader = DataLoader(d, batch_size=5, shuffle=True)
    print(len(loader))
    for k, v in loader:
        print(v)
