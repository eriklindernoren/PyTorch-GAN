import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch, sample_interval, n_samples=5):
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.sample_interval = sample_interval
        self.batches_done = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.n_samples = n_samples
        self.past_images = []

    def log(self, losses=None, images=None, epoch=0, batch=0):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        epoch += 1
        batch += 1

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (epoch, self.n_epochs, batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = 0

            self.losses[loss_name] += losses[loss_name].data[0]

            sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batches_done))

        batches_left = self.batches_epoch*(self.n_epochs - epoch) + self.batches_epoch - batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/self.batches_done)))

        # Save image sample
        image_sample = torch.cat((images['real_B'].data, images['fake_A'].data, images['real_A'].data), -2)
        self.past_images.append(image_sample)
        if len(self.past_images) > self.n_samples:
            self.past_images.pop(0)


        # If at sample interval save past samples
        if self.batches_done % self.sample_interval == 0 and images is not None:
            save_image(torch.cat(self.past_images, -1),
                        './images/%d.png' % self.batches_done,
                        normalize=True)

        self.batches_done += 1


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
