# -*- coding: utf-8 -*-

# Torch imports

# In package imports

# Package imports


# ========================================================================
class AverageMeter(object):
    # Computes and stores average and ucrent values
    def __init__(self):
        self.reset()

    def reset(self):
    # Resets values to 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.contents = []

    def update(self, val, n=1):
    # Updates by one (or n) step(s) the contents
        try:
            val = val.cpu() 
        except AttributeError:
            val = val
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum/self.count
        self.contents.append(val.item())
